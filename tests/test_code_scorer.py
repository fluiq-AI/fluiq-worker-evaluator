"""Code scorer: does it compute the right answer, and can it escape?

The second question matters more. Customer code runs inside the shared evaluator
worker, which holds database credentials and other orgs' data in the same
process, so the escape tests below are the ones that must never be quietly
deleted to make a feature work.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.helper.code_scorer import (  # noqa: E402
    MAX_COLLECTION,
    ScorerError,
    compile_scorer,
    run_scorer,
)


def score(src, **kw):
    return run_scorer(src, **kw)[0]


# ══ Escape attempts ══════════════════════════════════════════════════════════
# Each of these is a real technique used to break out of a Python sandbox.

def test_blocks_the_classic_subclasses_escape():
    """().__class__.__bases__[0].__subclasses__() is how every naive eval sandbox
    falls. It must fail at compile time, not at run time."""
    with pytest.raises(ScorerError):
        compile_scorer("().__class__.__bases__[0].__subclasses__()")


def test_blocks_dunder_attribute_on_a_scope_value():
    with pytest.raises(ScorerError):
        compile_scorer("output.__class__")


def test_blocks_globals_access():
    with pytest.raises(ScorerError):
        compile_scorer("contains.__globals__")


def test_blocks_import():
    with pytest.raises(ScorerError):
        compile_scorer("__import__('os')")


def test_blocks_import_statement_syntax():
    # `import os` is a statement, so mode="eval" rejects it as a syntax error —
    # asserted so a future move to mode="exec" cannot pass silently.
    with pytest.raises(ScorerError):
        compile_scorer("import os")


def test_blocks_builtins_lookup():
    with pytest.raises(ScorerError):
        compile_scorer("__builtins__")


def test_blocks_open():
    with pytest.raises(ScorerError):
        compile_scorer("open('/etc/passwd')")


def test_blocks_eval_and_exec():
    for src in ("eval('1')", "exec('x=1')", "compile('1','','eval')"):
        with pytest.raises(ScorerError):
            compile_scorer(src)


def test_blocks_getattr_as_an_attribute_backdoor():
    """getattr() would re-open every door the Attribute rule closes."""
    with pytest.raises(ScorerError):
        compile_scorer("getattr(output, '__class__')")


def test_blocks_lambda():
    with pytest.raises(ScorerError):
        compile_scorer("(lambda: 1)()")


def test_blocks_walrus_assignment():
    with pytest.raises(ScorerError):
        compile_scorer("(x := 1)")


def test_blocks_private_attributes():
    with pytest.raises(ScorerError):
        compile_scorer("output._secret()")


def test_blocks_str_format_which_can_reach_attributes():
    """"{0.__class__}".format(x) reaches an attribute without an Attribute node
    on the argument, so format has to be off the method allowlist."""
    with pytest.raises(ScorerError):
        compile_scorer("'{0.__class__}'.format(output)")


def test_blocks_format_map():
    with pytest.raises(ScorerError):
        compile_scorer("'{a}'.format_map(metadata)")


def test_blocks_calling_the_result_of_an_expression():
    """Only a plain name or an allowed method can be called. Calling whatever an
    expression evaluates to would let a value smuggled through a collection be
    invoked. (Parentheses alone don't count — `(f)()` is still just a Name.)"""
    for src in ("[contains][0]('a', 'b')", "(contains if True else contains)('a','b')"):
        with pytest.raises(ScorerError, match="Only named functions"):
            compile_scorer(src)


def test_blocks_argument_unpacking():
    with pytest.raises(ScorerError):
        compile_scorer("max(*[1, 2])")


def test_blocks_unknown_function():
    with pytest.raises(ScorerError, match="not a known function"):
        compile_scorer("evil(output)")


def test_blocks_undefined_name_at_runtime():
    with pytest.raises(ScorerError, match="not defined"):
        run_scorer("mystery > 1")


def test_blocks_async_comprehensions():
    """`ast.parse` accepts this outside an async function — the rule that forbids
    it is a compile-time check we never reach — so validation has to catch it."""
    with pytest.raises(ScorerError, match="Async"):
        compile_scorer("[x async for x in output]")


def test_blocks_yield():
    with pytest.raises(ScorerError):
        compile_scorer("(yield 1)")


# ══ Resource bounds ══════════════════════════════════════════════════════════

def test_bounds_exponentiation():
    """10**10**10 hangs a worker on one line."""
    with pytest.raises(ScorerError, match="Exponent"):
        run_scorer("10 ** 10 ** 10 > 0")


def test_allows_reasonable_exponentiation():
    assert score("1.0 if 2 ** 8 == 256 else 0.0") == 1.0


def test_bounds_range():
    with pytest.raises(ScorerError, match="longer than"):
        run_scorer(f"len(range({MAX_COLLECTION + 1})) > 0")


def test_bounds_comprehension_output():
    with pytest.raises(ScorerError):
        run_scorer(f"len([x for x in range({MAX_COLLECTION}) for y in range(100)]) > 0")


def test_bounds_string_growth():
    with pytest.raises(ScorerError, match="string"):
        run_scorer("len('a' * 300000) > 0")


def test_bounds_source_size():
    with pytest.raises(ScorerError, match="exceeds"):
        compile_scorer("1 + " * 5000 + "1")


def test_reports_division_by_zero_as_an_author_error():
    with pytest.raises(ScorerError, match="Division by zero"):
        run_scorer("1 / 0")


# ══ The return contract ══════════════════════════════════════════════════════

def test_boolean_true_is_a_perfect_score():
    assert score("True") == 1.0


def test_boolean_false_is_zero():
    assert score("False") == 0.0


def test_number_in_range_passes_through():
    assert score("0.75") == 0.75


def test_rejects_a_score_outside_the_unit_range():
    """Silently clamping would turn "I returned a percentage" into a real-looking
    grade, so this is an error the author has to see."""
    with pytest.raises(ScorerError, match="between 0 and 1"):
        run_scorer("75")


def test_clamp_is_the_documented_way_to_handle_that():
    assert score("clamp(75 / 100)") == 0.75
    assert score("clamp(1.4)") == 1.0


def test_rejects_a_non_numeric_return():
    with pytest.raises(ScorerError, match="must return a number"):
        run_scorer("'nearly'")


def test_rejects_nan():
    with pytest.raises(ScorerError, match="non-finite"):
        run_scorer("float('nan')")


def test_reason_names_the_returned_value():
    _, reason = run_scorer("True")
    assert "True" in reason


# ══ The checks people actually write ═════════════════════════════════════════

def test_contains_a_keyword():
    src = "contains(output, 'refund')"
    assert score(src, output="Your REFUND is on its way") == 1.0
    assert score(src, output="Your money is on its way") == 0.0


def test_contains_can_be_case_sensitive_when_it_matters():
    src = "contains(output, 'ERROR', case_sensitive=True)"
    assert score(src, output="ERROR: failed") == 1.0
    assert score(src, output="error: failed") == 0.0


def test_length_ceiling():
    """"is the response under 500 characters" — from the Braintrust demo."""
    src = "len(output) < 500"
    assert score(src, output="short") == 1.0
    assert score(src, output="x" * 600) == 0.0


def test_valid_json():
    src = "is_json(output)"
    assert score(src, output='{"ok": true}') == 1.0
    assert score(src, output="not json at all") == 0.0


def test_regex_format_check():
    src = r"matches(output, '^v[0-9]+\\.[0-9]+\\.[0-9]+$')"
    assert score(src, output="v1.2.3") == 1.0
    assert score(src, output="version one") == 0.0


def test_similarity_to_expected():
    assert score("similarity(output, expected)", output="hello", expected="hello") == 1.0
    assert score("similarity(output, expected)", output="hello", expected="world") < 0.5


def test_reads_json_fields_out_of_the_output():
    src = "json_parse(output, {}).get('status') == 'refunded'"
    assert score(src, output='{"status": "refunded"}') == 1.0
    assert score(src, output='{"status": "pending"}') == 0.0


def test_json_parse_survives_malformed_input():
    """A scorer that raises on every malformed row is a scorer nobody keeps."""
    assert score("json_parse(output, {}).get('x') == 1", output="{{{") == 0.0


def test_metadata_is_addressable():
    src = "metadata.get('tier') == 'gold'"
    assert score(src, metadata={"tier": "gold"}) == 1.0
    assert score(src, metadata={"tier": "free"}) == 0.0


def test_expected_output_is_addressable():
    assert score("output.strip() == expected.strip()", output=" yes ", expected="yes") == 1.0


def test_partial_credit_across_several_checks():
    """The pattern that makes code scorers worth having: a graded rubric with no
    model call and no variance."""
    src = (
        "clamp(("
        "  (1 if contains(output, 'sorry') else 0) +"
        "  (1 if contains(output, 'refund') else 0) +"
        "  (1 if len(output) < 300 else 0)"
        ") / 3)"
    )
    assert score(src, output="Sorry — your refund is on the way.") == 1.0
    assert round(score(src, output="Sorry about that."), 3) == round(2 / 3, 3)


def test_comprehension_over_required_phrases():
    src = (
        "clamp(len([p for p in ['hello', 'thanks', 'bye'] if contains(output, p)]) / 3)"
    )
    assert score(src, output="hello and thanks") == pytest.approx(2 / 3)


def test_word_count_band():
    src = "20 <= word_count(output) <= 60"
    assert score(src, output=" ".join(["word"] * 30)) == 1.0
    assert score(src, output="too short") == 0.0


def test_allowed_string_methods_work():
    assert score("output.lower().startswith('dear')", output="Dear customer") == 1.0
    assert score("output.count('a') == 3", output="banana") == 1.0
    assert score("output.count('z') == 3", output="banana") == 0.0


def test_f_strings_render_without_reaching_attributes():
    assert score("len(f'{output}!') == 4", output="abc") == 1.0


def test_f_string_conversions_are_honoured():
    """!r adds the quotes repr() puts round a string, so the rendered length grows."""
    assert score("len(f'{output!r}') == 3", output="a") == 1.0
    assert score("len(f'{output}') == 1", output="a") == 1.0


def test_f_string_format_specs_are_rejected_not_silently_dropped():
    """Ignoring a spec would hand the author a different string than they wrote."""
    with pytest.raises(ScorerError, match="Format specifiers"):
        run_scorer("len(f'{output:>10}') == 10", output="a")


def test_ternary_and_boolean_operators():
    assert score("1.0 if (len(output) > 2 and 'x' in output) else 0.0", output="axb") == 1.0
    assert score("0.5 if (False or len(output) == 0) else 0.25", output="") == 0.5


def test_chained_comparison():
    assert score("0 < len(output) < 10", output="hey") == 1.0


def test_compiling_once_and_running_many_times():
    """A batch run compiles the scorer once and reuses it across every example."""
    tree = compile_scorer("contains(output, 'ok')")
    assert run_scorer("", output="ok!", tree=tree)[0] == 1.0
    assert run_scorer("", output="no", tree=tree)[0] == 0.0


# ══ Author-facing errors ═════════════════════════════════════════════════════

def test_empty_scorer_is_rejected():
    with pytest.raises(ScorerError, match="empty"):
        compile_scorer("   ")


def test_syntax_error_reports_the_line():
    with pytest.raises(ScorerError, match="Syntax error"):
        compile_scorer("contains(output,")


def test_unknown_method_names_the_allowed_ones():
    with pytest.raises(ScorerError, match="allowed method"):
        compile_scorer("output.evaluate()")


def test_missing_dict_key_is_an_error_not_a_silent_zero():
    with pytest.raises(ScorerError, match="No such key"):
        run_scorer("metadata['absent'] == 1", metadata={})

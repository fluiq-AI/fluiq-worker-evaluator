"""Deterministic, code-based scorers.

An LLM judge answers subjective questions ("is this on-brand?"). A great many
useful checks are not subjective at all — "does the output contain the order
number", "is it under 500 characters", "is it valid JSON", "does it match the
changelog format" — and paying a model to answer those is slow, costly, and
*less* reliable than the two lines of code that decide them exactly.

This module runs those two lines safely.

Safety model
------------
Customer code runs inside the shared evaluator worker, which holds database
credentials and other orgs' data in the same process. Executing arbitrary Python
there would be indefensible, so this is **not** an interpreter with guard rails
bolted on — it is a small expression language that happens to use Python syntax:

* the source is parsed to an AST and walked; ``exec``/``eval`` of the source
  never happens
* every node type must be on an allowlist. ``import``, ``lambda``, assignment,
  ``def``/``class``, ``await``, ``yield``, and the walrus operator are all absent
  from it, so they are syntax the language does not have
* attribute access exists **only** as the callee of a method call, and only for
  method names on a fixed allowlist. That is what closes the classic
  ``().__class__.__bases__[0].__subclasses__()`` escape: there is no way to reach
  an attribute of any object, let alone a dunder
* ``str.format``/``format_map`` are excluded from the method allowlist
  specifically because format strings can themselves reach attributes
  (``"{0.__class__}".format(x)``)
* names resolve only against the scorer's own scope and a fixed builtin table.
  Nothing else is reachable, because nothing else is in scope
* work is bounded: a node budget, a recursion cap, string and collection size
  caps, and a ceiling on exponentiation

Known limitation: a pathological regular expression can still burn CPU on the
thread the scorer runs on (Python offers no way to interrupt one). Pattern and
subject length are capped to make that expensive rather than free, and the blast
radius is one worker thread. It is not a data-safety issue, but it is real, and
a runaway scorer will slow that org's own evaluations.

Contract
--------
The scorer is a single expression over ``output``, ``expected``, ``input``, and
``metadata``. It returns a number in 0..1, or a boolean (``True`` → 1.0).
"""
from __future__ import annotations

import ast
import difflib
import json
import re
from typing import Any, Dict, Mapping, Optional, Tuple

# ── Bounds ────────────────────────────────────────────────────────────────────
# Chosen to be far above any real scorer and far below anything that threatens
# the worker. Every one of them raises ScorerError rather than truncating, so a
# scorer that hits a limit reports it instead of silently scoring wrong.

MAX_SOURCE_CHARS   = 8_000
MAX_NODES          = 2_000     # parsed AST size
MAX_STEPS          = 50_000    # nodes actually evaluated (bounds comprehensions)
MAX_DEPTH          = 40
MAX_STRING_CHARS   = 200_000   # any single string produced or consumed
MAX_COLLECTION     = 10_000    # any single list/dict/set produced
MAX_EXPONENT       = 64        # 2**64 is plenty; 10**10**10 is not
MAX_PATTERN_CHARS  = 500


class ScorerError(Exception):
    """The scorer could not be compiled or run. The message is user-facing."""


# ── Allowlists ────────────────────────────────────────────────────────────────

_ALLOWED_NODES = frozenset({
    ast.Expression,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.JoinedStr, ast.FormattedValue,
    ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Subscript, ast.Slice,
    ast.Call, ast.Attribute, ast.keyword,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Store,  # comprehension targets bind here, nowhere else
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
})

# Methods callable on a value. Deliberately small, and deliberately excluding
# `format`/`format_map`, whose format-spec syntax can reach attributes.
_ALLOWED_METHODS = frozenset({
    # str
    "lower", "upper", "strip", "lstrip", "rstrip", "title", "casefold",
    "startswith", "endswith", "split", "rsplit", "splitlines", "partition",
    "replace", "count", "find", "rfind", "index", "join", "removeprefix",
    "removesuffix", "isdigit", "isalpha", "isalnum", "isspace", "isupper",
    "islower", "zfill", "ljust", "rjust",
    # dict
    "get", "keys", "values", "items",
    # list / set
    "union", "intersection", "difference", "issubset", "issuperset",
})

_DUNDER = re.compile(r"^__.*__$")


# ── Helper functions available to a scorer ────────────────────────────────────

def _check_str(value: Any, what: str = "value") -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) > MAX_STRING_CHARS:
        raise ScorerError(f"{what} exceeds {MAX_STRING_CHARS} characters")
    return text


def _fn_contains(haystack: Any, needle: Any, case_sensitive: bool = False) -> bool:
    """True when `needle` appears in `haystack`. Case-insensitive by default,
    because "does it mention the refund policy" almost never means "in this case"."""
    hay, need = _check_str(haystack), _check_str(needle)
    if not case_sensitive:
        hay, need = hay.lower(), need.lower()
    return need in hay


def _compile_pattern(pattern: Any) -> "re.Pattern[str]":
    pat = _check_str(pattern, "pattern")
    if len(pat) > MAX_PATTERN_CHARS:
        raise ScorerError(f"pattern exceeds {MAX_PATTERN_CHARS} characters")
    try:
        return re.compile(pat, re.DOTALL)
    except re.error as exc:
        raise ScorerError(f"invalid regular expression: {exc}") from exc


def _fn_matches(text: Any, pattern: Any) -> bool:
    """True when the pattern is found anywhere in the text."""
    return _compile_pattern(pattern).search(_check_str(text)) is not None


def _fn_find_all(text: Any, pattern: Any) -> list:
    out = _compile_pattern(pattern).findall(_check_str(text))
    if len(out) > MAX_COLLECTION:
        raise ScorerError("too many matches")
    return out


def _fn_is_json(text: Any) -> bool:
    try:
        json.loads(_check_str(text))
        return True
    except (ValueError, TypeError):
        return False


def _fn_json_parse(text: Any, default: Any = None) -> Any:
    """Parse JSON, returning `default` rather than raising — a scorer that has to
    guard every access is a scorer nobody writes correctly."""
    try:
        return json.loads(_check_str(text))
    except (ValueError, TypeError):
        return default


def _fn_similarity(a: Any, b: Any) -> float:
    """0..1 similarity ratio. Useful as a soft "close enough to expected" score."""
    return difflib.SequenceMatcher(None, _check_str(a), _check_str(b)).ratio()


def _fn_word_count(text: Any) -> int:
    return len(_check_str(text).split())


def _fn_clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(float(low), min(float(value), float(high)))
    except (TypeError, ValueError) as exc:
        raise ScorerError(f"clamp() needs a number, got {value!r}") from exc


def _safe_range(*args: Any) -> range:
    r = range(*args)
    if len(r) > MAX_COLLECTION:
        raise ScorerError(f"range() longer than {MAX_COLLECTION}")
    return r


_BUILTINS: Dict[str, Any] = {
    # types & maths
    "len": len, "abs": abs, "min": min, "max": max, "round": round, "sum": sum,
    "any": any, "all": all, "sorted": sorted,
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "range": _safe_range,
    # scoring helpers
    "contains": _fn_contains,
    "matches": _fn_matches,
    "find_all": _fn_find_all,
    "is_json": _fn_is_json,
    "json_parse": _fn_json_parse,
    "similarity": _fn_similarity,
    "word_count": _fn_word_count,
    "clamp": _fn_clamp,
}

#: Names a scorer can reference, for the editor's autocomplete and docs.
BUILTIN_NAMES = tuple(sorted(_BUILTINS))
SCOPE_NAMES = ("output", "expected", "input", "metadata")


# ── Validation ────────────────────────────────────────────────────────────────

def compile_scorer(source: str) -> ast.Expression:
    """Parse and validate a scorer, raising ScorerError with a usable message.

    Called by the API before saving so an invalid scorer is rejected at author
    time rather than failing silently on every example of a run.
    """
    if not isinstance(source, str) or not source.strip():
        raise ScorerError("Scorer code is empty.")
    if len(source) > MAX_SOURCE_CHARS:
        raise ScorerError(f"Scorer code exceeds {MAX_SOURCE_CHARS} characters.")

    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise ScorerError(f"Syntax error: {exc.msg} (line {exc.lineno})") from exc

    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if nodes > MAX_NODES:
            raise ScorerError("Scorer is too large.")
        _validate_node(node)
    return tree


def _validate_node(node: ast.AST) -> None:
    if type(node) not in _ALLOWED_NODES:
        raise ScorerError(f"{type(node).__name__} is not allowed in a scorer.")

    if isinstance(node, ast.Attribute):
        if _DUNDER.match(node.attr) or node.attr.startswith("_"):
            raise ScorerError("Private and dunder attributes are not allowed.")
        if node.attr not in _ALLOWED_METHODS:
            raise ScorerError(
                f"{node.attr!r} is not an allowed method. "
                f"Allowed: {', '.join(sorted(_ALLOWED_METHODS))}"
            )

    if isinstance(node, ast.Call):
        # An attribute callee is a method call, already checked above. Anything
        # else must be a plain name from the builtin table — no calling the
        # result of an expression, which is how you'd reach an arbitrary object.
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise ScorerError("Only named functions and allowed methods can be called.")
        if isinstance(node.func, ast.Name) and node.func.id not in _BUILTINS:
            raise ScorerError(
                f"{node.func.id!r} is not a known function. "
                f"Available: {', '.join(BUILTIN_NAMES)}"
            )
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise ScorerError("Argument unpacking is not allowed.")
        if any(kw.arg is None for kw in node.keywords):
            raise ScorerError("Keyword unpacking is not allowed.")

    if isinstance(node, ast.Name) and _DUNDER.match(node.id):
        raise ScorerError("Dunder names are not allowed.")

    # `ast.parse` accepts `async for` in a comprehension outside an async
    # function — the rule that forbids it is enforced at compile time, which we
    # never reach. Reject it here so it can't slip past validation and surface
    # only once a run is already underway.
    if isinstance(node, ast.comprehension) and node.is_async:
        raise ScorerError("Async comprehensions are not allowed.")


# ── Evaluation ────────────────────────────────────────────────────────────────

class _Interpreter:
    """Walks a validated AST. Every value it produces came from the scope, a
    literal, or a builtin — there is no path to anything else."""

    def __init__(self, scope: Mapping[str, Any]) -> None:
        self.scope = dict(scope)
        self.steps = 0

    def run(self, tree: ast.Expression) -> Any:
        return self.eval(tree.body, {}, 0)

    def eval(self, node: ast.AST, local: Dict[str, Any], depth: int) -> Any:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise ScorerError("Scorer did too much work; simplify it.")
        if depth > MAX_DEPTH:
            raise ScorerError("Scorer is nested too deeply.")

        method = getattr(self, f"_do_{type(node).__name__}", None)
        if method is None:
            raise ScorerError(f"{type(node).__name__} is not supported.")
        return method(node, local, depth + 1)

    # literals ---------------------------------------------------------------

    def _do_Constant(self, node: ast.Constant, _l: Dict[str, Any], _d: int) -> Any:
        return node.value

    def _do_List(self, node: ast.List, local: Dict[str, Any], depth: int) -> list:
        return self._sized([self.eval(e, local, depth) for e in node.elts])

    def _do_Tuple(self, node: ast.Tuple, local: Dict[str, Any], depth: int) -> tuple:
        return tuple(self._sized([self.eval(e, local, depth) for e in node.elts]))

    def _do_Set(self, node: ast.Set, local: Dict[str, Any], depth: int) -> set:
        return set(self._sized([self.eval(e, local, depth) for e in node.elts]))

    def _do_Dict(self, node: ast.Dict, local: Dict[str, Any], depth: int) -> dict:
        keys = [self.eval(k, local, depth) for k in node.keys if k is not None]
        vals = [self.eval(v, local, depth) for v in node.values]
        self._sized(vals)
        return dict(zip(keys, vals))

    def _do_JoinedStr(self, node: ast.JoinedStr, local: Dict[str, Any], depth: int) -> str:
        parts = [str(self.eval(v, local, depth)) for v in node.values]
        return _check_str("".join(parts))

    def _do_FormattedValue(
        self, node: ast.FormattedValue, local: Dict[str, Any], depth: int
    ) -> str:
        # Format specs are rejected rather than ignored. Applying one would mean
        # running format-spec syntax, which is another place attributes can be
        # reached from; ignoring one silently would pad nothing and quietly give
        # the author a different string than they wrote.
        if node.format_spec is not None:
            raise ScorerError(
                "Format specifiers in f-strings are not supported; "
                "use rjust()/ljust()/zfill() instead."
            )
        value = self.eval(node.value, local, depth)
        # Conversions operate on the value itself, never on its attributes, so
        # they are safe to honour.
        if node.conversion == 114:    # !r
            return repr(value)
        if node.conversion == 97:     # !a
            return ascii(value)
        return str(value)

    # names ------------------------------------------------------------------

    def _do_Name(self, node: ast.Name, local: Dict[str, Any], _d: int) -> Any:
        if node.id in local:
            return local[node.id]
        if node.id in self.scope:
            return self.scope[node.id]
        if node.id in _BUILTINS:
            return _BUILTINS[node.id]
        raise ScorerError(
            f"{node.id!r} is not defined. Available: "
            f"{', '.join(SCOPE_NAMES)} and {', '.join(BUILTIN_NAMES)}"
        )

    # operators --------------------------------------------------------------

    def _do_BinOp(self, node: ast.BinOp, local: Dict[str, Any], depth: int) -> Any:
        left = self.eval(node.left, local, depth)
        right = self.eval(node.right, local, depth)
        op = type(node.op)
        if op is ast.Pow:
            # Unbounded exponentiation is a one-line way to hang a worker.
            if isinstance(right, (int, float)) and abs(right) > MAX_EXPONENT:
                raise ScorerError(f"Exponent larger than {MAX_EXPONENT} is not allowed.")
        try:
            result = _BINOPS[op](left, right)
        except KeyError:
            raise ScorerError("Unsupported operator.") from None
        except ZeroDivisionError:
            raise ScorerError("Division by zero.") from None
        except TypeError as exc:
            raise ScorerError(f"Cannot apply operator: {exc}") from exc
        return self._sized_value(result)

    def _do_UnaryOp(self, node: ast.UnaryOp, local: Dict[str, Any], depth: int) -> Any:
        value = self.eval(node.operand, local, depth)
        op = type(node.op)
        if op is ast.Not:
            return not value
        if op is ast.USub:
            return -value
        if op is ast.UAdd:
            return +value
        raise ScorerError("Unsupported unary operator.")

    def _do_BoolOp(self, node: ast.BoolOp, local: Dict[str, Any], depth: int) -> Any:
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = self.eval(value, local, depth)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = self.eval(value, local, depth)
            if result:
                return result
        return result

    def _do_Compare(self, node: ast.Compare, local: Dict[str, Any], depth: int) -> bool:
        left = self.eval(node.left, local, depth)
        for op, comparator in zip(node.ops, node.comparators):
            right = self.eval(comparator, local, depth)
            try:
                ok = _COMPARES[type(op)](left, right)
            except KeyError:
                raise ScorerError("Unsupported comparison.") from None
            except TypeError as exc:
                raise ScorerError(f"Cannot compare: {exc}") from exc
            if not ok:
                return False
            left = right
        return True

    def _do_IfExp(self, node: ast.IfExp, local: Dict[str, Any], depth: int) -> Any:
        branch = node.body if self.eval(node.test, local, depth) else node.orelse
        return self.eval(branch, local, depth)

    # access -----------------------------------------------------------------

    def _do_Subscript(self, node: ast.Subscript, local: Dict[str, Any], depth: int) -> Any:
        value = self.eval(node.value, local, depth)
        key = self.eval(node.slice, local, depth)
        try:
            return value[key]
        except (KeyError, IndexError):
            raise ScorerError(f"No such key or index: {key!r}") from None
        except TypeError as exc:
            raise ScorerError(f"Cannot index that value: {exc}") from exc

    def _do_Slice(self, node: ast.Slice, local: Dict[str, Any], depth: int) -> slice:
        return slice(
            self.eval(node.lower, local, depth) if node.lower else None,
            self.eval(node.upper, local, depth) if node.upper else None,
            self.eval(node.step, local, depth) if node.step else None,
        )

    def _do_Attribute(self, node: ast.Attribute, local: Dict[str, Any], depth: int) -> Any:
        # Reached only as a Call's callee — validation rejects a bare attribute
        # load. The method name was allowlisted at compile time; this re-checks
        # because compile and run are separate entry points.
        if node.attr not in _ALLOWED_METHODS:
            raise ScorerError(f"{node.attr!r} is not an allowed method.")
        target = self.eval(node.value, local, depth)
        method = getattr(target, node.attr, None)
        if method is None or not callable(method):
            raise ScorerError(
                f"{type(target).__name__} has no method {node.attr!r}."
            )
        return method

    def _do_Call(self, node: ast.Call, local: Dict[str, Any], depth: int) -> Any:
        func = self.eval(node.func, local, depth)
        args = [self.eval(a, local, depth) for a in node.args]
        kwargs = {kw.arg: self.eval(kw.value, local, depth) for kw in node.keywords}
        try:
            return self._sized_value(func(*args, **kwargs))
        except ScorerError:
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced to the author, not swallowed
            raise ScorerError(f"{getattr(func, '__name__', 'call')} failed: {exc}") from exc

    # comprehensions ---------------------------------------------------------

    def _iter_comprehension(
        self, generators: list, local: Dict[str, Any], depth: int
    ):
        """Yield a scope per combination, binding only simple/tuple targets."""
        def rec(index: int, bound: Dict[str, Any]):
            if index == len(generators):
                yield bound
                return
            gen = generators[index]
            if gen.is_async:
                raise ScorerError("Async comprehensions are not allowed.")
            iterable = self.eval(gen.iter, {**local, **bound}, depth)
            count = 0
            for item in iterable:
                count += 1
                if count > MAX_COLLECTION:
                    raise ScorerError(f"Iterated more than {MAX_COLLECTION} items.")
                scope = dict(bound)
                self._bind(gen.target, item, scope)
                merged = {**local, **scope}
                if all(self.eval(c, merged, depth) for c in gen.ifs):
                    yield from rec(index + 1, scope)

        return rec(0, {})

    def _bind(self, target: ast.AST, value: Any, scope: Dict[str, Any]) -> None:
        if isinstance(target, ast.Name):
            scope[target.id] = value
            return
        if isinstance(target, ast.Tuple):
            try:
                items = list(value)
            except TypeError as exc:
                raise ScorerError(f"Cannot unpack {value!r}") from exc
            if len(items) != len(target.elts):
                raise ScorerError("Wrong number of values to unpack.")
            for sub, item in zip(target.elts, items):
                self._bind(sub, item, scope)
            return
        raise ScorerError("Unsupported loop target.")

    def _do_ListComp(self, node: ast.ListComp, local: Dict[str, Any], depth: int) -> list:
        out = [
            self.eval(node.elt, {**local, **bound}, depth)
            for bound in self._iter_comprehension(node.generators, local, depth)
        ]
        return self._sized(out)

    def _do_SetComp(self, node: ast.SetComp, local: Dict[str, Any], depth: int) -> set:
        return set(self._do_ListComp(  # type: ignore[arg-type]
            ast.ListComp(elt=node.elt, generators=node.generators), local, depth,
        ))

    def _do_GeneratorExp(self, node: ast.GeneratorExp, local: Dict[str, Any], depth: int) -> list:
        # Materialised rather than lazy, so the size cap actually applies.
        return self._do_ListComp(  # type: ignore[arg-type]
            ast.ListComp(elt=node.elt, generators=node.generators), local, depth,
        )

    def _do_DictComp(self, node: ast.DictComp, local: Dict[str, Any], depth: int) -> dict:
        out = {}
        for bound in self._iter_comprehension(node.generators, local, depth):
            merged = {**local, **bound}
            out[self.eval(node.key, merged, depth)] = self.eval(node.value, merged, depth)
            if len(out) > MAX_COLLECTION:
                raise ScorerError(f"Built more than {MAX_COLLECTION} items.")
        return out

    # bounds -----------------------------------------------------------------

    def _sized(self, items: list) -> list:
        if len(items) > MAX_COLLECTION:
            raise ScorerError(f"Built more than {MAX_COLLECTION} items.")
        return items

    def _sized_value(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
            raise ScorerError(f"Produced a string over {MAX_STRING_CHARS} characters.")
        if isinstance(value, (list, tuple, set, dict)) and len(value) > MAX_COLLECTION:
            raise ScorerError(f"Produced more than {MAX_COLLECTION} items.")
        return value


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_COMPARES = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


# ── Public entry point ────────────────────────────────────────────────────────

def _coerce_score(value: Any) -> float:
    """A scorer returns a 0..1 number or a boolean; anything else is an error the
    author needs to see, not a zero that looks like a genuine failing grade."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        score = float(value)
        if score != score or score in (float("inf"), float("-inf")):
            raise ScorerError("Scorer returned a non-finite number.")
        if not 0.0 <= score <= 1.0:
            raise ScorerError(
                f"Scorer returned {score:g}; it must be between 0 and 1 "
                f"(use clamp() if the value can go outside that range)."
            )
        return score
    raise ScorerError(
        f"Scorer returned {type(value).__name__}; it must return a number "
        f"between 0 and 1, or True/False."
    )


def run_scorer(
    source: str,
    *,
    output: str = "",
    expected: str = "",
    input: str = "",          # noqa: A002 — the name the scorer author sees
    metadata: Optional[Mapping[str, Any]] = None,
    tree: Optional[ast.Expression] = None,
) -> Tuple[float, str]:
    """Run a code scorer. Returns ``(score, reason)``.

    Raises :class:`ScorerError` for anything the author could fix — invalid
    syntax, a disallowed construct, a bad return value. The caller decides
    whether that skips the metric or fails the run.
    """
    validated = tree if tree is not None else compile_scorer(source)
    scope = {
        "output":   _check_str(output, "output"),
        "expected": _check_str(expected, "expected"),
        "input":    _check_str(input, "input"),
        "metadata": dict(metadata or {}),
    }
    interpreter = _Interpreter(scope)
    value = interpreter.run(validated)
    score = _coerce_score(value)
    reason = (
        f"Code scorer returned {value!r}"
        if isinstance(value, bool)
        else f"Code scorer returned {score:.3f}"
    )
    return score, reason


__all__ = [
    "BUILTIN_NAMES",
    "SCOPE_NAMES",
    "ScorerError",
    "compile_scorer",
    "run_scorer",
]

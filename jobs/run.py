import asyncio
import functools
import json
import logging
import time
from typing import Any, Dict, List, Optional

import config

from jobs.helper import crypto, judge_prompts
from jobs.helper.base import EvalResult
from jobs.helper.judge import (
    PROVIDERS as JUDGE_PROVIDERS,
    InMemoryCache,
    JudgeUsage,
    LLMJudge,
    PromptCache,
)
from jobs.helper.hallucination import HallucinationEvaluator
from jobs.helper.ragas import (
    AnswerRelevancy,
    Completeness,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    Coherence,
    Toxicity,
    Ragas,
)
from db.clickhouse import clickhouse_eval_client
from db.kafka import kafka_producer
from db.postgres import postgres_client

logger = logging.getLogger(__name__)

_judge_cache_backend = (
    InMemoryCache(max_size=config.JUDGE_CACHE_MAX) if config.JUDGE_CACHE_ENABLED else None
)

_RETRIEVAL_APIS = {
    "query", "search", "near_text", "near_vector",
    "hybrid", "bm25", "query_points", "fetch_objects",
}


class BYOKUnavailable(RuntimeError):
    """An org has a provider credential on file, but it cannot be used.

    Raised instead of quietly building a judge on Fluiq's managed key. Falling
    back would bill us for usage the customer was sold as bring-your-own-key,
    and would hide a broken credential from the person who has to rotate it.
    """


class OrgCredentials:
    """BYOK state for one org, resolved once and reused for the whole message.

    Resolving per judge call would mean a Postgres round trip (and a KMS
    unwrap) per juror. A jury of three across three metrics would pay that nine
    times for one eval.
    """

    __slots__ = ("_org_id", "_rows", "_unsealed", "_known", "_loop", "_reported")

    def __init__(
        self,
        org_id: Any,
        rows: Optional[dict],
        loop: Optional[Any] = None,
    ) -> None:
        self._org_id = org_id
        self._rows = rows or {}
        # None from the fetch means "Postgres could not tell us", which is not
        # the same as "this org has no keys" — see fetch_org_credentials.
        self._known = rows is not None
        self._unsealed: dict[str, str] = {}
        # Judge calls run in a worker thread via asyncio.to_thread, so writing
        # back a rejection needs the loop that owns the Postgres pool.
        self._loop = loop
        self._reported: set = set()

    def note_auth_failure(self, provider: str, reason: str) -> None:
        """Record that the provider rejected this org's key for ``provider``.

        Called from the judge thread. Schedules the Postgres write on the
        worker's event loop and returns immediately: the evaluation is already
        failing, and blocking it on a database round trip helps nobody.
        Reported once per message so a jury of three does not write three times.
        """
        row = self._rows.get(provider)
        if row is None or provider in self._reported:
            return
        self._reported.add(provider)
        if self._loop is None:
            return
        try:
            import asyncio as _asyncio
            _asyncio.run_coroutine_threadsafe(
                postgres_client.mark_credential_invalid(
                    self._org_id, row["credential_id"], reason,
                ),
                self._loop,
            )
        except Exception:  # noqa: BLE001 - never let bookkeeping break an eval
            logger.exception(
                "[EVALUATOR] could not schedule credential invalidation org=%s",
                self._org_id,
            )

    @property
    def resolved(self) -> bool:
        return self._known

    def api_key_for(self, provider: str) -> Optional[str]:
        """Return the org's key for ``provider``, or None to use managed keys.

        Raises :class:`BYOKUnavailable` when the org *has* a credential for this
        provider that we cannot use — an unknown state must never resolve to
        Fluiq's account.
        """
        row = self._rows.get(provider)
        if row is None:
            return None  # no BYOK for this provider → managed key, as before

        if row["status"] != "active":
            raise BYOKUnavailable(
                f"{provider} credential is marked {row['status']}"
            )

        cached = self._unsealed.get(provider)
        if cached is not None:
            return cached

        try:
            plaintext = crypto.unseal(
                crypto.SealedSecret(
                    ciphertext=bytes(row["ciphertext"]),
                    nonce=bytes(row["nonce"]),
                    wrapped_dek=bytes(row["wrapped_dek"]),
                    key_version=row["key_version"],
                ),
                org_id=str(self._org_id),
            )
        except Exception as exc:
            # Never let the exception carry ciphertext or key material outward.
            logger.exception(
                "[EVALUATOR] credential unseal failed org=%s provider=%s",
                self._org_id, provider,
            )
            raise BYOKUnavailable(f"{provider} credential could not be decrypted") from None

        self._unsealed[provider] = plaintext
        return plaintext


async def resolve_org_credentials(organization_id: Any) -> OrgCredentials:
    """Load an org's BYOK credentials once per eval message."""
    loop = asyncio.get_running_loop()
    if organization_id is None:
        return OrgCredentials(None, {}, loop)
    rows = await postgres_client.fetch_org_credentials(organization_id)
    if rows is None:
        logger.error(
            "[EVALUATOR] could not read credentials org=%s; falling back to managed "
            "keys for this message", organization_id,
        )
    return OrgCredentials(organization_id, rows, loop)


def parse_judge_spec(raw: Any) -> tuple[Optional[str], Optional[str]]:
    """Parse a ``"provider:model"`` string into ``(provider, model)``.

    Returns ``(None, None)`` for anything unusable so callers fall back to the
    server default. A caller-supplied judge is untrusted input: an unknown
    provider would raise deep inside LLMJudge, which is a poor way to learn
    that a dropdown sent something unexpected.
    """
    if not raw or not isinstance(raw, str) or ":" not in raw:
        return (None, None)
    provider, _, model = raw.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in JUDGE_PROVIDERS or not model:
        logger.warning("[EVALUATOR] ignoring unusable judge spec %r", raw)
        return (None, None)
    return (provider, model)


def parse_jury_specs(raw: Any) -> list[tuple[str, str]]:
    """Parse a list of ``"provider:model"`` strings into member specs.

    Unusable entries are dropped rather than failing the run: a jury with two
    good members is still a jury, and the alternative is a dropdown typo
    silently costing the customer a whole batch.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [part for part in raw.split(",")]
    specs: list[tuple[str, str]] = []
    for entry in raw:
        provider, model = parse_judge_spec(entry)
        if provider and model:
            specs.append((provider, model))
    return specs


_AUTH_ERROR_NAMES = (
    "AuthenticationError",
    "PermissionDeniedError",
    "Unauthorized",
    "Forbidden",
)


def is_auth_error(exc: BaseException) -> bool:
    """Whether a provider exception means "this key is bad".

    Provider SDKs do not share a base class, so this checks the two signals
    they do agree on: an HTTP status attribute, and a class name. Deliberately
    conservative — a false positive would disable a working customer key, so
    anything ambiguous is treated as a transient failure instead.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403):
        return True
    name = type(exc).__name__
    if any(marker in name for marker in _AUTH_ERROR_NAMES):
        return True
    return False


def _auth_error_reason(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    prefix = f"HTTP {status}: " if status else ""
    # Truncated and generic: provider error bodies can echo request metadata,
    # and this string is shown in the dashboard.
    return f"{prefix}provider rejected the key ({type(exc).__name__})"


def _build_judge(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    org_id: Any = None,
    creds: Optional[OrgCredentials] = None,
    usage: Optional[JudgeUsage] = None,
) -> LLMJudge:
    """Build a judge, optionally on the org's own provider key and cache bucket.

    ``org_id`` scopes the cache to one tenant. It is required to cache at all:
    ``_judge_cache_backend`` is process-wide and shared across every org this
    worker serves, so an unattributed entry could be served to a different org.
    A message without an organization is malformed, so we'd rather pay for the
    judge call than guess at the tenant.

    Raises :class:`BYOKUnavailable` when the org has an unusable credential for
    the requested provider.
    """
    resolved_provider = provider or config.JUDGE_PROVIDER
    api_key = creds.api_key_for(resolved_provider) if creds is not None else None

    judge = LLMJudge(
        provider=resolved_provider,
        model=model or config.JUDGE_MODEL,
        api_key=api_key,
        usage=usage,
    )
    def _underlying(prompt: str, choice_labels=None, **_params) -> str:
        # `choice_labels` has to survive this hop: it is what pins a forced-choice
        # judge's answer to an enum at the provider. Dropping it here would leave
        # the feature working only when no cache and no BYOK key are configured —
        # which is to say, never in production.
        #
        # Passed only when set, so a stub that replaces `_call_*` with a
        # prompt-only function keeps working.
        extra = {"choice_labels": choice_labels} if choice_labels else {}
        if judge.provider == "openai":
            return judge._call_openai(prompt, **extra)
        if judge.provider == "anthropic":
            return judge._call_anthropic(prompt, **extra)
        if judge.provider == "gemini":
            return judge._call_gemini(prompt, **extra)
        if judge.provider == "moonshot":
            return judge._call_moonshot(prompt, **extra)
        raise RuntimeError(f"Unsupported judge provider: {judge.provider}")

    call = _underlying

    if _judge_cache_backend is not None and org_id is not None:
        # The cache bucket is scoped to the tenant *and* to whichever credential
        # paid for the verdict, so a cached answer can never be served across an
        # org boundary or attributed to the wrong provider account.
        namespace = str(org_id)
        if api_key:
            namespace = f"{namespace}:{crypto.fingerprint(api_key)}"
        call = PromptCache(
            _underlying,
            model=f"{judge.provider}:{judge.model}",
            backend=_judge_cache_backend,
            ttl=config.JUDGE_CACHE_TTL,
            namespace=namespace,
        )

    if api_key and creds is not None:
        # A customer key the provider rejects is reported back so the dashboard
        # stops showing it Active. Wrapped here rather than in the handlers
        # because this is the only place that knows the call ran on BYOK.
        inner = call

        def _reporting(prompt: str, **params) -> str:
            try:
                return inner(prompt, **params)  # params carries choice_labels
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if is_auth_error(exc):
                    creds.note_auth_failure(
                        resolved_provider, _auth_error_reason(exc),
                    )
                raise

        call = _reporting

    if call is _underlying:
        return judge  # nothing wrapped — leave __call__ on the direct path

    # **kw forwards choice_labels for forced-choice judging; the cache keys on it
    # too, so a "Y/N" verdict can't be served for a differently-worded option set.
    judge._judge_fn = lambda prompt, **kw: call(
        prompt, temperature=judge.temperature, **kw,
    )
    return judge


def _build_evaluator(name: str, judge: LLMJudge):
    n = (name or "").lower().strip()
    if n in ("hallucination", "hallucinationevaluator"):
        return HallucinationEvaluator(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("faithfulness", "ragas.faithfulness"):
        return Faithfulness(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("answer_relevancy", "relevance", "ragas.answer_relevancy"):
        return AnswerRelevancy(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("context_precision", "ragas.context_precision"):
        return ContextPrecision(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("context_recall", "ragas.context_recall"):
        return ContextRecall(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("toxicity", "ragas.toxicity"):
        return Toxicity(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("coherence","ragas.coherence"):
        return Coherence(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("completeness", "ragas.completeness"):
        return Completeness(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("vision_faithfulness", "vision", "vision.faithfulness"):
        from jobs.helper.vision import VisionFaithfulness
        return VisionFaithfulness(judge=judge, threshold=config.JUDGE_THRESHOLD)
    if n in ("media_faithfulness", "media", "av_faithfulness"):
        from jobs.helper.vision import MediaFaithfulness
        return MediaFaithfulness(judge=judge, threshold=config.JUDGE_THRESHOLD)
    raise ValueError(f"Unknown evaluator: {name!r}")


def _is_retrieval_event(event: Dict[str, Any]) -> bool:
    return (
        event.get("type") == "vectorstore"
        and event.get("api") in _RETRIEVAL_APIS
    )


def _extract_question(event: Dict[str, Any]) -> str:
    query = event.get("query") or {}
    texts = query.get("texts") if isinstance(query, dict) else None
    if isinstance(texts, list) and texts:
        return "\n".join(str(t) for t in texts if t)
    return ""


def _extract_llm_question(event: Dict[str, Any]) -> str:
    messages = event.get("messages") or event.get("contents") or event.get("input") or []
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str) and content.strip():
                    return content
        parts = [
            str(m.get("content") or "")
            for m in messages
            if isinstance(m, dict) and m.get("content")
        ]
        return "\n".join(parts)
    return str(messages) if messages else ""


def _extract_contexts(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = event.get("result") or {}
    matches = result.get("matches") if isinstance(result, dict) else None
    items = matches.get("items") if isinstance(matches, dict) else None
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        text = m.get("text") or ""
        if not text:
            continue
        out.append({"id": m.get("id"), "score": m.get("score"), "text": str(text)})
    return out


def _stringify_grounding(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    try:
        return json.dumps(v, ensure_ascii=False)[:2000]
    except Exception:
        return str(v)[:2000]


def _fmt_grounding(name: Any, tool_input: Any, output: Any) -> Optional[str]:
    """One grounding block: 'Tool "name" (input: …) returned:\\n<output>'.

    The input is included when known so the judge can tie a result to its request
    (e.g. get_weather(city=Paris) → 18°C), which a bare output may not make
    explicit. Returns None when there is no output to ground against.
    """
    out = _stringify_grounding(output)
    if not out:
        return None
    head = f'Tool "{name or "tool"}"'
    inp = _stringify_grounding(tool_input)
    if inp:
        head += f" (input: {inp})"
    return f"{head} returned:\n{out}"


def _extract_tool_grounding(event: Dict[str, Any]) -> List[str]:
    """Tool + MCP results the agent retrieved, as grounding text for the judge.

    Mirrors the frontend `buildToolGrounding`: without this, a trace that calls a
    tool (or MCP server) and answers from its result looks like a hallucination
    because the judge never sees the tool output. Reads tool call inputs + results
    from the message history (OpenAI role:"tool" messages, Anthropic tool_result
    blocks, Gemini function_response) and MCP results from `mcp_calls`.
    """
    out: List[str] = []

    # Map tool-call id → (name, input args) so each result can carry its request.
    call_info: Dict[str, tuple] = {}

    def _record_calls(tcs: Any) -> None:
        if not isinstance(tcs, list):
            return
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
            src = fn or tc
            cid = tc.get("id")
            if isinstance(cid, str):
                args = src.get("arguments", tc.get("input") if tc.get("input") is not None else tc.get("args"))
                call_info[cid] = (src.get("name"), args)

    _record_calls(event.get("tool_calls"))
    msgs = event.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                _record_calls(m.get("tool_calls"))

    # ── Tool results in message history ──
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if str(m.get("role") or "").lower() == "tool":
                cid = m.get("tool_call_id")
                cname, cinput = call_info.get(cid, (None, None)) if isinstance(cid, str) else (None, None)
                block = _fmt_grounding(m.get("name") or cname, cinput, m.get("content"))
                if block:
                    out.append(block)
                continue
            content = m.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        cid = blk.get("tool_use_id")
                        cname, cinput = call_info.get(cid, (None, None)) if isinstance(cid, str) else (None, None)
                        block = _fmt_grounding(cname, cinput, blk.get("content"))
                        if block:
                            out.append(block)
    # Gemini function_response parts
    contents = event.get("contents")
    if isinstance(contents, list):
        for c in contents:
            parts = c.get("parts") if isinstance(c, dict) else None
            if not isinstance(parts, list):
                continue
            for pt in parts:
                fr = pt.get("function_response") if isinstance(pt, dict) else None
                if isinstance(fr, dict):
                    block = _fmt_grounding(fr.get("name"), None, fr.get("response"))
                    if block:
                        out.append(block)

    # ── MCP results ──
    mcp = event.get("mcp_calls")
    if isinstance(mcp, list):
        results_by_id: Dict[str, Any] = {}
        for field in (mcp, event.get("mcp_results")):
            if not isinstance(field, list):
                continue
            for item in field:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "mcp_tool_result"
                    and isinstance(item.get("tool_use_id"), str)
                ):
                    results_by_id[item["tool_use_id"]] = item.get("content")
        for item in mcp:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            server = item.get("server_label") or item.get("server_name")
            label = f'{item.get("name") or "mcp"}{f" (MCP server: {server})" if server else " (MCP)"}'
            if t == "mcp_call":
                block = _fmt_grounding(label, item.get("arguments"), item.get("output") or item.get("error"))
                if block:
                    out.append(block)
            elif t == "mcp_tool_use":
                block = _fmt_grounding(label, item.get("input"), results_by_id.get(item.get("id")))
                if block:
                    out.append(block)

    return out


async def run_evaluation(message: Dict[str, Any]) -> None:
    """Explicit eval (operation='evaluate')."""
    organization_id  = message.get("organization_id")
    api_key_prefix   = message.get("api_key_prefix")
    trace_id         = message.get("trace_id")
    evaluator_name   = (message.get("evaluator") or "").strip()
    inputs           = message.get("inputs") or {}

    if not evaluator_name:
        logger.warning("[EVALUATOR] Missing 'evaluator' in message; skipping")
        return

    creds = await resolve_org_credentials(organization_id)
    usage = JudgeUsage()
    # Per-message judge selection. Falls back to the server default when absent
    # or unparseable, so an older SDK keeps working unchanged.
    judge_provider, judge_model = parse_judge_spec(message.get("judge"))
    try:
        judge = _build_judge(
            judge_provider, judge_model,
            org_id=organization_id, creds=creds, usage=usage,
        )
    except BYOKUnavailable as exc:
        logger.error(
            "[EVALUATOR] not evaluating org=%s trace_id=%s: %s",
            organization_id, trace_id, exc,
        )
        return

    try:
        if evaluator_name.lower() == "ragas":
            results = await asyncio.to_thread(
                Ragas(judge=judge, threshold=config.JUDGE_THRESHOLD).evaluate,
                question=inputs.get("question", ""),
                answer=inputs.get("answer", ""),
                contexts=inputs.get("contexts"),
                reference=inputs.get("reference"),
            )
            for metric_name, result in results.items():
                await _persist_eval_result(
                    organization_id, api_key_prefix, trace_id,
                    evaluator="ragas",
                    metric=metric_name,
                    result=result,
                    judge=judge,
                )
            logger.info(
                "[EVALUATOR] ragas org=%s prefix=%s metrics=%s",
                organization_id, api_key_prefix, list(results),
            )
            return

        ev = _build_evaluator(evaluator_name, judge)
        result = await asyncio.to_thread(judge_prompts.captured_call, ev.evaluate, **inputs)
        await _persist_eval_result(
            organization_id, api_key_prefix, trace_id,
            evaluator=evaluator_name,
            metric=ev.name,
            result=result,
            judge=judge,
        )
        logger.info(
            "[EVALUATOR] %s org=%s prefix=%s score=%s passed=%s",
            ev.name, organization_id, api_key_prefix, result.score, result.passed,
        )
    except Exception:
        logger.exception(
            "[EVALUATOR] Evaluation failed evaluator=%s org=%s prefix=%s",
            evaluator_name, organization_id, api_key_prefix,
        )
        raise


async def auto_evaluate_retrieval(message: Dict[str, Any]) -> None:
    """Score a vectorstore retrieval trace with ContextPrecision."""
    event = message.get("event") or {}
    if not isinstance(event, dict) or not _is_retrieval_event(event):
        return

    organization_id = message.get("organization_id")
    api_key_prefix  = message.get("api_key_prefix")
    trace_id        = message.get("trace_id") or event.get("trace_id")
    root_trace_id   = event.get("root_trace_id") or trace_id

    question = _extract_question(event)
    contexts = _extract_contexts(event)
    if not question or not contexts:
        logger.info(
            "[EVALUATOR] Skipping auto-eval: trace_id=%s missing question or contexts",
            trace_id,
        )
        return

    creds = await resolve_org_credentials(organization_id)
    usage = JudgeUsage()
    # Per-message judge selection. Falls back to the server default when absent
    # or unparseable, so an older SDK keeps working unchanged.
    judge_provider, judge_model = parse_judge_spec(message.get("judge"))
    try:
        judge = _build_judge(
            judge_provider, judge_model,
            org_id=organization_id, creds=creds, usage=usage,
        )
    except BYOKUnavailable as exc:
        logger.error(
            "[EVALUATOR] not evaluating org=%s trace_id=%s: %s",
            organization_id, trace_id, exc,
        )
        return
    started = time.time()
    try:
        ev = ContextPrecision(judge=judge, threshold=config.JUDGE_THRESHOLD)
        result = await asyncio.to_thread(
            judge_prompts.captured_call,
            ev.evaluate,
            question=question,
            contexts=[c["text"] for c in contexts],
        )
    except Exception:
        logger.exception(
            "[EVALUATOR] auto-eval failed trace_id=%s integration=%s",
            trace_id, event.get("integration"),
        )
        raise

    details = result.model_dump(mode="json")
    flags = (details.get("details") or {}).get("flags") or []
    per_chunk = [
        {
            "id":     ctx.get("id"),
            "score":  ctx.get("score"),
            "useful": bool(flags[i]) if i < len(flags) else None,
        }
        for i, ctx in enumerate(contexts)
    ]
    details["per_chunk"] = per_chunk
    details["question"]  = question
    details["latency"]   = time.time() - started

    await _persist_eval_result(
        organization_id, api_key_prefix, trace_id,
        evaluator="vectorstore.context_relevance",
        metric=ev.name,
        result=result,
        judge=judge,
        root_trace_id=root_trace_id,
        details_override=details,
    )
    logger.info(
        "[EVALUATOR] context_relevance trace_id=%s score=%.3f chunks=%d",
        trace_id, result.score, len(contexts),
    )


async def auto_llm_eval(message: Dict[str, Any]) -> None:
    """Score an SDK LLM trace with the metrics in eval_config (warn mode)."""
    eval_config     = message.get("eval_config") or {}
    event           = message.get("event") or {}
    organization_id = message.get("organization_id")
    api_key_prefix  = message.get("api_key_prefix")
    trace_id        = message.get("trace_id") or event.get("trace_id")

    metrics       = eval_config.get("metrics") or ["hallucination", "relevance"]
    thresholds    = eval_config.get("thresholds") or {}
    custom_judges = eval_config.get("custom_judges") or {}

    question = _extract_llm_question(event)
    answer   = event.get("response") or event.get("output") or ""

    if not question or not answer:
        logger.info(
            "[EVALUATOR] Skipping LLM eval: trace_id=%s missing question or answer",
            trace_id,
        )
        return

    # Dataset metric runs grade the answer against the example's expected
    # output: reference-aware metrics (hallucination) use it directly, and
    # faithfulness treats it as the context to be grounded in. Metrics that
    # don't take a reference ignore the extra kwargs.
    #
    # Grounding for factual metrics (hallucination / faithfulness): the tool &
    # MCP outputs this run actually retrieved, so tool-backed claims aren't
    # flagged as unsupported. Mirrors the dashboard's Traces eval.
    #
    # `contexts` is always passed (may be empty) because the RAGAS evaluators
    # (faithfulness / context_precision / context_recall) take it as a required
    # positional arg — omitting it raises a TypeError; with none they return a
    # graceful "no contexts provided" 0.0 instead.
    reference = str(message.get("reference") or "").strip()
    contexts: List[str] = _extract_tool_grounding(event)
    # Trace-backed dataset runs pass grounding pre-extracted from all pinned spans
    # (the worker only receives the root event), so merge that in too.
    extra_grounding = message.get("tool_grounding")
    if isinstance(extra_grounding, list):
        for g in extra_grounding:
            if isinstance(g, str) and g.strip() and g not in contexts:
                contexts.append(g)
    if reference:
        # Dataset runs grade against the example's expected output too.
        contexts.append(reference)
    ref_kwargs: Dict[str, Any] = {"reference": reference, "contexts": contexts}

    creds = await resolve_org_credentials(organization_id)
    usage = JudgeUsage()
    # Per-message judge selection. Falls back to the server default when absent
    # or unparseable, so an older SDK keeps working unchanged.
    judge_provider, judge_model = parse_judge_spec(message.get("judge"))
    try:
        judge = _build_judge(
            judge_provider, judge_model,
            org_id=organization_id, creds=creds, usage=usage,
        )
    except BYOKUnavailable as exc:
        logger.error(
            "[EVALUATOR] not evaluating org=%s trace_id=%s: %s",
            organization_id, trace_id, exc,
        )
        return

    for metric_name in metrics:
        try:
            ev = _build_evaluator(metric_name, judge)
            # Pass the raw event so vision metrics can pull media refs from the
            # trace; text-only evaluators ignore the extra kwargs.
            result = await asyncio.to_thread(
                judge_prompts.captured_call,
                ev.evaluate, question=question, answer=answer, event=event,
                **ref_kwargs,
            )
            await _persist_eval_result(
                organization_id, api_key_prefix, trace_id,
                evaluator="fluiq.eval",
                metric=ev.name,
                result=result,
                judge=judge,
            )
            logger.info(
                "[EVALUATOR] sdk_llm %s org=%s trace_id=%s score=%.3f threshold=%.3f",
                ev.name, organization_id, trace_id,
                result.score, thresholds.get(metric_name, 0.0),
            )
        except ValueError:
            logger.warning(
                "[EVALUATOR] Unknown metric %r for trace_id=%s", metric_name, trace_id,
            )
        except Exception:
            logger.exception(
                "[EVALUATOR] LLM eval failed metric=%s trace_id=%s", metric_name, trace_id,
            )

    # Client-defined custom judges over the answer under test.
    await _run_custom_judges(
        custom_judges,
        organization_id=organization_id,
        api_key_prefix=api_key_prefix,
        trace_id=trace_id,
        judge=judge,
        question=question,
        answer=answer,
        context=reference,
        example_metadata=eval_config.get("example_metadata"),
    )


async def _run_custom_judges(
    custom_judges: Dict[str, Any],
    *,
    organization_id: Any,
    api_key_prefix: Any,
    trace_id: str,
    judge: Any,
    question: str,
    answer: str,
    context: str,
    root_trace_id: Optional[str] = None,
    example_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Score an answer with each client-defined scorer referenced by slug.

    A scorer is either an LLM-as-judge prompt (``kind='judge'``) or a
    deterministic expression (``kind='code'``); the client references both the
    same way, and this routes on what was actually saved. Persists one eval row
    per scorer, and is shared by the single-turn and agentic paths so a scorer
    behaves identically on both.

    Best-effort per scorer: a missing template, an unreachable Postgres, or an
    error inside one scorer skips it rather than failing the whole evaluation.
    """
    if not custom_judges:
        return
    from db.postgres import postgres_client
    from jobs.helper.code_scorer_evaluator import CodeScorerEvaluator
    from jobs.helper.custom_judge import CustomJudgeEvaluator

    from jobs.helper.choice_scores import parse_choices

    for slug, threshold in custom_judges.items():
        found = await postgres_client.fetch_custom_scorer(organization_id, slug)
        if not found or not found[1]:
            logger.info(
                "[EVALUATOR] custom scorer %r unavailable for org=%s; skipping",
                slug, organization_id,
            )
            continue
        kind, template, scorer_config = found
        try:
            if kind == "code":
                ev: Any = CodeScorerEvaluator(
                    source=template, name=slug, threshold=float(threshold or 0.0),
                )
                result = await asyncio.to_thread(
                    ev.evaluate,
                    question=question, answer=answer,
                    expected=context, metadata=example_metadata or {},
                )
            else:
                # A malformed choice set must not silently become a free-score
                # judge: that would change what the scorer means without saying so.
                choices = parse_choices((scorer_config or {}).get("choices"))
                ev = CustomJudgeEvaluator(
                    judge=judge, template=template, name=slug,
                    threshold=float(threshold or 0.0),
                    choices=choices,
                )
                result = await asyncio.to_thread(
                    ev.evaluate, question=question, answer=answer, context=context,
                )
            await _persist_eval_result(
                organization_id, api_key_prefix, trace_id,
                evaluator="fluiq.eval",
                metric=slug,
                result=result,
                # A code scorer makes no model call, so attributing one to a
                # judge model would misreport both cost and provenance.
                judge=None if kind == "code" else judge,
                root_trace_id=root_trace_id,
            )
            logger.info(
                "[EVALUATOR] custom_scorer(%s) %s org=%s trace_id=%s score=%.3f threshold=%.3f",
                kind, slug, organization_id, trace_id, result.score, float(threshold or 0.0),
            )
        except Exception:
            logger.exception(
                "[EVALUATOR] custom scorer failed kind=%s slug=%s trace_id=%s",
                kind, slug, trace_id,
            )


async def agent_evaluate(message: Dict[str, Any]) -> None:
    """Agentic evaluation (operation='agent_eval').

    Normalizes any trace source (Fluiq envelope, OpenInference/OTel spans, or
    raw JSON) into an ``AgentRun`` and runs the Phase 0→2 slice: deterministic
    checks (Layer 1) + tool-selection-quality judge (Layer 2). One eval row is
    persisted per LLM-judged metric; deterministic findings ride along in the
    details so the UI can render per-call issues without a judge call.
    """
    from jobs.agentic.adapters import normalize
    from jobs.agentic.orchestrator import evaluate_run
    from jobs.agentic.panel import build_panel

    organization_id = message.get("organization_id")
    api_key_prefix  = message.get("api_key_prefix")

    run = normalize(message)
    trace_id      = message.get("trace_id") or run.run_id
    root_trace_id = message.get("root_trace_id") or run.run_id or trace_id

    if not run.tool_calls and len(run.steps) < 2:
        # A single LLM turn with no tool calls has no agentic trajectory to
        # assess, and is already covered by the per-call hallucination/relevance
        # eval — so there's nothing agentic to add here. Multi-step reasoning
        # runs (>=2 steps) ARE evaluated below even without tools: L1
        # deterministic and L3 trajectory judge the steps/final answer and are
        # tool-agnostic; only the L2 tool-selection layer needs tool calls.
        logger.info(
            "[EVALUATOR] agent_eval trace_id=%s has no tool calls and <2 steps; "
            "nothing agentic to evaluate; skipping", trace_id,
        )
        return

    # Per-message overrides fall back to server config.
    depth      = (message.get("depth") or config.EVAL_AGENT_DEPTH).lower()
    panel_mode = (message.get("panel_mode") or config.EVAL_PANEL_MODE).lower()

    creds = await resolve_org_credentials(organization_id)
    usage = JudgeUsage()
    # Per-message judge selection. Falls back to the server default when absent
    # or unparseable, so an older SDK keeps working unchanged.
    judge_provider, judge_model = parse_judge_spec(message.get("judge"))
    try:
        judge = _build_judge(
            judge_provider, judge_model,
            org_id=organization_id, creds=creds, usage=usage,
        )
    except BYOKUnavailable as exc:
        logger.error(
            "[EVALUATOR] not evaluating org=%s trace_id=%s: %s",
            organization_id, trace_id, exc,
        )
        return
    panel = None
    if depth == "deep":
        panel = build_panel(
            primary=judge,
            # Caller-selected jury wins; the configured panel is the fallback,
            # so an org that does not pick one still gets the server default.
            member_specs=parse_jury_specs(message.get("jury")) or config.panel_members(),
            mode=panel_mode,
            gate_margin=config.EVAL_PANEL_GATE_MARGIN,
            threshold=config.JUDGE_THRESHOLD,
            # build_panel calls this as judge_factory(provider, model); bind the
            # org so jurors share the primary's tenant-scoped cache bucket, and
            # the credential set so each juror uses the org's key for *its* own
            # provider rather than the primary's.
            judge_factory=functools.partial(
                _build_judge, org_id=organization_id, creds=creds, usage=usage,
            ),
        )

    outcome = await asyncio.to_thread(
        evaluate_run, run, judge, config.JUDGE_THRESHOLD,
        depth=depth, panel=panel,
    )

    det = outcome["deterministic"]
    run_score  = outcome["run_score"]
    run_passed = outcome["run_passed"]
    metric_layers = outcome["metric_layers"]

    # Layer 1 gets its own queryable row (no judge cost), so dashboards can
    # separate deterministic pass-rate from judged quality.
    det_result = EvalResult(
        name="agentic.deterministic",
        score=float(det.get("score", 1.0)),
        passed=bool(det.get("error_calls", 0) == 0),
        reason=f"{det.get('error_calls', 0)} call(s) with errors of {det.get('total_calls', 0)}",
        details=det,
    )
    await _persist_eval_result(
        organization_id, api_key_prefix, trace_id,
        evaluator="fluiq.agent_eval",
        metric="agentic.deterministic",
        result=det_result,
        judge=judge,
        root_trace_id=root_trace_id,
        details_override=det,
        layer="deterministic",
        run_score=run_score,
        run_passed=run_passed,
    )

    for metric_name, result in outcome["metrics"].items():
        # Flatten: the evaluator's own details (panel, subgoals, efficiency,
        # goal_completion, agents, joins, …) go TOP-LEVEL in the CH details
        # column. The dashboard reads them flat (``details.panel`` &c.) — the
        # previous ``model_dump()`` shape nested them under details.details,
        # which left the jury trace invisible in the UI.
        details = dict(result.details or {})
        details["reason"]        = result.reason
        details["deterministic"] = det
        details["run_score"]     = run_score
        details["run_passed"]    = run_passed
        await _persist_eval_result(
            organization_id, api_key_prefix, trace_id,
            evaluator="fluiq.agent_eval",
            metric=metric_name,
            result=result,
            judge=judge,
            root_trace_id=root_trace_id,
            details_override=details,
            layer=metric_layers.get(metric_name, ""),
            run_score=run_score,
            run_passed=run_passed,
        )
    # Custom scorers apply to agentic runs too, graded against the run's final
    # answer with its goal as the question, so a dataset's scorers behave the
    # same whichever evaluator it runs.
    await _run_custom_judges(
        (message.get("eval_config") or {}).get("custom_judges") or {},
        organization_id=organization_id,
        api_key_prefix=api_key_prefix,
        trace_id=trace_id,
        judge=judge,
        question=run.goal or "",
        answer=run.final_output or "",
        context=run.goal or "",
        root_trace_id=root_trace_id,
        example_metadata=(message.get("eval_config") or {}).get("example_metadata"),
    )

    logger.info(
        "[EVALUATOR] agent_eval trace_id=%s source=%s depth=%s calls=%d "
        "run_score=%.3f det_errors=%d metrics=%s passed=%s",
        trace_id, run.source, outcome["depth"], outcome["tool_call_count"],
        run_score, det.get("error_calls", 0), list(outcome["metrics"]), run_passed,
    )


async def playground_eval(message: Dict[str, Any]) -> None:
    """Run a dashboard playground evaluation and reply to the API via Kafka.

    The API publishes this job with a ``correlation_id`` and awaits the reply
    on ``KAFKA_PLAYGROUND_REPLY_TOPIC``. All judge logic runs here in the
    worker — the API process never calls the judge directly.
    """
    correlation_id  = message.get("correlation_id")
    organization_id = message.get("organization_id")
    trace_id        = message.get("trace_id")
    prompt          = message.get("prompt") or ""
    response        = message.get("response") or ""
    ctx             = message.get("context") or ""
    metrics         = message.get("metrics") or ["hallucination", "relevance"]
    thresholds      = message.get("thresholds") or {}

    creds = await resolve_org_credentials(organization_id)
    usage = JudgeUsage()
    # Per-message judge selection. Falls back to the server default when absent
    # or unparseable, so an older SDK keeps working unchanged.
    judge_provider, judge_model = parse_judge_spec(message.get("judge"))
    try:
        judge = _build_judge(
            judge_provider, judge_model,
            org_id=organization_id, creds=creds, usage=usage,
        )
    except BYOKUnavailable as exc:
        logger.error(
            "[EVALUATOR] not evaluating org=%s trace_id=%s: %s",
            organization_id, trace_id, exc,
        )
        return

    results_list: List[Dict[str, Any]] = []
    scores:       Dict[str, float]     = {}
    failures:     List[str]            = []

    for metric_name in metrics:
        try:
            ev = _build_evaluator(metric_name, judge)
            kwargs: Dict[str, Any] = {"question": prompt, "answer": response}
            if ctx:
                kwargs["context"] = ctx
            result = await asyncio.to_thread(judge_prompts.captured_call, ev.evaluate, **kwargs)
            score     = float(result.score)
            threshold = float(thresholds.get(metric_name, 0.0))
            passed    = score >= threshold if threshold > 0 else True
            if not passed and threshold > 0:
                failures.append(metric_name)
            scores[metric_name] = score
            results_list.append({
                "metric": metric_name,
                "score":  score,
                "reason": result.reason or "",
                "passed": passed,
            })
            await _persist_eval_result(
                organization_id, None, trace_id,
                evaluator="fluiq.playground",
                metric=ev.name,
                result=result,
                judge=judge,
            )
            logger.info(
                "[EVALUATOR] playground_eval %s org=%s score=%.3f",
                metric_name, organization_id, score,
            )
        except ValueError:
            logger.warning("[EVALUATOR] Unknown metric %r in playground_eval", metric_name)
        except Exception:
            logger.exception(
                "[EVALUATOR] playground_eval failed metric=%s correlation_id=%s",
                metric_name, correlation_id,
            )

    if not correlation_id:
        logger.warning("[EVALUATOR] playground_eval missing correlation_id — reply skipped")
        return

    reply = {
        "correlation_id": correlation_id,
        "result": {
            "scores":   scores,
            "results":  results_list,
            "passed":   len(failures) == 0,
            "failures": failures,
        },
    }
    try:
        await kafka_producer.publish(
            reply,
            topic=config.KAFKA_PLAYGROUND_REPLY_TOPIC,
            key=str(organization_id) if organization_id is not None else None,
        )
    except Exception:
        logger.exception(
            "[EVALUATOR] Failed to publish playground reply correlation_id=%s", correlation_id,
        )


async def propose_scorers(message: Dict[str, Any]) -> None:
    """Draft custom LLM-as-judge scorers for a dataset from a sample of its
    examples — the 'eval engineering' step: given real data, suggest what's worth
    scoring. The API publishes this with a ``correlation_id`` and awaits the
    reply on ``KAFKA_PLAYGROUND_REPLY_TOPIC`` (the same channel playground uses).

    Best-effort: any failure replies with an empty proposal list plus a reason,
    so the caller shows a clean 'couldn't suggest' rather than hanging.
    """
    correlation_id  = message.get("correlation_id")
    organization_id = message.get("organization_id")
    examples        = message.get("examples") or []
    count           = max(1, min(6, int(message.get("count") or 4)))
    existing        = [str(n) for n in (message.get("existing") or []) if n]

    async def _reply(result: Dict[str, Any]) -> None:
        if not correlation_id:
            return
        try:
            await kafka_producer.publish(
                {"correlation_id": correlation_id, "result": result},
                topic=config.KAFKA_PLAYGROUND_REPLY_TOPIC,
                key=str(organization_id) if organization_id is not None else None,
            )
        except Exception:
            logger.exception(
                "[EVALUATOR] Failed to publish propose_scorers reply cid=%s", correlation_id,
            )

    creds = await resolve_org_credentials(organization_id)
    usage = JudgeUsage()
    judge_provider, judge_model = parse_judge_spec(message.get("judge"))
    try:
        judge = _build_judge(
            judge_provider, judge_model,
            org_id=organization_id, creds=creds, usage=usage,
        )
    except BYOKUnavailable as exc:
        await _reply({"proposals": [], "error": str(exc)})
        return

    sample_parts: List[str] = []
    for ex in examples[:12]:
        inp = str(ex.get("input") or "")[:600]
        out = str(ex.get("expected_output") or "")[:600]
        sample_parts.append(f"INPUT: {inp}\nEXPECTED OUTPUT: {out}")
    sample_block = "\n\n---\n\n".join(sample_parts) or "(no examples provided)"

    # Built by concatenation, not .format/f-string, so the literal JSON braces
    # and the {{answer}} placeholder we want the model to emit survive verbatim.
    meta = (
        "You are an eval engineer. From a sample of a dataset's INPUT / EXPECTED "
        "OUTPUT pairs, propose " + str(count) + " DISTINCT custom LLM-as-judge "
        "scorers. Each scorer checks ONE important, specific quality dimension "
        "the data suggests matters — for example: adherence to the expected "
        "format, faithfulness to the expected output, required fields present, "
        "tone, or safety. Make them concrete to THIS data, not generic. "
        "Do not duplicate these existing scorers: " + (", ".join(existing) or "none") + ". "
        "Every scorer's prompt MUST reference the {{answer}} placeholder (the "
        "output under test); it may also use {{question}} and {{context}}; and it "
        "MUST tell the model to return ONLY a JSON object "
        '{"score": <float 0..1>, "reason": "<short>"}. '
        "Return ONLY a JSON object: "
        '{"proposals": [{"name": "<short title>", "description": "<one line>", '
        '"prompt": "<the judge prompt>", "threshold": <float 0..1 pass cutoff>}]}.'
        "\n\nDATASET SAMPLE:\n" + sample_block
    )

    try:
        data = await asyncio.to_thread(judge.judge_json, meta)
    except Exception:
        logger.exception("[EVALUATOR] propose_scorers judge call failed cid=%s", correlation_id)
        await _reply({"proposals": [], "error": "The judge call failed."})
        return

    raw = data.get("proposals") if isinstance(data, dict) else None
    proposals: List[Dict[str, Any]] = []
    for p in (raw or [])[:count]:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        prompt = str(p.get("prompt") or "")
        # Drop anything the API would reject anyway (no answer placeholder).
        if not name or ("{{answer}}" not in prompt and "$answer" not in prompt):
            continue
        try:
            threshold = float(p.get("threshold"))
        except (TypeError, ValueError):
            threshold = 0.5
        threshold = min(1.0, max(0.0, threshold))
        proposals.append({
            "name":        name[:120],
            "description": str(p.get("description") or "")[:300],
            "prompt":      prompt,
            "threshold":   threshold,
        })

    logger.info(
        "[EVALUATOR] propose_scorers org=%s cid=%s proposed=%d",
        organization_id, correlation_id, len(proposals),
    )
    await _reply({"proposals": proposals})


async def _persist_eval_result(
    organization_id: Any,
    api_key_prefix: Any,
    trace_id: Any,
    *,
    evaluator: str,
    metric: str,
    result: Any,
    judge: Optional[LLMJudge],
    root_trace_id: Optional[str] = None,
    details_override: Optional[Dict[str, Any]] = None,
    layer: str = "",
    step_id: str = "",
    run_score: float = 0.0,
    run_passed: bool = True,
) -> None:
    """Persist one metric row and fan it out to the live trace stream.

    ``judge`` is None for scorers that make no model call (a deterministic code
    scorer). Those rows carry no judge model and no token spend, so a report that
    sums judge usage stays truthful instead of attributing a free scorer's result
    to whichever model happened to be configured.
    """
    details = details_override if details_override is not None else result.model_dump(mode="json")
    # Prompt provenance is attached to the evaluator's own details; lift it to
    # the top level of the CH details column so the dashboard reads it flat.
    nested = details.get("details")
    if "judge_prompts" not in details and isinstance(nested, dict) and nested.get("judge_prompts"):
        details["judge_prompts"] = nested.pop("judge_prompts")
    if judge is not None:
        details.setdefault("judge_provider", judge.provider)
    score = float(result.score)
    # Drained, not read: one message can persist several metric rows, and the
    # judge spend belongs to the message as a whole (a jury's calls are not
    # divisible per metric). The first row carries the totals and later rows
    # carry zeros, so SUM over rows is the true spend rather than a multiple.
    judge_in, judge_out, judge_calls = judge.usage.drain() if judge is not None else (0, 0, 0)
    judge_model = judge.model if judge is not None else ""
    record = {
        "organization_id": organization_id,
        "api_key_prefix":  api_key_prefix,
        "trace_id":        trace_id,
        "root_trace_id":   root_trace_id or trace_id,
        "evaluator":       evaluator,
        "metric":          metric,
        "score":           score,
        "judge_model":     judge_model,
        "judge_input_tokens":  judge_in,
        "judge_output_tokens": judge_out,
        "judge_calls":         judge_calls,
        "details":         details,
        "layer":           layer,
        "step_id":         step_id,
        "run_score":       run_score,
        "run_passed":      run_passed,
    }
    await clickhouse_eval_client.insert_evaluation(record)

    enriched = {
        "kind":             "enriched",
        "enrichment":       "evaluation",
        "organization_id":  str(organization_id) if organization_id is not None else None,
        "api_key_prefix":   api_key_prefix,
        "trace_id":         trace_id,
        "root_trace_id":    root_trace_id or trace_id,
        "evaluation": {
            "metric":      metric,
            "score":       score,
            "evaluator":   evaluator,
            "judge_model": judge_model,
            "details":     details,
        },
    }
    try:
        await kafka_producer.publish(
            enriched,
            key=str(organization_id) if organization_id is not None else None,
        )
    except Exception:
        logger.exception(
            "[EVALUATOR] Failed to publish trace.enriched (evaluation) trace_id=%s", trace_id,
        )

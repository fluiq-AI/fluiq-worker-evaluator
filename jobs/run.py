import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import config

from jobs.helper.base import EvalResult
from jobs.helper.judge import InMemoryCache, LLMJudge, PromptCache
from jobs.helper.hallucination import HallucinationEvaluator
from jobs.helper.ragas import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    Coherence,
    Toxicity,
    Ragas,
)
from db.clickhouse import clickhouse_eval_client
from db.kafka import kafka_producer

logger = logging.getLogger(__name__)

_judge_cache_backend = (
    InMemoryCache(max_size=config.JUDGE_CACHE_MAX) if config.JUDGE_CACHE_ENABLED else None
)

_RETRIEVAL_APIS = {
    "query", "search", "near_text", "near_vector",
    "hybrid", "bm25", "query_points", "fetch_objects",
}


def _build_judge(provider: Optional[str] = None, model: Optional[str] = None) -> LLMJudge:
    judge = LLMJudge(
        provider=provider or config.JUDGE_PROVIDER,
        model=model or config.JUDGE_MODEL,
    )
    if _judge_cache_backend is None:
        return judge

    def _underlying(prompt: str, **_params) -> str:
        if judge.provider == "openai":
            return judge._call_openai(prompt)
        if judge.provider == "anthropic":
            return judge._call_anthropic(prompt)
        if judge.provider == "gemini":
            return judge._call_gemini(prompt)
        if judge.provider == "fluiq":
            return judge._call_fluiq(prompt)
        raise RuntimeError(f"Unsupported judge provider: {judge.provider}")

    cache = PromptCache(
        _underlying,
        model=f"{judge.provider}:{judge.model}",
        backend=_judge_cache_backend,
        ttl=config.JUDGE_CACHE_TTL,
    )
    judge._judge_fn = lambda prompt: cache(prompt, temperature=judge.temperature)
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

    judge = _build_judge()

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
        result = await asyncio.to_thread(ev.evaluate, **inputs)
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

    judge   = _build_judge()
    started = time.time()
    try:
        ev = ContextPrecision(judge=judge, threshold=config.JUDGE_THRESHOLD)
        result = await asyncio.to_thread(
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

    judge = _build_judge()

    for metric_name in metrics:
        try:
            ev = _build_evaluator(metric_name, judge)
            # Pass the raw event so vision metrics can pull media refs from the
            # trace; text-only evaluators ignore the extra kwargs.
            result = await asyncio.to_thread(
                ev.evaluate, question=question, answer=answer, event=event,
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

    # Client-defined custom judges: resolve each slug to its saved judge template
    # (org-scoped, kind='judge') and score the answer with it. Best-effort —
    # missing template / PG down / judge error just skips that judge.
    if custom_judges:
        from db.postgres import postgres_client
        from jobs.helper.custom_judge import CustomJudgeEvaluator

        for slug, threshold in custom_judges.items():
            template = await postgres_client.fetch_custom_judge(organization_id, slug)
            if not template:
                logger.info(
                    "[EVALUATOR] custom judge %r unavailable for org=%s; skipping",
                    slug, organization_id,
                )
                continue
            try:
                ev = CustomJudgeEvaluator(
                    judge=judge, template=template, name=slug,
                    threshold=float(threshold or 0.0),
                )
                result = await asyncio.to_thread(ev.evaluate, question=question, answer=answer)
                await _persist_eval_result(
                    organization_id, api_key_prefix, trace_id,
                    evaluator="fluiq.eval",
                    metric=slug,
                    result=result,
                    judge=judge,
                )
                logger.info(
                    "[EVALUATOR] custom_judge %s org=%s trace_id=%s score=%.3f threshold=%.3f",
                    slug, organization_id, trace_id, result.score, float(threshold or 0.0),
                )
            except Exception:
                logger.exception(
                    "[EVALUATOR] custom judge failed slug=%s trace_id=%s", slug, trace_id,
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

    judge = _build_judge()
    panel = None
    if depth == "deep":
        panel = build_panel(
            primary=judge,
            member_specs=config.panel_members(),
            mode=panel_mode,
            gate_margin=config.EVAL_PANEL_GATE_MARGIN,
            threshold=config.JUDGE_THRESHOLD,
            judge_factory=_build_judge,
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

    judge = _build_judge()

    results_list: List[Dict[str, Any]] = []
    scores:       Dict[str, float]     = {}
    failures:     List[str]            = []

    for metric_name in metrics:
        try:
            ev = _build_evaluator(metric_name, judge)
            kwargs: Dict[str, Any] = {"question": prompt, "answer": response}
            if ctx:
                kwargs["context"] = ctx
            result = await asyncio.to_thread(ev.evaluate, **kwargs)
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


async def _persist_eval_result(
    organization_id: Any,
    api_key_prefix: Any,
    trace_id: Any,
    *,
    evaluator: str,
    metric: str,
    result: Any,
    judge: LLMJudge,
    root_trace_id: Optional[str] = None,
    details_override: Optional[Dict[str, Any]] = None,
    layer: str = "",
    step_id: str = "",
    run_score: float = 0.0,
    run_passed: bool = True,
) -> None:
    details = details_override if details_override is not None else result.model_dump(mode="json")
    details.setdefault("judge_provider", judge.provider)
    score = float(result.score)
    record = {
        "organization_id": organization_id,
        "api_key_prefix":  api_key_prefix,
        "trace_id":        trace_id,
        "root_trace_id":   root_trace_id or trace_id,
        "evaluator":       evaluator,
        "metric":          metric,
        "score":           score,
        "judge_model":     judge.model,
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
            "judge_model": judge.model,
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

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import config

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


def _build_judge() -> LLMJudge:
    judge = LLMJudge(provider=config.JUDGE_PROVIDER, model=config.JUDGE_MODEL)
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

    metrics    = eval_config.get("metrics") or ["hallucination", "relevance"]
    thresholds = eval_config.get("thresholds") or {}

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
            result = await asyncio.to_thread(ev.evaluate, question=question, answer=answer)
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

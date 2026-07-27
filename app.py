import asyncio
import json
import logging
import config
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

from aiokafka import AIOKafkaConsumer, TopicPartition

from db.clickhouse import clickhouse_eval_client
from db.kafka import kafka_producer
from db.postgres import postgres_client
from jobs.helper import judge_prompts
from jobs.run import (
    agent_evaluate,
    auto_evaluate_retrieval,
    auto_llm_eval,
    playground_eval,
    propose_scorers,
    run_evaluation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seek-back retry pacing for transient (connection-class) failures.
_RETRY_INITIAL_S = 1.0
_RETRY_MAX_S = 30.0


def _is_transient(exc: BaseException) -> bool:
    """True for connection-class failures to a backing store (ClickHouse,
    Postgres, Kafka, network). The message itself is fine — the caller seeks
    back to its offset and retries, instead of letting a later ``commit()``
    advance the position past it (which silently loses the message).

    Walks the cause/context chain because drivers wrap the underlying network
    error (e.g. clickhouse_connect raises OperationalError *from* an aiohttp
    ClientConnectorError).
    """
    seen: set[int] = set()
    e: BaseException | None = exc
    depth = 0
    while e is not None and id(e) not in seen and depth < 10:
        seen.add(id(e))
        depth += 1
        if isinstance(e, (OSError, TimeoutError, asyncio.TimeoutError)):
            return True
        mod = type(e).__module__ or ""
        name = type(e).__name__
        if mod.startswith("clickhouse_connect") and name in ("OperationalError", "NetworkError"):
            return True
        if mod.startswith("aiohttp"):
            return True
        if mod.startswith("asyncpg") and (
            "Connect" in name or "TooManyConnections" in name or name == "InterfaceError"
        ):
            return True
        if mod.startswith(("aiokafka", "kafka")) and (
            "Connection" in name or "Timeout" in name or "NodeNotReady" in name
        ):
            return True
        e = e.__cause__ or e.__context__
    return False


async def _start_with_retry(attempt: Callable[[], Awaitable[Any]], what: str) -> Any:
    """Bring up one startup dependency, retrying with backoff until reachable.

    Replaces the boot crash-loop (raise → process exit → docker restart →
    repeat until the store is up) with an in-process wait: a worker's job is to
    outlast its stores, not die with them.
    """
    backoff = _RETRY_INITIAL_S
    while True:
        try:
            return await attempt()
        except Exception as exc:
            logger.warning(
                "[EVALUATOR] %s not ready at startup (%s) — retrying in %.1fs",
                what, type(exc).__name__, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RETRY_MAX_S)


OPERATIONS: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {
    "evaluate":              run_evaluation,
    "auto":                  auto_evaluate_retrieval,
    "sdk_llm":               auto_llm_eval,
    "playground_eval":       playground_eval,
    "propose_scorers":       propose_scorers,
    "agent_eval":            agent_evaluate,
}


def _looks_agentic(message: dict[str, Any]) -> bool:
    """Heuristic routing for messages with no explicit ``operation``: a trace
    that carries tool/MCP activity (or a normalized run / OTel spans) is an
    agentic eval, not a plain retrieval auto-eval."""
    if message.get("agent_run") or message.get("spans") or message.get("events"):
        return True
    event = message.get("event")
    if isinstance(event, dict):
        return any(
            event.get(k) for k in ("tool_calls", "tool_uses", "function_calls", "mcp_calls")
        )
    return False


async def dispatch(message: dict[str, Any]) -> None:
    # Messages from the API fan-out have no `operation` field — they're raw
    # trace envelopes. Route by presence of well-known fields:
    #   explicit evaluator  → run_evaluation
    #   eval_config present → sdk_llm
    #   everything else     → auto (vectorstore ContextPrecision)
    operation = message.get("operation")
    if operation is None:
        if message.get("evaluator"):
            operation = "evaluate"
        elif message.get("eval_config"):
            operation = "sdk_llm"
        elif _looks_agentic(message):
            operation = "agent_eval"
        else:
            operation = "auto"
    handler = OPERATIONS.get(operation)
    if handler is None:
        logger.warning("[EVALUATOR] Unknown operation: %s", operation)
        return
    # Pull any admin-edited judge-prompt overrides into the in-process snapshot.
    # TTL-gated and best-effort, so this is a cheap no-op on most messages and
    # never blocks evaluation if Postgres is slow or unset.
    await judge_prompts.refresh()
    # Select this message's org so per-org prompt overrides resolve during
    # rendering. Safe as a per-message global: messages are processed serially.
    judge_prompts.set_org(message.get("organization_id"))
    # Per-run prompt overrides ride on the job itself (a dataset's own metric
    # prompts), and outrank the org/platform layers for this message only.
    judge_prompts.set_run_overrides(
        (message.get("eval_config") or {}).get("judge_prompt_overrides")
    )
    try:
        await handler(message)
    finally:
        judge_prompts.set_org(None)
        judge_prompts.set_run_overrides(None)


async def consume() -> None:

    async def _consumer_attempt() -> AIOKafkaConsumer:
        # Recreate the consumer per attempt: a failed AIOKafkaConsumer.start()
        # can leave partial client state, so retrying a fresh instance is the
        # only clean path.
        c = AIOKafkaConsumer(
            config.KAFKA_EVAL_TOPIC,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_EVAL_GROUP_ID,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_partition_fetch_bytes=config.KAFKA_MAX_FETCH_BYTES,
            **config.kafka_auth_kwargs(),
        )
        try:
            await c.start()
            return c
        except Exception:
            try:
                await c.stop()
            except Exception:
                pass
            raise

    # Pin all blocking work (asyncio.to_thread → run_scan, judge calls) to a
    # single executor thread. Messages are processed one at a time, so no
    # parallelism is lost — and keeping torch/spaCy on one thread means one
    # glibc malloc arena instead of one per default-pool thread, which is the
    # main driver of RSS growth on small instances.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-worker")
    )

    consumer = await _start_with_retry(_consumer_attempt, "kafka consumer")
    await _start_with_retry(clickhouse_eval_client.start, "clickhouse")
    await _start_with_retry(kafka_producer.start, "kafka producer")
    # Judge-prompt overrides (optional): connect, seed canonical defaults, and
    # prime the snapshot. Genuinely best-effort — a Postgres outage must not
    # keep the evaluator down; everything falls back to the built-in prompts.
    try:
        await postgres_client.start()
        await judge_prompts.seed()
        await judge_prompts.refresh(force=True)
    except Exception:
        logger.exception(
            "[EVALUATOR] Judge-prompt overrides unavailable at startup; using built-in defaults",
        )
    logger.info(
        "[EVALUATOR] Consuming topic=%s group=%s servers=%s",
        config.KAFKA_EVAL_TOPIC, config.KAFKA_EVAL_GROUP_ID, config.KAFKA_BOOTSTRAP_SERVERS,
    )
    try:
        backoff = _RETRY_INITIAL_S
        retries = 0
        async for msg in consumer:
            try:
                await dispatch(msg.value)
                await consumer.commit()
                backoff, retries = _RETRY_INITIAL_S, 0
            except Exception as exc:
                if _is_transient(exc):
                    # Backing store unreachable — the message is fine. Seek back
                    # to it and retry with backoff so a later commit() can never
                    # advance past it (that was silent data loss). Blocks the
                    # partition until the dependency recovers; Kafka retention
                    # holds the backlog.
                    retries += 1
                    consumer.seek(TopicPartition(msg.topic, msg.partition), msg.offset)
                    log = logger.error if retries % 20 == 0 else logger.warning
                    log(
                        "[EVALUATOR] Transient %s at offset=%s partition=%s — seeking back, "
                        "retry #%d in %.1fs",
                        type(exc).__name__, msg.offset, msg.partition, retries, backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, _RETRY_MAX_S)
                else:
                    # Poison message — retrying won't help. Skip it DELIBERATELY
                    # by committing past it (previously this skip happened as an
                    # accident of the next successful commit).
                    logger.exception(
                        "[EVALUATOR] Failed to process message offset=%s partition=%s — skipping",
                        msg.offset, msg.partition,
                    )
                    try:
                        await consumer.commit()
                    except Exception:
                        logger.warning(
                            "[EVALUATOR] Commit after poison skip failed; message may re-deliver on restart",
                        )
                    backoff, retries = _RETRY_INITIAL_S, 0
    finally:
        await consumer.stop()
        await kafka_producer.stop()
        await clickhouse_eval_client.stop()
        await postgres_client.stop()


def main() -> None:
    try:
        asyncio.run(consume())
    except KeyboardInterrupt:
        logger.info("[EVALUATOR] Shutdown requested")


if __name__ == "__main__":
    main()

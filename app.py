import asyncio
import json
import logging
import config
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable

from aiokafka import AIOKafkaConsumer

from db.clickhouse import clickhouse_eval_client
from db.kafka import kafka_producer
from db.postgres import postgres_client
from jobs.helper import judge_prompts
from jobs.run import auto_evaluate_retrieval, auto_llm_eval, playground_eval, run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OPERATIONS: dict[str, Callable[[dict[str, Any]], Awaitable[None]]] = {
    "evaluate":              run_evaluation,
    "auto":                  auto_evaluate_retrieval,
    "sdk_llm":               auto_llm_eval,
    "playground_eval":       playground_eval,
}


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
    await handler(message)


async def consume() -> None:

    consumer = AIOKafkaConsumer(
        config.KAFKA_EVAL_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        group_id=config.KAFKA_EVAL_GROUP_ID,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_partition_fetch_bytes=config.KAFKA_MAX_FETCH_BYTES,
        **config.kafka_auth_kwargs(),
    )
    # Pin all blocking work (asyncio.to_thread → run_scan, judge calls) to a
    # single executor thread. Messages are processed one at a time, so no
    # parallelism is lost — and keeping torch/spaCy on one thread means one
    # glibc malloc arena instead of one per default-pool thread, which is the
    # main driver of RSS growth on small instances.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-worker")
    )

    await consumer.start()
    await clickhouse_eval_client.start()
    await kafka_producer.start()
    # Judge-prompt overrides (optional): connect, seed canonical defaults, and
    # prime the snapshot. All best-effort — failures fall back to built-ins.
    await postgres_client.start()
    await judge_prompts.seed()
    await judge_prompts.refresh(force=True)
    logger.info(
        "[EVALUATOR] Consuming topic=%s group=%s servers=%s",
        config.KAFKA_EVAL_TOPIC, config.KAFKA_EVAL_GROUP_ID, config.KAFKA_BOOTSTRAP_SERVERS,
    )
    try:
        async for msg in consumer:
            try:
                await dispatch(msg.value)
                await consumer.commit()
            except Exception:
                logger.exception(
                    "[EVALUATOR] Failed to process message offset=%s partition=%s",
                    msg.offset, msg.partition,
                )
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

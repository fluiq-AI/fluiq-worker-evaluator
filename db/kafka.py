import json
import logging
from typing import Any, Optional
import config
from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)


class KafkaProducer:
    """Async Kafka producer used by the evaluator to fan out trace.enriched
    notifications (evaluation results) to the API process for SSE streaming.

    Reuses the same ``traces.persisted`` topic as the tracer; the SSE route
    routes by the ``kind`` field on each payload.
    """

    def __init__(
        self,
        bootstrap_servers: str = config.KAFKA_BOOTSTRAP_SERVERS,
        default_topic: str = config.KAFKA_TRACE_PERSISTED_TOPIC,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.default_topic = default_topic
        self._producer: Optional[AIOKafkaProducer] = None

    async def start(self) -> None:
        if self._producer is not None:
            return

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            acks="all",
            **config.kafka_auth_kwargs(),
            enable_idempotence=True,
        )
        await self._producer.start()
        logger.info("[KAFKA] Evaluator producer started: %s", self.bootstrap_servers)

    async def stop(self) -> None:
        if self._producer is None:
            return
        await self._producer.stop()
        self._producer = None
        logger.info("[KAFKA] Evaluator producer stopped")

    async def publish(
        self,
        message: dict[str, Any],
        topic: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        if self._producer is None:
            await self.start()
        await self._producer.send_and_wait(
            topic or self.default_topic,
            value=message,
            key=key,
        )


kafka_producer = KafkaProducer()

__all__ = [
    "KafkaProducer",
    "kafka_producer",
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_TRACE_PERSISTED_TOPIC",
]

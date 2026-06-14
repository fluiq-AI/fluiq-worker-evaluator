import logging
from typing import Any, Optional
import config
import clickhouse_connect
from clickhouse_connect.driver.asyncclient import AsyncClient


logger = logging.getLogger(__name__)



class ClickHouseEvalClient:
    """Async ClickHouse client for storing evaluation results."""

    def __init__(
        self,
        host: str = config.CLICKHOUSE_HOST,
        port: int = config.CLICKHOUSE_PORT,
        username: str = config.CLICKHOUSE_USER,
        password: str = config.CLICKHOUSE_PASSWORD,
        database: str = config.CLICKHOUSE_DATABASE,
        default_table: str = config.CLICKHOUSE_EVALUATIONS_TABLE,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.default_table = default_table
        self._client: Optional[AsyncClient] = None

    async def start(self) -> None:
        if self._client is not None:
            return
        self._client = await clickhouse_connect.get_async_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
        )
        logger.info(
            "[CLICKHOUSE] Eval client started: %s:%s/%s",
            self.host, self.port, self.database,
        )

    async def stop(self) -> None:
        if self._client is None:
            return
        await self._client.close()
        self._client = None
        logger.info("[CLICKHOUSE] Eval client stopped")

    async def get_session_risk_scores(
        self,
        organization_id: Any,
        root_trace_id: Any,
        exclude_trace_id: Any,
        table: Optional[str] = None,
    ) -> list[float]:
        """Return ordered security_risk_score values for the session, excluding the current trace."""
        if self._client is None:
            await self.start()
        target = table or config.CLICKHOUSE_SECURITY_TABLE
        result = await self._client.query(
            f"SELECT security_risk_score FROM {target} "
            f"WHERE organization_id = {{org_id:UUID}} "
            f"AND root_trace_id = {{root_id:UUID}} "
            f"AND trace_id != {{exc_id:UUID}} "
            f"ORDER BY ingested_at ASC LIMIT 19",
            parameters={
                "org_id": str(organization_id),
                "root_id": str(root_trace_id),
                "exc_id": str(exclude_trace_id),
            },
        )
        return [float(row[0]) for row in result.result_rows]

    async def insert_security_scan(
        self,
        record: dict[str, Any],
        table: Optional[str] = None,
    ) -> None:
        if self._client is None:
            await self.start()
        target = table or config.CLICKHOUSE_SECURITY_TABLE
        await self._client.insert(
            target,
            [[
                record.get("organization_id"),
                record.get("api_key_prefix") or "",
                record.get("trace_id"),
                record.get("root_trace_id") or record.get("trace_id"),
                record.get("mode") or "warn",
                record.get("prompt_redacted") or "",
                record.get("response_redacted") or "",
                record.get("pii_entities_prompt") or [],
                record.get("pii_entities_response") or [],
                int(bool(record.get("injection_detected"))),
                record.get("injection_patterns") or [],
                int(bool(record.get("jailbreak_detected"))),
                record.get("jailbreak_patterns") or [],
                int(bool(record.get("skeleton_key_detected"))),
                record.get("skeleton_key_patterns") or [],
                int(bool(record.get("secrets_detected"))),
                record.get("secret_types") or [],
                int(bool(record.get("indirect_injection_detected"))),
                record.get("indirect_injection_sources") or [],
                float(record.get("semantic_attack_score") or 0.0),
                record.get("security_risk_level") or "clean",
                float(record.get("security_risk_score") or 0.0),
                int(bool(record.get("should_block"))),
                float(record.get("scan_latency") or 0.0),
                record.get("extra") or {},
            ]],
            column_names=[
                "organization_id", "api_key_prefix", "trace_id", "root_trace_id",
                "mode", "prompt_redacted", "response_redacted",
                "pii_entities_prompt", "pii_entities_response",
                "injection_detected", "injection_patterns",
                "jailbreak_detected", "jailbreak_patterns",
                "skeleton_key_detected", "skeleton_key_patterns",
                "secrets_detected", "secret_types",
                "indirect_injection_detected", "indirect_injection_sources",
                "semantic_attack_score", "security_risk_level",
                "security_risk_score", "should_block", "scan_latency", "extra",
            ],
        )

    async def insert_evaluation(
        self,
        record: dict[str, Any],
        table: Optional[str] = None,
    ) -> None:
        if self._client is None:
            await self.start()
        target = table or self.default_table
        trace_id = record.get("trace_id")
        root_trace_id = record.get("root_trace_id") or trace_id
        try:
            score = float(record.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        await self._client.insert(
            target,
            [[
                record.get("organization_id"),
                record.get("api_key_prefix") or "",
                trace_id,
                root_trace_id,
                record.get("evaluator") or "",
                record.get("metric") or "",
                score,
                record.get("judge_model") or "",
                record.get("details") or {},
            ]],
            column_names=[
                "organization_id",
                "api_key_prefix",
                "trace_id",
                "root_trace_id",
                "evaluator",
                "metric",
                "score",
                "judge_model",
                "details",
            ],
        )


clickhouse_eval_client = ClickHouseEvalClient()

__all__ = [
    "ClickHouseEvalClient",
    "clickhouse_eval_client",
    "CLICKHOUSE_HOST",
    "CLICKHOUSE_PORT",
    "CLICKHOUSE_DATABASE",
    "CLICKHOUSE_EVAL_TABLE",
]

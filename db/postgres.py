"""Minimal async Postgres pool for the evaluator worker.

Only used to read (and seed) the admin-editable LLM-as-Judge prompt overrides
in ``eval_judge_prompts``. Everything here is best-effort: if the DSN is unset
or Postgres is unreachable, callers fall back to the built-in default prompts
so evaluations never break because of this table.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import asyncpg

import config

logger = logging.getLogger(__name__)


class PostgresClient:
    def __init__(self, dsn: Optional[str] = None) -> None:
        self.dsn = dsn if dsn is not None else config.POSTGRES_DSN
        self._pool: Optional[asyncpg.Pool] = None

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    async def start(self) -> None:
        if self._pool is not None or not self.enabled:
            return
        try:
            self._pool = await asyncpg.create_pool(dsn=self.dsn, min_size=1, max_size=4)
            logger.info("[EVALUATOR][PG] judge-prompt pool started")
        except Exception:
            logger.exception("[EVALUATOR][PG] pool start failed; using default prompts")
            self._pool = None

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def seed_judge_prompts(self, rows: list[dict[str, Any]]) -> None:
        """Insert canonical defaults, never clobbering an existing (edited) row."""
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO eval_judge_prompts
                        (name, template, default_template, description, required_vars)
                    VALUES ($1, $2, $2, $3, $4::jsonb)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    [
                        (r["name"], r["template"], r.get("description"), r["required_vars"])
                        for r in rows
                    ],
                )
            logger.info("[EVALUATOR][PG] seeded %d judge prompt defaults", len(rows))
        except Exception:
            logger.exception("[EVALUATOR][PG] seed failed; using default prompts")

    async def fetch_custom_judge(self, organization_id: Any, slug: str) -> Optional[str]:
        """Return the live template for an org's client-defined judge prompt, or None.

        Resolves a ``custom_judges`` slug from ``fluiq.eval()`` to the template saved
        on the Prompts page (``kind = 'judge'``). Best-effort: returns None when the
        pool is down or the slug doesn't resolve, so the metric is simply skipped.
        """
        if self._pool is None:
            return None
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT template FROM prompts "
                    "WHERE org_id = $1::uuid AND slug = $2 AND kind = 'judge'",
                    str(organization_id), slug,
                )
            return row["template"] if row else None
        except Exception:
            logger.exception("[EVALUATOR][PG] custom judge fetch failed slug=%s", slug)
            return None

    async def fetch_judge_prompts(self) -> list[dict[str, Any]]:
        """Return [{name, template, required_vars}] or [] when unavailable."""
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                records = await conn.fetch(
                    "SELECT name, template, required_vars FROM eval_judge_prompts"
                )
            return [dict(r) for r in records]
        except Exception:
            logger.exception("[EVALUATOR][PG] fetch failed; using cached/default prompts")
            return []


postgres_client = PostgresClient()

__all__ = ["postgres_client", "PostgresClient"]

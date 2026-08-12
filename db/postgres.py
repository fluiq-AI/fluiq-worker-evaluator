"""Minimal async Postgres pool for the evaluator worker.

Only used to read (and seed) the admin-editable LLM-as-Judge prompt overrides
in ``eval_judge_prompts``. Everything here is best-effort: if the DSN is unset
or Postgres is unreachable, callers fall back to the built-in default prompts
so evaluations never break because of this table.
"""
from __future__ import annotations

import logging
import os
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
        """Seed canonical defaults, refreshing them without clobbering edits.

        ``DO NOTHING`` was wrong here: once a row existed, every later
        improvement to the shipped prompt was inert in production, because the
        row is only ever written on first boot. The API's own seeder
        (``db_queues/postgresql/eval_prompts.py``) already had the correct
        semantics; this mirrors them so the two agree:

        - ``default_template`` always tracks the code default, so "reset to
          platform default" in Admin restores the *current* prompt, not the one
          that happened to be shipping the day the row was created.
        - ``template`` moves with it only while ``is_overridden`` is false. An
          org that edited its prompt keeps that edit untouched.
        """
        if self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO eval_judge_prompts
                        (name, template, default_template, description, required_vars)
                    VALUES ($1, $2, $2, $3, $4::jsonb)
                    ON CONFLICT (name) DO UPDATE SET
                        default_template = EXCLUDED.default_template,
                        description      = EXCLUDED.description,
                        required_vars    = EXCLUDED.required_vars,
                        template = CASE
                            WHEN eval_judge_prompts.is_overridden
                            THEN eval_judge_prompts.template
                            ELSE EXCLUDED.template
                        END
                    WHERE eval_judge_prompts.default_template
                              IS DISTINCT FROM EXCLUDED.default_template
                       OR eval_judge_prompts.description
                              IS DISTINCT FROM EXCLUDED.description
                       OR eval_judge_prompts.required_vars
                              IS DISTINCT FROM EXCLUDED.required_vars
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

    async def fetch_org_credentials(
        self, organization_id: Any,
    ) -> Optional[dict[str, dict[str, Any]]]:
        """Return this org's provider credentials keyed by provider, or None.

        The return value is deliberately three-valued, because "this org has no
        BYOK key" and "we could not find out" must not be treated the same way:

            ``{...}``  rows found (may include non-active ones — the caller
                       decides, because an invalid key must stop the eval
                       rather than silently fall back to Fluiq's account)
            ``{}``     the org definitively has no credentials → managed keys
            ``None``   Postgres is unreachable or the table is missing → we
                       cannot tell, so the caller falls back to managed keys
                       and logs it

        That last case is the one deliberate fail-open here: a Postgres blip
        should not stop every eval in the fleet. The cost is that Fluiq briefly
        absorbs token spend for BYOK orgs, which is bounded and noisy in the
        logs rather than silent.

        Returns sealed rows, not plaintext — decryption is the caller's job so
        this layer never handles key material.
        """
        if self._pool is None:
            return None
        table = os.getenv("POSTGRES_CREDENTIALS_TABLE", "org_provider_credentials")
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT credential_id, provider, status, ciphertext, nonce, "
                    f"wrapped_dek, key_version FROM {table} WHERE org_id = $1::uuid",
                    str(organization_id),
                )
        except asyncpg.UndefinedTableError:
            # BYOK not deployed yet on this environment — that is "no
            # credentials", not "unknown".
            return {}
        except Exception:
            logger.exception(
                "[EVALUATOR][PG] credential fetch failed org=%s", organization_id,
            )
            return None
        return {r["provider"]: dict(r) for r in rows}

    async def mark_credential_invalid(
        self, organization_id: Any, credential_id: Any, error: str,
    ) -> None:
        """Flag a BYOK credential the provider rejected at eval time.

        Without this, a key revoked upstream fails every evaluation while the
        dashboard still shows it Active — the customer has no way to learn that
        the thing to fix is their key. Deliberately does not delete the row:
        they need to see *which* key broke in order to rotate it.
        """
        if self._pool is None:
            return
        table = os.getenv("POSTGRES_CREDENTIALS_TABLE", "org_provider_credentials")
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"UPDATE {table} SET status = 'invalid', last_error = $3, "
                    f"updated_at = NOW() "
                    f"WHERE org_id = $1::uuid AND credential_id = $2::uuid "
                    f"  AND status = 'active'",
                    str(organization_id), str(credential_id), (error or "")[:500],
                )
            logger.warning(
                "[EVALUATOR][PG] credential marked invalid org=%s credential=%s",
                organization_id, credential_id,
            )
        except Exception:
            logger.exception(
                "[EVALUATOR][PG] could not mark credential invalid org=%s", organization_id,
            )

    async def fetch_judge_prompts(self) -> list[dict[str, Any]]:
        """Return [{name, template, required_vars, version, is_overridden}] or []."""
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                records = await conn.fetch(
                    "SELECT name, template, required_vars, version, is_overridden "
                    "FROM eval_judge_prompts"
                )
            return [dict(r) for r in records]
        except Exception:
            logger.exception("[EVALUATOR][PG] fetch failed; using cached/default prompts")
            return []

    async def fetch_org_judge_prompts(self) -> list[dict[str, Any]]:
        """Return per-org judge-prompt overrides, or [] when unavailable.

        The table ships with a later API deploy than this worker may be running
        against, so a missing relation is expected and stays quiet.
        """
        if self._pool is None:
            return []
        try:
            async with self._pool.acquire() as conn:
                records = await conn.fetch(
                    "SELECT org_id::text AS org_id, name, template, version "
                    "FROM eval_judge_prompt_org_overrides"
                )
            return [dict(r) for r in records]
        except asyncpg.UndefinedTableError:
            return []
        except Exception:
            logger.exception("[EVALUATOR][PG] org-override fetch failed; using cached prompts")
            return []


postgres_client = PostgresClient()

__all__ = ["postgres_client", "PostgresClient"]

"""
User SQL database connector.

Safety model (layered defense-in-depth):
  1. PRIMARY: DB-level read-only role. The user should connect with a read-only
     DB user — this is the strongest guarantee, enforced by the database itself.
     Document this clearly in the README.
  2. SECONDARY: Statement-type rejection — any non-SELECT is rejected before
     execution. Catches accidental mutations even with a read-only role.
  3. TERTIARY: EXPLAIN-before-execute — catches syntactically malformed queries
     before they touch the database.

IMPORTANT CAVEAT (documented in code, not hidden):
  EXPLAIN and statement-type checks only catch *unsafe or malformed* queries.
  They cannot detect a syntactically valid query that joins the wrong table or
  filters the wrong column — that failure mode produces a confidently-cited
  wrong answer with no error signal. Mitigation is TRANSPARENCY: the generated
  SQL is always shown to the user in the chat UI so a human can sanity-check it.
  Additionally, the schema prompt is seeded with few-shot Q→SQL examples to
  reduce wrong-table/wrong-column mistakes at generation time.

CLOUD-DEPENDENCY AUDIT: No external APIs called. All DB traffic stays on
localhost or the Docker network. Credentials never reach the frontend or logs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

# Non-SELECT keywords that indicate a mutation or DDL statement.
# This is a secondary check; the DB role is the primary safety boundary.
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|EXECUTE|EXEC|CALL|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Dialects that support EXPLAIN; others fall through to keyword-only check.
_EXPLAIN_SUPPORTED = {"postgresql", "mysql", "sqlite"}


class SQLConnector:
    """
    Manages a single user-configured SQL connection for the lifetime of the app.
    Thread-safe; the underlying async engine uses a connection pool.
    """

    def __init__(self, connection_string: str) -> None:
        # Redact credentials from the stored form — never log the raw string.
        self._redacted = self._redact_credentials(connection_string)
        self._engine: AsyncEngine = create_async_engine(
            connection_string,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        self._dialect = self._engine.dialect.name

    @staticmethod
    def _redact_credentials(conn_str: str) -> str:
        """Replace password in connection string for logging."""
        return re.sub(r"(?<=://)([^:]+):([^@]+)@", r"\1:***@", conn_str)

    @property
    def redacted_url(self) -> str:
        return self._redacted

    @property
    def connection_hash(self) -> str:
        """Stable hash of the connection string for schema cache keying."""
        return hashlib.sha256(str(self._engine.url).encode()).hexdigest()[:16]

    async def test_connection(self) -> dict[str, Any]:
        """Verify connectivity. Returns status dict."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(sa.text("SELECT 1"))
            return {"ok": True, "dialect": self._dialect, "url": self._redacted}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "url": self._redacted}

    async def introspect_schema(self) -> dict[str, Any]:
        """
        Reflect all tables, columns, types, and foreign keys.
        Returns a structured dict suitable for injection into the LLM prompt.
        """
        async with self._engine.connect() as conn:
            def _reflect(sync_conn):
                meta = sa.MetaData()
                meta.reflect(bind=sync_conn)
                tables = {}
                for table_name, table in meta.tables.items():
                    columns = []
                    for col in table.columns:
                        columns.append({
                            "name": col.name,
                            "type": str(col.type),
                            "nullable": col.nullable,
                            "primary_key": col.primary_key,
                        })
                    foreign_keys = []
                    for fk in table.foreign_keys:
                        foreign_keys.append({
                            "column": fk.parent.name,
                            "references": f"{fk.column.table.name}.{fk.column.name}",
                        })
                    tables[table_name] = {
                        "columns": columns,
                        "foreign_keys": foreign_keys,
                    }
                return tables

            tables = await conn.run_sync(_reflect)
        return {"dialect": self._dialect, "tables": tables}

    def validate_query(self, sql: str) -> tuple[bool, str]:
        """
        Secondary safety check: reject non-SELECT statements.
        Returns (is_safe, reason).
        Note: this does NOT catch semantically wrong queries (wrong table, etc).
        """
        stripped = sql.strip().rstrip(";")
        match = _FORBIDDEN_KEYWORDS.search(stripped)
        if match:
            return False, f"Rejected: query contains forbidden keyword '{match.group()}'"
        if not stripped.upper().startswith("SELECT"):
            return False, "Rejected: only SELECT statements are permitted"
        return True, "ok"

    async def explain_query(self, sql: str) -> tuple[bool, str]:
        """
        Tertiary check: run EXPLAIN to catch syntactically invalid queries.
        Returns (is_valid, error_message_or_empty).
        Falls back to a no-op for unsupported dialects.
        """
        if self._dialect not in _EXPLAIN_SUPPORTED:
            return True, ""
        explain_prefix = {
            "postgresql": "EXPLAIN ",
            "mysql": "EXPLAIN ",
            "sqlite": "EXPLAIN QUERY PLAN ",
        }
        prefix = explain_prefix.get(self._dialect, "EXPLAIN ")
        try:
            async with self._engine.connect() as conn:
                await conn.execute(sa.text(prefix + sql))
            return True, ""
        except Exception as exc:
            return False, str(exc)

    async def execute_read_only(
        self, sql: str, retry_sql: str | None = None
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Execute a SELECT query through all safety layers.
        Returns (rows, sql_actually_executed).

        If the query errors, we allow one LLM-supplied retry SQL.
        After that we give up — we do NOT loop (avoids infinite retry cycles).

        Raises ValueError for safety rejections.
        Raises RuntimeError for query execution failures after retry.
        """
        for attempt, current_sql in enumerate([sql, retry_sql]):
            if current_sql is None:
                break

            # Layer 2: statement-type check
            is_safe, reason = self.validate_query(current_sql)
            if not is_safe:
                raise ValueError(reason)

            # Layer 3: EXPLAIN validation
            is_valid, explain_error = await self.explain_query(current_sql)
            if not is_valid:
                if attempt == 0 and retry_sql:
                    continue  # try the retry SQL
                raise RuntimeError(f"Query validation failed: {explain_error}")

            # Execute
            try:
                async with self._engine.connect() as conn:
                    result = await conn.execute(sa.text(current_sql))
                    rows = [dict(zip(result.keys(), row)) for row in result.fetchall()]
                return rows, current_sql
            except Exception as exc:
                if attempt == 0 and retry_sql:
                    continue  # let the retry SQL have a chance
                raise RuntimeError(
                    f"Query execution failed after retry: {exc}"
                ) from exc

        raise RuntimeError("Query execution failed and no valid retry was provided")

    async def dispose(self) -> None:
        await self._engine.dispose()


# ── Module-level singleton management ────────────────────────────────────────

_connector: SQLConnector | None = None


def set_connector(conn_str: str) -> SQLConnector:
    global _connector
    # Synchronously swap — the old engine's pool will be GC'd.
    # We can't await dispose() here (sync context), so we let the pool close
    # naturally. For a single-user local app this is acceptable.
    _connector = SQLConnector(conn_str)
    return _connector


def get_connector() -> SQLConnector | None:
    return _connector

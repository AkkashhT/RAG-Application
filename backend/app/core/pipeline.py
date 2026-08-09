"""
RAG pipeline — the core query processing logic.

Pipeline shape (for document path):
  1. Embed query (Ollama, nomic-embed-text)
  2. Hybrid search top-20 from Qdrant (dense + BM25, RRF fusion)
  3. Rerank top-20 → top-5 (BGE cross-encoder)
  4. Confidence gate: if top reranker score < threshold, skip LLM entirely
  5. Build context window from top chunks
  6. Stream generation from Ollama (qwen3:14b)
  7. Return structured citations

For SQL path:
  1. LLM generates SQL from schema + few-shot examples
  2. Validate (statement type + EXPLAIN)
  3. Execute; on error, feed error back to LLM for one retry
  4. Confidence gate: if no schema match, skip LLM
  5. Stream generation with SQL results as context
  6. Return SQL query alongside citations (always shown to user)

For "both" path:
  Runs doc retrieval and SQL in parallel, merges into one context window.

CLOUD-DEPENDENCY AUDIT: All LLM calls go to `ollama_base_url` (localhost).
No LangChain, no LlamaIndex — direct Ollama API calls. This ensures no
accidental routing to cloud endpoints via framework defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.db.connector import get_connector
from app.ingestion.embedder import get_embedder
from app.retrieval.qdrant import get_qdrant_store
from app.retrieval.reranker import RankedResult, get_reranker

logger = logging.getLogger(__name__)

RouterDecision = Literal["docs", "sql", "both", "not_found"]


@dataclass
class Citation:
    source_type: Literal["document", "sql"]
    # Document citations
    document_id: str | None = None
    filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_heading: str | None = None
    is_ocr: bool = False
    rerank_score: float | None = None
    text_preview: str = ""
    # SQL citations
    table_name: str | None = None
    sql_query: str | None = None
    row_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class QueryContext:
    """Accumulated state for a single query — passed through pipeline stages."""
    question: str
    session_id: str | None
    router_decision: RouterDecision = "docs"
    ranked_chunks: list[RankedResult] = field(default_factory=list)
    sql_executed: str | None = None
    sql_results: list[dict] | None = None
    sql_error: str | None = None
    citations: list[Citation] = field(default_factory=list)
    top_rerank_score: float = 0.0
    confidence_gate_passed: bool = False
    context_text: str = ""
    latency_ms: float = 0.0
    # For logging
    retrieved_candidates: list[dict] = field(default_factory=list)
    rerank_details: dict = field(default_factory=dict)


_NOT_FOUND_RESPONSE = (
    "I couldn't find relevant information in your documents or connected database "
    "to answer this question. Please check that the relevant documents have been "
    "uploaded and indexed, or that the connected database contains this information."
)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.

Rules:
- Answer ONLY using the information in the <context> block below.
- If the context does not contain enough information to answer, say exactly: "I don't have enough information in the provided context to answer this question."
- Do NOT fabricate facts, statistics, names, dates, or any information not present in the context.
- Do NOT use any knowledge from your training data — only the context provided.
- Be concise and direct. Cite the source document or table in your answer where relevant.
- For SQL results, note that the data reflects what was returned from the query at the time of asking."""

_INTENT_CLASSIFICATION_PROMPT = """You are a query classifier. Classify whether the following question should be answered by:
- "docs": searching document files (reports, manuals, PDFs, text files)
- "sql": querying a SQL database (tabular data, counts, aggregates, records)  
- "both": requires both document search and database query

Question: {question}

Respond with exactly one word: docs, sql, or both. No other text."""

_NL_TO_SQL_PROMPT = """You are a SQL expert. Generate a single SELECT query to answer the question.

Database schema:
{schema}

Few-shot examples:
{examples}

Question: {question}

Rules:
- Generate ONLY a SELECT statement — no INSERT, UPDATE, DELETE, DROP, etc.
- Use only tables and columns that exist in the schema above.
- Return ONLY the SQL query, no explanation, no markdown code blocks.
- If you cannot answer with the available schema, return exactly: CANNOT_ANSWER"""


class RAGPipeline:
    """
    Hand-rolled RAG pipeline. No LangChain, no LlamaIndex.
    Direct calls to Ollama API, Qdrant SDK, and FlagEmbedding.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._http = httpx.AsyncClient(
            base_url=self._settings.ollama_base_url.rstrip("/"),
            timeout=self._settings.ollama_request_timeout,
        )
        # LLM concurrency semaphore — prevents VRAM OOM from overlapping calls
        self._llm_semaphore = asyncio.Semaphore(
            self._settings.max_concurrent_llm_calls
        )

    # ── Intent classification ─────────────────────────────────────────────

    async def classify_intent(self, question: str) -> RouterDecision:
        """
        Use the local LLM to classify query intent.
        Falls back to 'docs' on any error to avoid blocking the user.
        """
        settings = get_settings()
        if settings.router_mode != "auto":
            return settings.router_mode  # type: ignore[return-value]

        connector = get_connector()
        if connector is None:
            return "docs"  # No DB connected, always use docs

        prompt = _INTENT_CLASSIFICATION_PROMPT.format(question=question)
        try:
            resp = await self._http.post(
                "/api/generate",
                json={
                    "model": self._settings.ollama_generation_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
            )
            resp.raise_for_status()
            answer = resp.json()["response"].strip().lower()
            if answer in ("docs", "sql", "both"):
                return answer  # type: ignore[return-value]
        except Exception as exc:
            logger.warning("Intent classification failed (%s), defaulting to docs", exc)
        return "docs"

    # ── Document retrieval ────────────────────────────────────────────────

    async def retrieve_and_rerank(
        self,
        ctx: QueryContext,
        document_ids: list[str] | None = None,
    ) -> None:
        """
        Embed the query, run hybrid search, rerank, set ctx.ranked_chunks.
        Mutates ctx in place.
        """
        embedder = get_embedder()
        qdrant = get_qdrant_store()
        reranker = get_reranker()
        settings = get_settings()

        dense_query = await embedder.embed_query(ctx.question)

        candidates = await qdrant.hybrid_search(
            dense_query=dense_query,
            query_text=ctx.question,
            top_k=settings.reranker_initial_top_k,
            document_ids=document_ids,
        )
        ctx.retrieved_candidates = candidates

        if not candidates:
            ctx.ranked_chunks = []
            ctx.top_rerank_score = 0.0
            return

        ranked = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: reranker.rerank(ctx.question, candidates),
        )
        ctx.ranked_chunks = ranked
        ctx.top_rerank_score = ranked[0].score if ranked else 0.0
        ctx.rerank_details = {
            "reranked": [
                {"chunk_id": r.chunk_id, "score": r.score, "original_rank": r.original_rank}
                for r in ranked
            ],
            "top_score": ctx.top_rerank_score,
        }

    # ── SQL path ──────────────────────────────────────────────────────────

    async def run_sql_path(self, ctx: QueryContext) -> None:
        """
        Generate SQL from schema, validate, execute, handle one retry.
        Mutates ctx in place.
        """
        connector = get_connector()
        if connector is None:
            return

        schema_dict = await connector.introspect_schema()
        schema_text = self._format_schema_for_prompt(schema_dict)

        # Generate SQL
        sql = await self._generate_sql(ctx.question, schema_text)
        if not sql or sql.strip().upper() == "CANNOT_ANSWER":
            ctx.sql_error = "The LLM determined this question cannot be answered from the database schema."
            return

        # Execute with one retry on error
        retry_sql: str | None = None
        try:
            rows, executed_sql = await connector.execute_read_only(sql, retry_sql=None)
            ctx.sql_executed = executed_sql
            ctx.sql_results = rows
        except Exception as first_exc:
            logger.warning("SQL execution failed (%s), requesting retry", first_exc)
            retry_sql = await self._generate_sql(
                ctx.question, schema_text, error_hint=str(first_exc), previous_sql=sql
            )
            if retry_sql and retry_sql.upper() != "CANNOT_ANSWER":
                try:
                    rows, executed_sql = await connector.execute_read_only(retry_sql)
                    ctx.sql_executed = executed_sql
                    ctx.sql_results = rows
                except Exception as retry_exc:
                    ctx.sql_error = f"Query failed after retry: {retry_exc}"
            else:
                ctx.sql_error = str(first_exc)

    async def _generate_sql(
        self,
        question: str,
        schema_text: str,
        error_hint: str | None = None,
        previous_sql: str | None = None,
    ) -> str:
        """Call the local LLM to generate a SQL query."""
        examples = (
            "-- Example: How many customers do we have?\n"
            "SELECT COUNT(*) AS customer_count FROM customers;\n\n"
            "-- Example: What are the top 5 products by revenue?\n"
            "SELECT product_name, SUM(quantity * unit_price) AS revenue "
            "FROM order_items JOIN products USING(product_id) "
            "GROUP BY product_name ORDER BY revenue DESC LIMIT 5;"
        )

        prompt = _NL_TO_SQL_PROMPT.format(
            schema=schema_text, examples=examples, question=question
        )

        if error_hint and previous_sql:
            prompt += (
                f"\n\nThe previous query failed:\n{previous_sql}\n"
                f"Error: {error_hint}\n"
                "Please correct the query and try again."
            )

        resp = await self._http.post(
            "/api/generate",
            json={
                "model": self._settings.ollama_generation_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512},
            },
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    @staticmethod
    def _format_schema_for_prompt(schema: dict) -> str:
        lines = [f"Dialect: {schema.get('dialect', 'unknown')}\n"]
        for table_name, table_info in schema.get("tables", {}).items():
            lines.append(f"Table: {table_name}")
            for col in table_info.get("columns", []):
                pk = " (PK)" if col.get("primary_key") else ""
                nullable = "" if col.get("nullable") else " NOT NULL"
                lines.append(f"  {col['name']} {col['type']}{pk}{nullable}")
            for fk in table_info.get("foreign_keys", []):
                lines.append(f"  FK: {fk['column']} → {fk['references']}")
            lines.append("")
        return "\n".join(lines)

    # ── Context assembly ──────────────────────────────────────────────────

    def _build_context(self, ctx: QueryContext) -> str:
        """
        Assemble the context window from ranked chunks and/or SQL results.
        Also builds the citations list.
        """
        parts: list[str] = []
        ctx.citations = []

        if ctx.ranked_chunks:
            parts.append("## Retrieved Document Passages\n")
            for i, chunk in enumerate(ctx.ranked_chunks, 1):
                page_label = (
                    f"pages {chunk.page_start}–{chunk.page_end}"
                    if chunk.page_end and chunk.page_end != chunk.page_start
                    else f"page {chunk.page_start}" if chunk.page_start else "unknown page"
                )
                ocr_note = " (OCR-extracted, may contain errors)" if chunk.is_ocr else ""
                parts.append(
                    f"[Source {i}: {page_label}{ocr_note}]\n{chunk.text}\n"
                )
                ctx.citations.append(Citation(
                    source_type="document",
                    document_id=chunk.document_id,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_heading=chunk.section_heading,
                    is_ocr=chunk.is_ocr,
                    rerank_score=chunk.score,
                    text_preview=chunk.text[:200],
                ))

        if ctx.sql_results is not None and ctx.sql_executed:
            parts.append("\n## Database Query Results\n")
            parts.append(f"Query executed: `{ctx.sql_executed}`\n")
            if ctx.sql_results:
                # Render first 20 rows as markdown table
                headers = list(ctx.sql_results[0].keys())
                parts.append("| " + " | ".join(headers) + " |")
                parts.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in ctx.sql_results[:20]:
                    parts.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
                if len(ctx.sql_results) > 20:
                    parts.append(f"... and {len(ctx.sql_results) - 20} more rows")
            else:
                parts.append("(Query returned no rows)")

            ctx.citations.append(Citation(
                source_type="sql",
                sql_query=ctx.sql_executed,
                row_count=len(ctx.sql_results),
            ))

        return "\n".join(parts)

    # ── Confidence gate ───────────────────────────────────────────────────

    def _passes_confidence_gate(self, ctx: QueryContext) -> bool:
        """
        Hard gate: if neither the doc retrieval nor SQL path found anything
        above the threshold, skip the LLM call entirely.

        This is implemented in application code, NOT delegated to the LLM's
        self-reported uncertainty — the LLM will often fabricate plausibly
        rather than admit ignorance.
        """
        settings = get_settings()
        threshold = settings.confidence_threshold

        has_good_docs = (
            ctx.ranked_chunks and ctx.top_rerank_score >= threshold
        )
        has_sql_results = (
            ctx.sql_results is not None and len(ctx.sql_results) > 0
        )
        return has_good_docs or has_sql_results

    # ── Generation ────────────────────────────────────────────────────────

    async def stream_answer(
        self, ctx: QueryContext
    ) -> AsyncGenerator[str, None]:
        """
        Stream the LLM's answer token by token.
        Acquires the concurrency semaphore before calling Ollama.
        """
        settings = get_settings()
        full_prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"<context>\n{ctx.context_text}\n</context>\n\n"
            f"Question: {ctx.question}"
        )

        # Wait for semaphore — queue the request rather than firing all at once.
        # The UI should show "waiting for model" during this wait.
        async with self._llm_semaphore:
            yield "__STATUS__:generating"
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(2),
                    wait=wait_exponential(min=1, max=4),
                    retry=retry_if_exception_type(httpx.ConnectError),
                ):
                    with attempt:
                        async with self._http.stream(
                            "POST",
                            "/api/generate",
                            json={
                                "model": settings.ollama_generation_model,
                                "prompt": full_prompt,
                                "stream": True,
                                "options": {
                                    "temperature": settings.llm_temperature,
                                    "num_predict": settings.llm_max_tokens,
                                },
                            },
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    token = data.get("response", "")
                                    if token:
                                        yield token
                                    if data.get("done"):
                                        return
                                except json.JSONDecodeError:
                                    continue
            except httpx.ConnectError:
                yield "\n\n[ERROR: Local LLM unavailable — check that Ollama is running]"
            except httpx.TimeoutException:
                yield "\n\n[ERROR: LLM request timed out — model may still be loading]"
            except Exception as exc:
                logger.error("LLM generation error: %s", exc)
                yield f"\n\n[ERROR: Generation failed — {type(exc).__name__}]"

    # ── Main entry point ──────────────────────────────────────────────────

    async def run(
        self,
        question: str,
        session_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Full RAG pipeline. Yields SSE-compatible dicts:
          {"type": "status", "content": "..."} — progress updates
          {"type": "token", "content": "..."} — streamed LLM tokens
          {"type": "citations", "content": [...]} — final citations
          {"type": "sql", "content": "..."} — SQL query used (always shown)
          {"type": "done", "content": {...}} — final metadata
          {"type": "error", "content": "..."} — error message
        """
        start = time.monotonic()
        ctx = QueryContext(question=question, session_id=session_id)

        try:
            # ── 1. Intent classification ──────────────────────────────────
            yield {"type": "status", "content": "Classifying query intent..."}
            ctx.router_decision = await self.classify_intent(question)
            yield {"type": "status", "content": f"Router: {ctx.router_decision}"}

            # ── 2. Retrieval (parallel where applicable) ──────────────────
            tasks = []
            if ctx.router_decision in ("docs", "both"):
                tasks.append(self.retrieve_and_rerank(ctx, document_ids=document_ids))
            if ctx.router_decision in ("sql", "both"):
                tasks.append(self.run_sql_path(ctx))

            if tasks:
                yield {"type": "status", "content": "Retrieving relevant context..."}
                await asyncio.gather(*tasks)

            # ── 3. Confidence gate ────────────────────────────────────────
            ctx.confidence_gate_passed = self._passes_confidence_gate(ctx)
            if not ctx.confidence_gate_passed:
                yield {"type": "status", "content": "Confidence gate: no relevant context found"}
                yield {"type": "token", "content": _NOT_FOUND_RESPONSE}
                ctx.router_decision = "not_found"
                ctx.latency_ms = (time.monotonic() - start) * 1000
                yield {
                    "type": "done",
                    "content": {
                        "router_decision": ctx.router_decision,
                        "citations": [],
                        "top_rerank_score": ctx.top_rerank_score,
                        "latency_ms": ctx.latency_ms,
                    },
                }
                return

            # ── 4. Build context window ───────────────────────────────────
            yield {"type": "status", "content": "Building context window..."}
            ctx.context_text = self._build_context(ctx)

            # ── 5. Emit SQL query (always shown to user) ──────────────────
            if ctx.sql_executed:
                yield {"type": "sql", "content": ctx.sql_executed}

            # ── 6. Stream generation ──────────────────────────────────────
            yield {"type": "status", "content": "Generating answer..."}
            async for token in self.stream_answer(ctx):
                if token.startswith("__STATUS__:"):
                    yield {"type": "status", "content": token[len("__STATUS__:"):]}
                else:
                    yield {"type": "token", "content": token}

            # ── 7. Emit citations ─────────────────────────────────────────
            ctx.latency_ms = (time.monotonic() - start) * 1000
            yield {
                "type": "citations",
                "content": [c.to_dict() for c in ctx.citations],
            }
            yield {
                "type": "done",
                "content": {
                    "router_decision": ctx.router_decision,
                    "top_rerank_score": ctx.top_rerank_score,
                    "latency_ms": ctx.latency_ms,
                    "confidence_gate_passed": ctx.confidence_gate_passed,
                },
            }

        except Exception as exc:
            logger.exception("Pipeline error for question: %s", question)
            ctx.latency_ms = (time.monotonic() - start) * 1000
            yield {"type": "error", "content": f"Pipeline error: {type(exc).__name__}: {exc}"}

        finally:
            # Always log the query for RAGAS eval and debugging
            await self._log_query(ctx)

    async def _log_query(self, ctx: QueryContext) -> None:
        """Persist query details to the app database for eval/debugging."""
        import os, json
        from datetime import datetime, timezone
        settings = get_settings()
        log_dir = settings.query_log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": ctx.session_id,
            "question": ctx.question,
            "router_decision": ctx.router_decision,
            "top_rerank_score": ctx.top_rerank_score,
            "confidence_gate_passed": ctx.confidence_gate_passed,
            "retrieved_count": len(ctx.retrieved_candidates),
            "rerank_details": ctx.rerank_details,
            "sql_executed": ctx.sql_executed,
            "sql_results_preview": str(ctx.sql_results[:3]) if ctx.sql_results else None,
            "latency_ms": ctx.latency_ms,
        }
        fname = f"{log_dir}/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
        with open(fname, "w") as f:
            json.dump(log_entry, f, indent=2)

    async def health_check(self) -> dict[str, Any]:
        """Check Ollama and Qdrant reachability."""
        results: dict[str, Any] = {}
        # Check Ollama
        try:
            resp = await self._http.get("/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            gen_model = self._settings.ollama_generation_model
            embed_model = self._settings.ollama_embed_model
            results["ollama"] = {
                "status": "ok",
                "models_available": models,
                "generation_model_ready": any(
                    m.startswith(gen_model.split(":")[0]) for m in models
                ),
                "embed_model_ready": any(
                    m.startswith(embed_model.split(":")[0]) for m in models
                ),
            }
            # Check GPU usage via Ollama's ps endpoint
            try:
                ps_resp = await self._http.get("/api/ps", timeout=5.0)
                if ps_resp.status_code == 200:
                    ps_data = ps_resp.json()
                    results["ollama"]["gpu_active"] = bool(
                        ps_data.get("models", [])
                    )
            except Exception:
                results["ollama"]["gpu_active"] = None

        except Exception as exc:
            results["ollama"] = {
                "status": "error",
                "error": str(exc),
                "warning": (
                    "Ollama is unreachable. Start Ollama and ensure models are pulled. "
                    "Chat will not work until Ollama is running."
                ),
            }

        # Check Qdrant
        qdrant = get_qdrant_store()
        qdrant_ok = await qdrant.health_check()
        results["qdrant"] = {
            "status": "ok" if qdrant_ok else "error",
            "warning": (
                None if qdrant_ok else
                "Qdrant is unreachable. Document search will not work."
            ),
        }

        # Check reranker GPU
        reranker = get_reranker()
        results["reranker"] = reranker.gpu_status()

        return results


# ── Module-level singleton ────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

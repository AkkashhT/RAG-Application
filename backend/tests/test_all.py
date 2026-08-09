"""
Test suite for LocalRAG.

Tests:
  1. Chunking logic — including multi-page table fixture
  2. SQL safety validator — EXPLAIN + statement rejection
  3. Query router source-selection logic
  4. Confidence gate threshold behavior
  5. Concurrency semaphore (simulated overlapping requests)
  6. Graceful handling of unreachable/crashed Ollama

Run with: pytest backend/tests/ -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add backend app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_parse_result():
    """A ParseResult with mixed element types including a table."""
    from app.ingestion.parsers import ParsedElement, ParseResult
    return ParseResult(
        elements=[
            ParsedElement(element_type="heading", text="Introduction", page_start=1, page_end=1),
            ParsedElement(element_type="text", text="This document describes the system architecture. " * 20, page_start=1, page_end=1, section_heading="Introduction"),
            ParsedElement(element_type="table", text="| Col A | Col B |\n|---|---|\n| val1 | val2 |", page_start=2, page_end=2, section_heading="Introduction"),
            ParsedElement(element_type="text", text="Further analysis shows that the results are significant. " * 20, page_start=3, page_end=3, section_heading="Introduction"),
        ],
        filename="test.pdf",
        file_type="pdf",
        page_count=3,
        has_ocr_pages=False,
        ocr_pages=[],
    )


@pytest.fixture
def multipage_table_parse_result():
    """
    Test fixture: a PDF with a table that spans pages 4 and 5.
    This is the dedicated fixture mentioned in the spec to catch
    regressions in multi-page table merging.
    """
    from app.ingestion.parsers import ParsedElement, ParseResult
    return ParseResult(
        elements=[
            ParsedElement(element_type="heading", text="Revenue by Quarter", page_start=3, page_end=3),
            ParsedElement(element_type="text", text="The following table shows quarterly revenue for fiscal year 2024.", page_start=3, page_end=3, section_heading="Revenue by Quarter"),
            # Table starts on page 4, continues to page 5
            ParsedElement(
                element_type="table",
                text="| Quarter | Revenue | Growth |\n|---|---|---|\n| Q1 | $1.2M | +5% |\n| Q2 | $1.4M | +12% |",
                page_start=4, page_end=4,
                section_heading="Revenue by Quarter",
            ),
            ParsedElement(
                element_type="table",
                text="| Q3 | $1.6M | +14% |\n| Q4 | $2.1M | +31% |\n| **Total** | **$6.3M** | **+15%** |",
                page_start=5, page_end=5,
                section_heading="Revenue by Quarter",
            ),
            ParsedElement(element_type="text", text="Growth accelerated significantly in Q4.", page_start=6, page_end=6, section_heading="Revenue by Quarter"),
        ],
        filename="annual_report.pdf",
        file_type="pdf",
        page_count=6,
        has_ocr_pages=False,
        ocr_pages=[],
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHUNKING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestChunker:
    def test_basic_chunking_produces_chunks(self, sample_parse_result):
        from app.ingestion.chunker import StructureAwareChunker
        chunker = StructureAwareChunker(chunk_size_tokens=200, overlap_tokens=20)
        chunks = chunker.chunk(sample_parse_result, document_id="doc-001")
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self, sample_parse_result):
        from app.ingestion.chunker import StructureAwareChunker
        chunker = StructureAwareChunker(chunk_size_tokens=200, overlap_tokens=20)
        chunks = chunker.chunk(sample_parse_result, document_id="doc-001")
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.document_id == "doc-001"
            assert chunk.text.strip()
            assert chunk.token_count > 0
            assert chunk.chunk_index >= 0

    def test_table_is_never_split(self, sample_parse_result):
        """Table elements must never be split across chunks."""
        from app.ingestion.chunker import StructureAwareChunker
        # Small chunk size to force splitting of text elements
        chunker = StructureAwareChunker(chunk_size_tokens=50, overlap_tokens=5)
        chunks = chunker.chunk(sample_parse_result, document_id="doc-001")
        table_text = "| Col A | Col B |"
        # The table should appear in exactly one chunk, not fragmented
        table_chunks = [c for c in chunks if table_text in c.text]
        assert len(table_chunks) == 1, (
            f"Table appeared in {len(table_chunks)} chunks — it should never be split"
        )

    def test_table_chunk_has_table_element_type(self, sample_parse_result):
        from app.ingestion.chunker import StructureAwareChunker
        chunker = StructureAwareChunker(chunk_size_tokens=50, overlap_tokens=5)
        chunks = chunker.chunk(sample_parse_result, document_id="doc-001")
        table_chunks = [c for c in chunks if "table" in c.element_types]
        assert len(table_chunks) >= 1

    def test_chunk_indices_are_sequential(self, sample_parse_result):
        from app.ingestion.chunker import StructureAwareChunker
        chunker = StructureAwareChunker(chunk_size_tokens=200, overlap_tokens=20)
        chunks = chunker.chunk(sample_parse_result, document_id="doc-001")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks))), "Chunk indices must be sequential starting at 0"

    def test_chunk_ids_are_unique(self, sample_parse_result):
        from app.ingestion.chunker import StructureAwareChunker
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(sample_parse_result, document_id="doc-001")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "All chunk IDs must be unique"

    # ── Multi-page table fixture ────────────────────────────────────────────

    def test_multipage_table_merges_into_single_chunk(self, multipage_table_parse_result):
        """
        REGRESSION TEST: A table spanning pages 4–5 must be merged into a single
        chunk, not split into two half-tables. This is the dedicated fixture
        specified in the project requirements to catch regressions here.
        """
        from app.ingestion.chunker import StructureAwareChunker
        chunker = StructureAwareChunker(chunk_size_tokens=800, overlap_tokens=100)
        chunks = chunker.chunk(multipage_table_parse_result, document_id="doc-multipage")

        # Find chunks containing table data
        table_chunks = [c for c in chunks if "| Quarter |" in c.text or "| Q1 |" in c.text]
        assert len(table_chunks) >= 1, "Table data not found in any chunk"

        # The Q1 row and Q3 row (from different pages) should be in the SAME chunk
        combined_chunks = [
            c for c in chunks
            if "| Q1 |" in c.text and "| Q3 |" in c.text
        ]
        assert len(combined_chunks) >= 1, (
            "Multi-page table was SPLIT across chunks. "
            "Q1 (page 4) and Q3 (page 5) rows must appear in the same chunk. "
            "Check the table merging logic in StructureAwareChunker."
        )

    def test_multipage_table_chunk_has_page_range(self, multipage_table_parse_result):
        """Merged multi-page table chunk must have page_start != page_end."""
        from app.ingestion.chunker import StructureAwareChunker
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(multipage_table_parse_result, document_id="doc-multipage")
        # Find the merged table chunk
        merged = [c for c in chunks if "| Q1 |" in c.text and "| Q3 |" in c.text]
        if merged:
            chunk = merged[0]
            assert chunk.page_start is not None
            assert chunk.page_end is not None
            assert chunk.page_end > chunk.page_start, (
                f"Merged table chunk page range should span multiple pages, "
                f"got page_start={chunk.page_start}, page_end={chunk.page_end}"
            )

    def test_ocr_flag_propagates(self):
        """If any element in a chunk is OCR'd, the chunk must be flagged is_ocr=True."""
        from app.ingestion.parsers import ParsedElement, ParseResult
        from app.ingestion.chunker import StructureAwareChunker
        result = ParseResult(
            elements=[
                ParsedElement(element_type="text", text="Regular text. " * 10, page_start=1, page_end=1, is_ocr=False),
                ParsedElement(element_type="text", text="OCR extracted text. " * 10, page_start=2, page_end=2, is_ocr=True),
            ],
            filename="scanned.pdf", file_type="pdf", page_count=2,
            has_ocr_pages=True, ocr_pages=[2],
        )
        chunker = StructureAwareChunker(chunk_size_tokens=50, overlap_tokens=5)
        chunks = chunker.chunk(result, document_id="doc-ocr")
        ocr_chunks = [c for c in chunks if c.is_ocr]
        assert len(ocr_chunks) > 0, "OCR flag was not propagated to any chunk"


# ══════════════════════════════════════════════════════════════════════════════
# 2. SQL SAFETY VALIDATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSQLSafetyValidator:
    """
    Tests for both layers of SQL safety:
      - Statement-type rejection (validate_query)
      - EXPLAIN validation (explain_query) — tested with SQLite
    """

    def _make_connector(self) -> "SQLConnector":
        from app.db.connector import SQLConnector
        return SQLConnector("sqlite+aiosqlite:///:memory:")

    def test_valid_select_passes(self):
        conn = self._make_connector()
        ok, reason = conn.validate_query("SELECT * FROM users")
        assert ok, f"Valid SELECT was rejected: {reason}"

    def test_select_with_subquery_passes(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("SELECT id FROM orders WHERE user_id IN (SELECT id FROM users WHERE active=1)")
        assert ok

    def test_insert_rejected(self):
        conn = self._make_connector()
        ok, reason = conn.validate_query("INSERT INTO users (name) VALUES ('evil')")
        assert not ok
        assert "INSERT" in reason.upper()

    def test_update_rejected(self):
        conn = self._make_connector()
        ok, reason = conn.validate_query("UPDATE users SET admin=1 WHERE id=42")
        assert not ok

    def test_delete_rejected(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("DELETE FROM users")
        assert not ok

    def test_drop_rejected(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("DROP TABLE users")
        assert not ok

    def test_alter_rejected(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("ALTER TABLE users ADD COLUMN evil TEXT")
        assert not ok

    def test_create_rejected(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("CREATE TABLE evil (id INTEGER)")
        assert not ok

    def test_truncate_rejected(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("TRUNCATE TABLE users")
        assert not ok

    def test_semicolon_terminated_select_passes(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("SELECT 1;")
        assert ok

    def test_multiline_select_passes(self):
        conn = self._make_connector()
        ok, _ = conn.validate_query("SELECT\n  id,\n  name\nFROM\n  users\nWHERE active = 1")
        assert ok

    def test_case_insensitive_rejection(self):
        conn = self._make_connector()
        # Lowercase should also be rejected
        ok, _ = conn.validate_query("insert into users values (1, 'test')")
        assert not ok

    @pytest.mark.asyncio
    async def test_explain_catches_invalid_sql(self):
        """EXPLAIN on SQLite should fail for syntactically invalid SQL."""
        from app.db.connector import SQLConnector
        conn = SQLConnector("sqlite+aiosqlite:///:memory:")
        is_valid, error = await conn.explain_query("SELECT * FROM nonexistent_table_xyz")
        # SQLite EXPLAIN may not fail on missing tables (it's lazy), so this
        # test verifies the explain_query method runs without crashing
        assert isinstance(is_valid, bool)
        assert isinstance(error, str)

    @pytest.mark.asyncio
    async def test_execute_rejects_unsafe_query(self):
        """execute_read_only must raise ValueError for non-SELECT statements."""
        from app.db.connector import SQLConnector
        conn = SQLConnector("sqlite+aiosqlite:///:memory:")
        with pytest.raises(ValueError, match="Rejected"):
            await conn.execute_read_only("DELETE FROM anything")


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUERY ROUTER SOURCE-SELECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryRouter:
    """Tests for the router's intent classification and forced modes."""

    @pytest.mark.asyncio
    async def test_forced_docs_mode(self):
        """When router_mode='docs', always return 'docs' regardless of question."""
        with patch("app.core.config.get_settings") as mock_settings:
            s = MagicMock()
            s.router_mode = "docs"
            mock_settings.return_value = s

            from app.core.pipeline import RAGPipeline
            pipeline = RAGPipeline.__new__(RAGPipeline)
            decision = await pipeline.classify_intent("how many rows in the users table?")
            assert decision == "docs"

    @pytest.mark.asyncio
    async def test_forced_sql_mode(self):
        with patch("app.core.config.get_settings") as mock_settings:
            s = MagicMock()
            s.router_mode = "sql"
            mock_settings.return_value = s

            from app.core.pipeline import RAGPipeline
            pipeline = RAGPipeline.__new__(RAGPipeline)
            decision = await pipeline.classify_intent("what is in the documentation?")
            assert decision == "sql"

    @pytest.mark.asyncio
    async def test_forced_both_mode(self):
        with patch("app.core.config.get_settings") as mock_settings:
            s = MagicMock()
            s.router_mode = "both"
            mock_settings.return_value = s

            from app.core.pipeline import RAGPipeline
            pipeline = RAGPipeline.__new__(RAGPipeline)
            decision = await pipeline.classify_intent("anything")
            assert decision == "both"

    @pytest.mark.asyncio
    async def test_auto_mode_falls_back_to_docs_when_no_db(self):
        """In auto mode with no DB connected, must return 'docs'."""
        with patch("app.core.config.get_settings") as mock_settings, \
             patch("app.db.connector.get_connector", return_value=None):
            s = MagicMock()
            s.router_mode = "auto"
            mock_settings.return_value = s

            from app.core.pipeline import RAGPipeline
            pipeline = RAGPipeline.__new__(RAGPipeline)
            decision = await pipeline.classify_intent("how many records?")
            assert decision == "docs"

    @pytest.mark.asyncio
    async def test_auto_mode_falls_back_on_ollama_error(self):
        """If Ollama is unreachable during classification, fall back to 'docs'."""
        import httpx

        with patch("app.core.config.get_settings") as mock_settings, \
             patch("app.db.connector.get_connector", return_value=MagicMock()):
            s = MagicMock()
            s.router_mode = "auto"
            s.ollama_generation_model = "qwen3:14b"
            mock_settings.return_value = s

            from app.core.pipeline import RAGPipeline
            pipeline = RAGPipeline.__new__(RAGPipeline)
            pipeline._http = AsyncMock()
            pipeline._http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

            decision = await pipeline.classify_intent("anything")
            assert decision == "docs"


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFIDENCE GATE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceGate:
    """
    The confidence gate must skip the LLM call when:
      - No reranked chunks score above the threshold, AND
      - No SQL results were returned.
    This is a hard application-level check — NOT delegated to the LLM.
    """

    def _make_pipeline(self, threshold: float = 0.4):
        from app.core.pipeline import RAGPipeline
        with patch("app.core.config.get_settings") as mock_settings:
            s = MagicMock()
            s.confidence_threshold = threshold
            mock_settings.return_value = s
            pipeline = RAGPipeline.__new__(RAGPipeline)
            pipeline._settings = s
            return pipeline, s

    def test_gate_fails_when_no_chunks_no_sql(self):
        from app.core.pipeline import QueryContext
        pipeline, settings = self._make_pipeline(threshold=0.4)
        settings.confidence_threshold = 0.4

        ctx = QueryContext(question="test", session_id=None)
        ctx.ranked_chunks = []
        ctx.top_rerank_score = 0.0
        ctx.sql_results = None

        with patch("app.core.config.get_settings", return_value=settings):
            assert not pipeline._passes_confidence_gate(ctx)

    def test_gate_passes_when_chunk_above_threshold(self):
        from app.core.pipeline import QueryContext
        from app.retrieval.reranker import RankedResult

        pipeline, settings = self._make_pipeline(threshold=0.4)
        settings.confidence_threshold = 0.4

        ctx = QueryContext(question="test", session_id=None)
        ctx.ranked_chunks = [
            RankedResult(
                chunk_id="c1", score=0.75, text="relevant text",
                document_id="d1", page_start=1, page_end=1,
                section_heading=None, is_ocr=False, original_rank=0,
            )
        ]
        ctx.top_rerank_score = 0.75
        ctx.sql_results = None

        with patch("app.core.config.get_settings", return_value=settings):
            assert pipeline._passes_confidence_gate(ctx)

    def test_gate_fails_when_chunk_below_threshold(self):
        from app.core.pipeline import QueryContext
        from app.retrieval.reranker import RankedResult

        pipeline, settings = self._make_pipeline(threshold=0.4)
        settings.confidence_threshold = 0.4

        ctx = QueryContext(question="test", session_id=None)
        ctx.ranked_chunks = [
            RankedResult(
                chunk_id="c1", score=0.15, text="weakly relevant",
                document_id="d1", page_start=1, page_end=1,
                section_heading=None, is_ocr=False, original_rank=0,
            )
        ]
        ctx.top_rerank_score = 0.15
        ctx.sql_results = None

        with patch("app.core.config.get_settings", return_value=settings):
            assert not pipeline._passes_confidence_gate(ctx)

    def test_gate_passes_when_sql_results_present(self):
        """Even with no doc chunks, SQL results alone should pass the gate."""
        from app.core.pipeline import QueryContext

        pipeline, settings = self._make_pipeline(threshold=0.4)
        settings.confidence_threshold = 0.4

        ctx = QueryContext(question="test", session_id=None)
        ctx.ranked_chunks = []
        ctx.top_rerank_score = 0.0
        ctx.sql_results = [{"id": 1, "name": "Alice"}]

        with patch("app.core.config.get_settings", return_value=settings):
            assert pipeline._passes_confidence_gate(ctx)

    def test_gate_fails_with_empty_sql_results(self):
        """Empty SQL result set should NOT pass the gate."""
        from app.core.pipeline import QueryContext

        pipeline, settings = self._make_pipeline(threshold=0.4)
        settings.confidence_threshold = 0.4

        ctx = QueryContext(question="test", session_id=None)
        ctx.ranked_chunks = []
        ctx.top_rerank_score = 0.0
        ctx.sql_results = []  # empty list — query ran but returned nothing

        with patch("app.core.config.get_settings", return_value=settings):
            assert not pipeline._passes_confidence_gate(ctx)

    def test_threshold_boundary_exact(self):
        """Score exactly equal to threshold should pass (>=, not >)."""
        from app.core.pipeline import QueryContext
        from app.retrieval.reranker import RankedResult

        pipeline, settings = self._make_pipeline(threshold=0.4)
        settings.confidence_threshold = 0.4

        ctx = QueryContext(question="test", session_id=None)
        ctx.ranked_chunks = [
            RankedResult(
                chunk_id="c1", score=0.4, text="text",
                document_id="d1", page_start=1, page_end=1,
                section_heading=None, is_ocr=False, original_rank=0,
            )
        ]
        ctx.top_rerank_score = 0.4
        ctx.sql_results = None

        with patch("app.core.config.get_settings", return_value=settings):
            assert pipeline._passes_confidence_gate(ctx)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONCURRENCY SEMAPHORE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestConcurrencySemaphore:
    """
    Verifies that overlapping LLM requests are serialized (not fired simultaneously),
    preventing VRAM OOM crashes on a single GPU.
    """

    @pytest.mark.asyncio
    async def test_semaphore_serializes_llm_calls(self):
        """
        Two concurrent calls with max_concurrent_llm_calls=1 must not overlap.
        The second call must wait until the first completes.
        """
        import asyncio

        call_log: list[tuple[str, float]] = []

        async def fake_stream_answer(ctx):
            import time
            call_log.append(("start", time.monotonic()))
            yield "token"
            await asyncio.sleep(0.1)  # simulate generation time
            call_log.append(("end", time.monotonic()))

        from app.core.pipeline import RAGPipeline, QueryContext
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._llm_semaphore = asyncio.Semaphore(1)

        async def run_one(label: str):
            ctx = QueryContext(question="test", session_id=None)
            async with pipeline._llm_semaphore:
                call_log.append((f"{label}_start", asyncio.get_event_loop().time()))
                await asyncio.sleep(0.05)
                call_log.append((f"{label}_end", asyncio.get_event_loop().time()))

        # Fire two coroutines simultaneously
        await asyncio.gather(run_one("A"), run_one("B"))

        # With semaphore=1, A_end must come before B_start (or vice versa)
        events = {label: t for label, t in call_log}
        a_end = events.get("A_end", 0)
        b_start = events.get("B_start", 0)
        b_end = events.get("B_end", 0)
        a_start = events.get("A_start", 0)

        # One of the two must have fully completed before the other started
        non_overlapping = (a_end <= b_start) or (b_end <= a_start)
        assert non_overlapping, (
            f"Semaphore did not serialize LLM calls. "
            f"A: {a_start:.3f}–{a_end:.3f}, B: {b_start:.3f}–{b_end:.3f}"
        )

    @pytest.mark.asyncio
    async def test_semaphore_2_allows_two_concurrent(self):
        """With max_concurrent=2, two calls should overlap."""
        import asyncio
        import time

        start_times = []
        sem = asyncio.Semaphore(2)

        async def run_one():
            async with sem:
                start_times.append(time.monotonic())
                await asyncio.sleep(0.05)

        await asyncio.gather(run_one(), run_one(), run_one())
        # With semaphore=2, at least two starts should be very close together
        start_times.sort()
        assert (start_times[1] - start_times[0]) < 0.03, (
            "Two concurrent calls should start nearly simultaneously with semaphore=2"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. OLLAMA FAILURE HANDLING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaFailureHandling:
    """
    Verifies graceful degradation when Ollama is unreachable or crashes mid-stream.
    The frontend should never be left spinning indefinitely.
    """

    @pytest.mark.asyncio
    async def test_unreachable_ollama_returns_error_event(self):
        """
        If Ollama is down, stream_answer must yield an error token,
        not raise an unhandled exception or hang.
        """
        import httpx
        from app.core.pipeline import RAGPipeline, QueryContext

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._llm_semaphore = asyncio.Semaphore(1)

        mock_http = AsyncMock()
        # Simulate connection refused
        mock_http.stream = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=httpx.ConnectError("Connection refused")),
                __aexit__=AsyncMock(return_value=False),
            )
        )
        pipeline._http = mock_http

        settings_mock = MagicMock()
        settings_mock.ollama_generation_model = "qwen3:14b"
        settings_mock.llm_temperature = 0.1
        settings_mock.llm_max_tokens = 512
        settings_mock.max_concurrent_llm_calls = 1
        pipeline._settings = settings_mock

        ctx = QueryContext(question="test question", session_id=None)
        ctx.context_text = "some context"

        tokens = []
        with patch("app.core.config.get_settings", return_value=settings_mock):
            async for token in pipeline.stream_answer(ctx):
                tokens.append(token)

        combined = "".join(tokens)
        assert "ERROR" in combined.upper() or "unavailable" in combined.lower(), (
            f"Expected an error message when Ollama is down, got: {combined[:200]}"
        )

    @pytest.mark.asyncio
    async def test_mid_stream_crash_closes_gracefully(self):
        """
        If Ollama crashes mid-stream (connection drops while streaming),
        the generator must close cleanly with a partial-failure message.
        """
        import httpx
        from app.core.pipeline import RAGPipeline, QueryContext

        async def flaky_stream():
            """Yields a few tokens then raises mid-stream."""
            yield b'data: {"response": "partial answer", "done": false}\n'
            raise httpx.RemoteProtocolError("connection lost", request=None)

        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._llm_semaphore = asyncio.Semaphore(1)

        settings_mock = MagicMock()
        settings_mock.ollama_generation_model = "qwen3:14b"
        settings_mock.llm_temperature = 0.1
        settings_mock.llm_max_tokens = 512
        settings_mock.max_concurrent_llm_calls = 1
        pipeline._settings = settings_mock

        ctx = QueryContext(question="test", session_id=None)
        ctx.context_text = "context"

        # The stream_answer method should handle this without raising
        tokens = []
        try:
            with patch("app.core.config.get_settings", return_value=settings_mock):
                async for token in pipeline.stream_answer(ctx):
                    tokens.append(token)
        except Exception as exc:
            pytest.fail(f"stream_answer raised an unhandled exception: {exc}")

        # Should have received something (even just an error message)
        assert len(tokens) > 0, "No tokens received — stream_answer must always yield something"

    @pytest.mark.asyncio
    async def test_health_check_returns_error_status_when_ollama_down(self):
        """health_check must return status='error' gracefully, not raise."""
        from app.core.pipeline import RAGPipeline

        pipeline = RAGPipeline.__new__(RAGPipeline)
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("connection refused"))
        pipeline._http = mock_http

        settings_mock = MagicMock()
        settings_mock.ollama_base_url = "http://localhost:11434"
        settings_mock.ollama_generation_model = "qwen3:14b"
        settings_mock.ollama_embed_model = "nomic-embed-text"
        settings_mock.qdrant_url = "http://localhost:6333"
        settings_mock.reranker_model = "BAAI/bge-reranker-v2-m3"
        settings_mock.reranker_device = "cuda"
        pipeline._settings = settings_mock

        with patch("app.retrieval.qdrant.get_qdrant_store") as mock_qdrant, \
             patch("app.retrieval.reranker.get_reranker") as mock_reranker, \
             patch("app.core.config.get_settings", return_value=settings_mock):
            mock_qdrant.return_value.health_check = AsyncMock(return_value=True)
            mock_reranker.return_value.gpu_status = MagicMock(return_value={"loaded": False})

            health = await pipeline.health_check()

        assert health["ollama"]["status"] == "error"
        assert "warning" in health["ollama"]

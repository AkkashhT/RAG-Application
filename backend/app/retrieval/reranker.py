"""
BGE cross-encoder reranker.

Uses BAAI/bge-reranker-v2-m3 via the FlagEmbedding library, running entirely
locally on the same GPU as Ollama (or CPU if GPU is unavailable).

Why a cross-encoder reranker?
  Bi-encoders (like nomic-embed-text) encode query and document independently
  — they can't model fine-grained query-document interactions. This means the
  initial vector search ranks by rough semantic proximity, not true relevance.
  A cross-encoder sees (query, document) together, which dramatically improves
  precision — especially for long or complex questions where the answer is a
  specific passage, not a vague topic match.

  The pattern here is: vector search top-20 (recall-focused) → rerank top-5
  (precision-focused) → LLM context. This is the single highest-leverage
  improvement over naive vector search.

CLOUD-DEPENDENCY AUDIT: FlagEmbedding loads BAAI/bge-reranker-v2-m3 from
HuggingFace Hub on first startup (weights are cached locally after that).
The container image pre-downloads the weights during build. At inference time,
no network calls are made — the model runs fully locally on GPU/CPU.

If you see network calls during reranking after the initial download, something
has changed upstream — pin the FlagEmbedding version in requirements.txt and
inspect their source.

GPU note: FlagEmbedding auto-detects CUDA if torch is built with CUDA support
and `use_fp16=True` is set (half-precision on GPU). The reranker process must
have GPU access — in Docker Compose this means the same GPU passthrough
configuration as the Ollama service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A reranked retrieval result with its cross-encoder score."""
    chunk_id: str
    score: float           # cross-encoder score (0–1 range after sigmoid)
    text: str
    document_id: str | None
    page_start: int | None
    page_end: int | None
    section_heading: str | None
    is_ocr: bool
    original_rank: int     # position in the pre-rerank list (for debugging)


class BGEReranker:
    """
    Wraps BAAI/bge-reranker-v2-m3 for cross-encoder reranking.

    The model is loaded lazily on first use to avoid slowing down startup
    (the checkpoint is ~278M params; loading takes a few seconds on GPU).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.reranker_model
        self._device = settings.reranker_device
        self._top_k = settings.reranker_top_k
        self._model = None  # lazy init

    def _load_model(self) -> None:
        """Load the cross-encoder. Called once, cached in self._model."""
        if self._model is not None:
            return

        logger.info(
            "Loading reranker %s on device=%s", self._model_name, self._device
        )
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self._model_name,
                use_fp16=(self._device == "cuda"),
                device=self._device,
            )
            logger.info("Reranker loaded successfully")
        except Exception as exc:
            logger.error("Failed to load reranker: %s", exc)
            raise

    def _verify_gpu_usage(self) -> bool:
        """
        Detect whether the reranker is actually on GPU.
        Returns True if confirmed GPU, False if CPU or unknown.
        """
        if self._model is None:
            return False
        try:
            import torch
            # FlagReranker exposes the underlying model
            underlying = getattr(self._model, "model", None)
            if underlying is None:
                return False
            device = next(underlying.parameters()).device
            return device.type == "cuda"
        except Exception:
            return False

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """
        Rerank a list of candidate chunks against the query.

        Args:
            query: The user's question.
            candidates: List of dicts from QdrantStore.hybrid_search().
            top_k: How many to keep. Defaults to config value.

        Returns:
            Top-k RankedResult objects, sorted by score descending.
        """
        self._load_model()

        if not candidates:
            return []

        k = top_k or self._top_k
        texts = [c["text"] for c in candidates]

        # Cross-encoder: score each (query, passage) pair
        pairs = [[query, t] for t in texts]
        try:
            scores = self._model.compute_score(pairs, normalize=True)
        except Exception as exc:
            logger.error("Reranker scoring failed: %s", exc)
            # Graceful degradation: fall back to original order with score=0
            scores = [0.0] * len(candidates)

        # Ensure scores is a flat list (some versions return numpy arrays)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, float):
            scores = [scores]

        ranked = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            ranked.append(RankedResult(
                chunk_id=candidate["chunk_id"],
                score=float(score),
                text=candidate["text"],
                document_id=candidate.get("document_id"),
                page_start=candidate.get("page_start"),
                page_end=candidate.get("page_end"),
                section_heading=candidate.get("section_heading"),
                is_ocr=candidate.get("is_ocr", False),
                original_rank=i,
            ))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:k]

    def gpu_status(self) -> dict[str, Any]:
        """Return GPU status info for the startup health check."""
        if self._model is None:
            return {"loaded": False, "on_gpu": False, "device": self._device}
        on_gpu = self._verify_gpu_usage()
        return {
            "loaded": True,
            "on_gpu": on_gpu,
            "device": self._device,
            "model": self._model_name,
            "warning": (
                None if on_gpu else
                "Reranker is running on CPU — reranking latency will be higher. "
                "Set RERANKER_DEVICE=cuda and ensure the container has GPU access."
            ),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_reranker: BGEReranker | None = None


def get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker

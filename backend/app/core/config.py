"""
Central configuration. All tunables live here and are overridable via environment
variables or the Settings UI (which writes to a local .env file).

CLOUD-DEPENDENCY AUDIT: This file has a startup check that raises immediately
if any known cloud LLM credentials are present in the environment. See
`assert_no_cloud_keys()` — call it from main.py before starting the server.
"""

import os
import sys
from functools import lru_cache
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


KNOWN_CLOUD_KEY_ENVVARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "COHERE_API_KEY",
    "MISTRAL_API_KEY",
    "GOOGLE_API_KEY",
    "HUGGINGFACEHUB_API_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "TOGETHER_API_KEY",
    "REPLICATE_API_TOKEN",
    "AWS_ACCESS_KEY_ID",       # could indicate Bedrock usage
    "GROQ_API_KEY",
    "FIREWORKS_API_KEY",
]


def assert_no_cloud_keys() -> None:
    """
    Hard fail if any cloud LLM credentials are found in the environment.
    This is the primary guard against accidentally shipping data to a cloud API.
    Call this once at startup before anything else initialises.
    """
    found = [k for k in KNOWN_CLOUD_KEY_ENVVARS if os.environ.get(k)]
    if found:
        print(
            f"\n[FATAL] Cloud LLM credentials detected in environment: {found}\n"
            "This application must run fully locally. Remove these variables "
            "before starting the server to ensure no data leaves your machine.\n",
            file=sys.stderr,
        )
        sys.exit(1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_generation_model: str = "qwen3:14b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_request_timeout: float = 300.0   # seconds; large model, slow CPU fallback

    # ── Reranker ────────────────────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"           # "cuda" | "cpu"
    reranker_top_k: int = 5                 # chunks kept after reranking
    reranker_initial_top_k: int = 20        # candidates fetched from Qdrant before rerank

    # ── Qdrant ──────────────────────────────────────────────────────────────
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "documents"
    qdrant_dense_dim: int = 768             # nomic-embed-text output dim

    # Hybrid search: weight for dense score in RRF fusion (sparse weight = 1 - this)
    hybrid_dense_weight: float = 0.6

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # ── Confidence gate ─────────────────────────────────────────────────────
    # Default is a *starting point*. Run `make eval` after loading real documents
    # to calibrate this against actual reranker score distributions.
    confidence_threshold: float = 0.4      # deliberately conservative default

    # ── Generation ──────────────────────────────────────────────────────────
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── Concurrency ─────────────────────────────────────────────────────────
    # Single-user safety valve: prevents VRAM OOM from overlapping requests.
    # This is NOT multi-tenant auth — it's purely a resource guard.
    max_concurrent_llm_calls: int = 1

    # ── SQL connector ───────────────────────────────────────────────────────
    user_db_connection_string: str | None = None  # set via Settings UI
    # Never logged or exposed to frontend; reads from env only as a convenience.

    # ── Query router ────────────────────────────────────────────────────────
    router_mode: Literal["auto", "docs", "sql", "both"] = "auto"
    # "auto" = LLM-based intent classification
    # "docs" / "sql" / "both" = force a path (useful for debugging)

    # ── App metadata DB ─────────────────────────────────────────────────────
    app_db_url: str = "sqlite+aiosqlite:///./data/app_metadata.db"

    # ── Storage paths ───────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    query_log_dir: str = "./data/query_logs"
    eval_log_dir: str = "./data/eval_logs"

    # ── Ports ───────────────────────────────────────────────────────────────
    backend_port: int = 8000

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def warn_if_cpu_reranker(self) -> "Settings":
        if self.reranker_device == "cpu":
            import warnings
            warnings.warn(
                "Reranker is configured to run on CPU. "
                "Reranking latency will be noticeably higher. "
                "Set RERANKER_DEVICE=cuda to use the GPU.",
                stacklevel=2,
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

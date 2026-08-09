"""
FastAPI application entry point.

Startup sequence:
  1. assert_no_cloud_keys() — hard fail if any cloud LLM env vars are set
  2. Create data directories
  3. init_db() — create SQLite tables
  4. ensure_collection() — create Qdrant collection if missing
  5. Health check — log warnings if Ollama/Qdrant unreachable or GPU not active
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import assert_no_cloud_keys, get_settings

# ── Logging setup ─────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup and shutdown logic using the modern lifespan pattern."""
    settings = get_settings()

    # Create storage directories
    for d in [settings.upload_dir, settings.query_log_dir, settings.eval_log_dir]:
        os.makedirs(d, exist_ok=True)

    # Init app metadata DB
    from app.db.session import init_db
    await init_db()
    logger.info("App metadata database initialised")

    # Ensure Qdrant collection exists
    from app.retrieval.qdrant import get_qdrant_store
    try:
        qdrant = get_qdrant_store()
        await qdrant.ensure_collection()
        logger.info("Qdrant collection ready")
    except Exception as exc:
        logger.warning(
            "Qdrant not reachable at startup — document search unavailable",
            error=str(exc),
        )

    # Health check — surface warnings but don't crash the server
    from app.core.pipeline import get_pipeline
    pipeline = get_pipeline()
    try:
        health = await pipeline.health_check()

        ollama = health.get("ollama", {})
        if ollama.get("status") != "ok":
            logger.warning(
                f"⚠️  Ollama is unreachable. Chat will not work until Ollama is running. "
                f"Error: {ollama.get('error')}",
            )
        else:
            if not ollama.get("generation_model_ready"):
                model = settings.ollama_generation_model
                logger.warning(
                    f"⚠️  Generation model '{model}' not found in Ollama. "
                    f"Run: ollama pull {model}"
                )
            if not ollama.get("embed_model_ready"):
                model = settings.ollama_embed_model
                logger.warning(
                    f"⚠️  Embedding model '{model}' not found in Ollama. "
                    f"Run: ollama pull {model}"
                )
            if ollama.get("gpu_active") is False:
                logger.warning(
                    "⚠️  Ollama does not appear to be using the GPU. "
                    "Generation and embedding will be noticeably slower on CPU."
                )

        reranker = health.get("reranker", {})
        if reranker.get("warning"):
            logger.warning(f"⚠️  Reranker: {reranker['warning']}")

    except Exception as exc:
        logger.warning("Startup health check failed", error=str(exc))

    logger.info(
        "LocalRAG started",
        generation_model=settings.ollama_generation_model,
        embed_model=settings.ollama_embed_model,
        reranker=settings.reranker_model,
        qdrant=settings.qdrant_url,
    )

    yield  # app runs here

    # Shutdown: nothing critical needed for a local single-user app


def create_app() -> FastAPI:
    # ── Cloud key guard (must run before any other init) ──────────────────
    assert_no_cloud_keys()

    settings = get_settings()

    app = FastAPI(
        title="LocalRAG",
        description="Fully local RAG application — no data leaves your machine",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # ── CORS (frontend dev server on :5173) ───────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000", "http://frontend:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    from app.api.chat import router as chat_router
    from app.api.database import router as db_router
    from app.api.documents import router as docs_router
    from app.api.settings import router as settings_router

    app.include_router(docs_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    app.include_router(db_router, prefix="/api")
    app.include_router(settings_router, prefix="/api")

    @app.get("/api/health")
    async def root_health():
        """Quick liveness check."""
        return {"status": "ok", "service": "localrag"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    # When run as `python app/main.py` from the backend/ directory,
    # the module path is app.main:app.
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.backend_port,
        reload=False,
    )

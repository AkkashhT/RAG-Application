.PHONY: up down build pull-models test eval eval-calibrate logs clean

# ── Docker Compose ─────────────────────────────────────────────────────────

up:
	docker compose up -d

up-cpu:
	docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d

down:
	docker compose down

build:
	docker compose build

# ── Model pulling (run after `make up`) ───────────────────────────────────

pull-models:
	@echo "Pulling qwen3:14b (generation model)…"
	docker compose exec ollama ollama pull qwen3:14b
	@echo "Pulling nomic-embed-text (embedding model)…"
	docker compose exec ollama ollama pull nomic-embed-text

pull-small:
	@echo "Pulling qwen3:8b (for <10GB VRAM)…"
	docker compose exec ollama ollama pull qwen3:8b
	docker compose exec ollama ollama pull nomic-embed-text

# ── Tests ──────────────────────────────────────────────────────────────────

test:
	cd backend && python -m pytest tests/ -v

# ── RAGAS evaluation ───────────────────────────────────────────────────────

eval:
	cd backend && python ../eval/run_eval.py

eval-calibrate:
	cd backend && python ../eval/run_eval.py --calibrate

# ── Dev convenience ────────────────────────────────────────────────────────

logs:
	docker compose logs -f backend

shell-backend:
	docker compose exec backend bash

health:
	curl -s http://localhost:8000/api/settings/health | python3 -m json.tool

# ── Cleanup ────────────────────────────────────────────────────────────────

clean:
	docker compose down -v
	@echo "All volumes removed (uploaded files, Qdrant vectors, Ollama models)"
	@echo "Run 'make pull-models' again after 'make up' to re-download models"

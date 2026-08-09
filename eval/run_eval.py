#!/usr/bin/env python3
"""
RAGAS-based evaluation script.

Usage:
    python eval/run_eval.py                    # run full eval
    python eval/run_eval.py --calibrate        # also output threshold calibration report
    python eval/run_eval.py --question-id doc_001   # run single test case

What it measures:
  - faithfulness: does the answer only use information from the retrieved context?
  - answer_relevancy: does the answer actually address the question?
  - context_precision: are the retrieved chunks relevant to the question?
  - context_recall: did retrieval find the chunks needed to answer?

Threshold calibration (--calibrate):
  Runs all test cases, logs reranker scores for known-relevant vs
  known-irrelevant questions, and reports where the distributions separate.
  Use this output to set a real confidence_threshold default rather than
  relying on the 0.4 guess in config.py.

CLOUD-DEPENDENCY AUDIT:
  RAGAS's default evaluators call OpenAI. This script overrides them to use
  the local Ollama LLM via a custom LangChain-compatible wrapper.
  If you see any requests to api.openai.com during eval, something has changed
  in the RAGAS version — check the version pinned in requirements.txt and
  inspect ragas.llms.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add backend to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.config import assert_no_cloud_keys, get_settings
from app.core.pipeline import get_pipeline

# ── RAGAS local LLM wrapper ───────────────────────────────────────────────────

def _make_local_ragas_llm():
    """
    Create a RAGAS-compatible LLM that points at the local Ollama server.

    CLOUD-DEPENDENCY AUDIT: RAGAS imports langchain under the hood and its
    default LLM is OpenAI. We override this with OllamaLLM (from langchain-ollama)
    which only ever calls localhost:11434. No OpenAI key is used or needed.
    If RAGAS starts making external calls, the assert_no_cloud_keys() guard
    at the top of this script will catch it on next run — but also check
    whether a newer RAGAS version changed the default LLM.
    """
    try:
        from langchain_ollama import OllamaLLM
        from ragas.llms import LangchainLLMWrapper

        settings = get_settings()
        ollama_llm = OllamaLLM(
            model=settings.ollama_generation_model,
            base_url=settings.ollama_base_url,
        )
        return LangchainLLMWrapper(ollama_llm)
    except ImportError as exc:
        print(
            f"[WARN] Could not create local RAGAS LLM wrapper: {exc}\n"
            "Install langchain-ollama: pip install langchain-ollama\n"
            "Falling back to non-LLM metrics only (context_precision, context_recall)."
        )
        return None


def _make_local_ragas_embeddings():
    """Local embedding wrapper for RAGAS using Ollama."""
    try:
        from langchain_ollama import OllamaEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        settings = get_settings()
        emb = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )
        return LangchainEmbeddingsWrapper(emb)
    except ImportError:
        return None


# ── Core eval runner ──────────────────────────────────────────────────────────

async def run_single_case(
    pipeline,
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Run a single test case through the full pipeline.
    Returns a result dict including the answer, retrieved chunks, and scores.
    """
    question = case["question"]
    expected_source = case["expected_source_type"]

    print(f"  Running: [{case['id']}] {question[:70]}…")
    start = time.monotonic()

    full_answer = []
    citations = []
    router_decision = None
    top_rerank_score = 0.0
    retrieved_chunks = []
    confidence_passed = None

    try:
        async for event in pipeline.run(question=question, session_id="eval"):
            if event["type"] == "token":
                full_answer.append(event["content"])
            elif event["type"] == "citations":
                citations = event["content"]
            elif event["type"] == "done":
                d = event["content"]
                router_decision = d.get("router_decision")
                top_rerank_score = d.get("top_rerank_score", 0.0)
                confidence_passed = d.get("confidence_gate_passed")
    except Exception as exc:
        print(f"    [ERROR] Pipeline failed: {exc}")
        return {
            "id": case["id"],
            "question": question,
            "expected_source_type": expected_source,
            "answer": "",
            "router_decision": "error",
            "top_rerank_score": 0.0,
            "confidence_passed": False,
            "latency_ms": (time.monotonic() - start) * 1000,
            "error": str(exc),
            "citations": [],
        }

    latency = (time.monotonic() - start) * 1000
    answer = "".join(full_answer)

    result = {
        "id": case["id"],
        "question": question,
        "expected_source_type": expected_source,
        "expected_answer_gist": case.get("expected_answer_gist", ""),
        "answer": answer,
        "router_decision": router_decision,
        "top_rerank_score": top_rerank_score,
        "confidence_passed": confidence_passed,
        "latency_ms": round(latency, 1),
        "citations": citations,
        "notes": case.get("notes", ""),
    }

    # Basic correctness checks (non-LLM)
    if expected_source == "none":
        result["gate_correct"] = router_decision == "not_found"
    else:
        result["gate_correct"] = router_decision != "not_found"

    return result


async def run_eval(
    question_id: str | None = None,
    calibrate: bool = False,
) -> None:
    """Main eval entry point."""

    # Hard guard: must run before anything else
    assert_no_cloud_keys()

    settings = get_settings()
    test_set_path = Path(__file__).parent / "test_set.json"
    test_cases = json.loads(test_set_path.read_text())

    if question_id:
        test_cases = [c for c in test_cases if c["id"] == question_id]
        if not test_cases:
            print(f"[ERROR] No test case found with id '{question_id}'")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"LocalRAG Eval — {len(test_cases)} test cases")
    print(f"Model: {settings.ollama_generation_model}")
    print(f"Confidence threshold: {settings.confidence_threshold}")
    print(f"{'='*60}\n")

    pipeline = get_pipeline()

    # ── Run all cases ──────────────────────────────────────────────────────
    results = []
    for case in test_cases:
        result = await run_single_case(pipeline, case)
        results.append(result)
        status = "✓" if result.get("gate_correct") else "✗"
        print(
            f"  {status} score={result['top_rerank_score']:.3f} "
            f"router={result['router_decision']} "
            f"latency={result['latency_ms']:.0f}ms"
        )

    # ── Basic metrics (non-LLM) ────────────────────────────────────────────
    gate_correct = sum(1 for r in results if r.get("gate_correct", False))
    print(f"\n── Confidence Gate Accuracy: {gate_correct}/{len(results)} ({100*gate_correct//len(results)}%)")

    # ── RAGAS metrics (requires Ollama to be up) ───────────────────────────
    local_llm = _make_local_ragas_llm()
    local_emb = _make_local_ragas_embeddings()

    if local_llm and local_emb:
        print("\n── Running RAGAS metrics (using local Ollama LLM)…")
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )

            # Build RAGAS dataset — only for cases that returned actual answers
            ragas_rows = []
            for r in results:
                if not r["answer"] or r["router_decision"] == "not_found":
                    continue
                contexts = [c.get("text_preview", "") for c in r["citations"] if c.get("text_preview")]
                if not contexts:
                    continue
                ragas_rows.append({
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": contexts,
                    "ground_truth": r["expected_answer_gist"],
                })

            if ragas_rows:
                ds = Dataset.from_list(ragas_rows)
                metrics = [
                    Faithfulness(llm=local_llm),
                    AnswerRelevancy(llm=local_llm, embeddings=local_emb),
                    ContextPrecision(llm=local_llm),
                    ContextRecall(llm=local_llm),
                ]
                ragas_result = evaluate(ds, metrics=metrics)
                print(f"\n── RAGAS Scores:")
                for metric, score in ragas_result.items():
                    print(f"   {metric:<25} {score:.3f}")
            else:
                print("  No answered questions with citations — skipping RAGAS metrics")
        except Exception as exc:
            print(f"  [WARN] RAGAS evaluation failed: {exc}")
            print("  (This does not affect the confidence gate accuracy metrics above)")
    else:
        print("\n── Skipping RAGAS LLM metrics (langchain-ollama not available)")

    # ── Threshold calibration ──────────────────────────────────────────────
    if calibrate:
        print("\n── Threshold Calibration Report")
        print("   (Use this to set a real confidence_threshold instead of guessing)")
        print()

        relevant_scores = [
            r["top_rerank_score"]
            for r in results
            if r["expected_source_type"] != "none" and r["top_rerank_score"] > 0
        ]
        irrelevant_scores = [
            r["top_rerank_score"]
            for r in results
            if r["expected_source_type"] == "none"
        ]

        if relevant_scores:
            print(f"   Known-RELEVANT questions (n={len(relevant_scores)}):")
            print(f"     min={min(relevant_scores):.3f}  max={max(relevant_scores):.3f}  mean={sum(relevant_scores)/len(relevant_scores):.3f}")
        if irrelevant_scores:
            print(f"   Known-IRRELEVANT questions (n={len(irrelevant_scores)}):")
            print(f"     min={min(irrelevant_scores):.3f}  max={max(irrelevant_scores):.3f}  mean={sum(irrelevant_scores)/len(irrelevant_scores):.3f}")

        if relevant_scores and irrelevant_scores:
            # Suggest a threshold between the two distributions
            suggested = (min(relevant_scores) + max(irrelevant_scores)) / 2
            print(f"\n   ► Suggested threshold: {suggested:.3f}")
            print(f"     (midpoint between lowest relevant score and highest irrelevant score)")
            print(f"     Current threshold in config: {settings.confidence_threshold}")
            print(
                f"\n   NOTE: This is calibrated on {len(test_cases)} synthetic test cases. "
                f"Re-run after loading your real documents — score distributions shift with corpus."
            )

    # ── Save results to disk ───────────────────────────────────────────────
    os.makedirs(settings.eval_log_dir, exist_ok=True)
    out_path = (
        Path(settings.eval_log_dir)
        / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": settings.ollama_generation_model,
                "confidence_threshold": settings.confidence_threshold,
                "gate_accuracy": f"{gate_correct}/{len(results)}",
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n── Results saved to {out_path}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalRAG RAGAS evaluation")
    parser.add_argument("--question-id", help="Run a single test case by ID")
    parser.add_argument("--calibrate", action="store_true", help="Include threshold calibration report")
    args = parser.parse_args()
    asyncio.run(run_eval(question_id=args.question_id, calibrate=args.calibrate))


if __name__ == "__main__":
    main()

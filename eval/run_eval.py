"""
Eval runner — runs the full RAG pipeline against the golden dataset and writes eval_results.json.
Usage: python eval/run_eval.py [--output eval_results.json]
Requires: GROQ_API_KEY and GEMINI_API_KEY in .env, indexes built via scripts/ingest_docs.py
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

from eval.llm_judge import judge
from eval.ragas_eval import context_recall, faithfulness
from src.generation.llm_client import LLMClient
from src.generation.rag_chain import RAGChain
from src.retrieval.retriever import HybridRetriever

GOLDEN_DATASET_PATH = Path("eval/golden_dataset.json")

THRESHOLDS = {
    "faithfulness":    0.75,
    "context_recall":  0.70,
    "correctness":     0.65,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="eval_results.json")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_eval(chain: RAGChain, pairs: list[dict]) -> list[dict]:
    """Runs each golden Q&A pair through the full pipeline and all eval metrics."""
    per_question = []

    for i, pair in enumerate(pairs, 1):
        qid      = pair["id"]
        question = pair["question"]
        ref      = pair["reference_answer"]
        print(f"  [{i:02d}/{len(pairs)}] {qid} — {question[:60]}")

        try:
            result = chain.query(question)
        except Exception as e:
            print(f"         RAG chain error: {e}")
            per_question.append({"id": qid, "question": question, "error": str(e)})
            continue

        answer = result["answer"]
        chunks = result["retrieved_chunks"]

        faith   = faithfulness(answer, chunks)
        recall  = context_recall(ref, chunks)
        scores  = judge(question, answer, ref)

        record = {
            "id":               qid,
            "difficulty":       pair.get("difficulty", "unknown"),
            "question":         question,
            "answer":           answer,
            "reference_answer": ref,
            "faithfulness":     faith,
            "context_recall":   recall,
            "correctness":      scores["correctness"],
            "groundedness":     scores["groundedness"],
            "citation_quality": scores["citation_quality"],
            "reasoning":        scores["reasoning"],
            "citations_found":  len(result["citations"]),
            "chunks_used":      result["chunks_used"],
            "low_confidence":   result["low_confidence"],
            "usage":            result["usage"],
        }
        per_question.append(record)
        print(
            f"         faith={faith:.2f}  recall={recall:.2f}  "
            f"correct={scores['correctness']:.2f}  ground={scores['groundedness']:.2f}  "
            f"cite_q={scores['citation_quality']:.2f}"
        )

    return per_question


def build_aggregate(per_question: list[dict]) -> dict:
    """Computes mean scores and pass_rate from per-question results."""
    valid = [r for r in per_question if "error" not in r]

    def means(key: str) -> float:
        return _mean([r[key] for r in valid])

    pass_rate = _mean([1.0 if r["correctness"] >= THRESHOLDS["correctness"] else 0.0 for r in valid])

    return {
        "mean_faithfulness":    means("faithfulness"),
        "mean_context_recall":  means("context_recall"),
        "mean_correctness":     means("correctness"),
        "mean_groundedness":    means("groundedness"),
        "mean_citation_quality": means("citation_quality"),
        "pass_rate":            pass_rate,
    }


def print_summary(aggregate: dict, thresholds_passed: dict) -> None:
    print("\n" + "=" * 65)
    print("  EVAL SUMMARY")
    print("=" * 65)
    rows = [
        ("Faithfulness",     aggregate["mean_faithfulness"],   THRESHOLDS["faithfulness"],   thresholds_passed["faithfulness"]),
        ("Context Recall",   aggregate["mean_context_recall"], THRESHOLDS["context_recall"], thresholds_passed["context_recall"]),
        ("Correctness",      aggregate["mean_correctness"],    THRESHOLDS["correctness"],    thresholds_passed["correctness"]),
        ("Groundedness",     aggregate["mean_groundedness"],   None,                         None),
        ("Citation Quality", aggregate["mean_citation_quality"], None,                       None),
        ("Pass Rate",        aggregate["pass_rate"],           None,                         None),
    ]
    for name, score, threshold, passed in rows:
        thresh_str = f"(>= {threshold:.2f})" if threshold is not None else "       "
        status = ""
        if passed is not None:
            status = "  [PASS]" if passed else "  [FAIL]"
        print(f"  {name:<20} {score:.3f}  {thresh_str}{status}")
    print()


def main() -> None:
    args = parse_args()

    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    pairs   = dataset["pairs"]
    print(f"Loaded {len(pairs)} golden Q&A pairs\n")

    print("Initializing pipeline...")
    chain = RAGChain(HybridRetriever(), LLMClient())
    print("Ready. Running eval...\n")

    per_question = run_eval(chain, pairs)
    aggregate    = build_aggregate(per_question)

    thresholds_passed = {
        "faithfulness":   aggregate["mean_faithfulness"]   >= THRESHOLDS["faithfulness"],
        "context_recall": aggregate["mean_context_recall"] >= THRESHOLDS["context_recall"],
        "correctness":    aggregate["mean_correctness"]    >= THRESHOLDS["correctness"],
    }

    output = {
        "run_id":             datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "num_questions":      len(pairs),
        "aggregate":          aggregate,
        "per_question":       per_question,
        "thresholds_passed":  thresholds_passed,
    }

    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results written to {args.output}")

    print_summary(aggregate, thresholds_passed)

    if not all(thresholds_passed.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

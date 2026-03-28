"""
Retrieval pipeline health check.
Usage: python scripts/test_retrieval.py
"""

import logging
import time

logging.disable(logging.CRITICAL)

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import fuse
from src.retrieval.reranker import Reranker
from src.retrieval.vector_retriever import VectorRetriever

QUERIES = [
    "how does reciprocal rank fusion combine sparse and dense retrieval",
    "BERT masked language model pretraining",
    "hallucination in large language models",
]

LATENCY_LIMIT_SECONDS = 5.0
RESULTS: dict[str, bool] = {}


def _pass(label: str, msg: str) -> None:
    RESULTS[label] = True
    print(f"  [PASS] {msg}")


def _fail(label: str, msg: str) -> None:
    RESULTS[label] = False
    print(f"  [FAIL] {msg}")


def _header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def run_pipeline_checks(vector: VectorRetriever, bm25: BM25Retriever, reranker: Reranker) -> None:
    for query in QUERIES:
        _header(f"Query: {query[:60]}")

        t0 = time.perf_counter()
        vector_results = vector.retrieve(query, top_k=20)
        t_vector = time.perf_counter() - t0

        t1 = time.perf_counter()
        bm25_results = bm25.retrieve(query, top_k=20)
        t_bm25 = time.perf_counter() - t1

        t2 = time.perf_counter()
        fused = fuse(vector_results, bm25_results)
        t_fusion = time.perf_counter() - t2

        t3 = time.perf_counter()
        reranked = reranker.rerank(query, fused, top_n=5)
        t_rerank = time.perf_counter() - t3

        total = t_vector + t_bm25 + t_fusion + t_rerank

        # ── Results table ────────────────────────────────────────────────────
        print(f"\n  {'#':<3} {'rerank':>8} {'rrf':>8} {'vector':>8} {'bm25':>8}  title")
        print(f"  {'-' * 62}")
        for i, r in enumerate(reranked, 1):
            rerank = f"{r.rerank_score:.3f}" if r.rerank_score is not None else "   -"
            rrf    = f"{r.rrf_score:.4f}"    if r.rrf_score    is not None else "      -"
            vec    = f"{r.vector_score:.3f}" if r.vector_score  is not None else "   -"
            bm     = f"{r.bm25_score:.3f}"   if r.bm25_score   is not None else "   -"
            print(f"  {i:<3} {rerank:>8} {rrf:>8} {vec:>8} {bm:>8}  {r.title[:35]}")
            print(f"      {r.content[:110].replace(chr(10), ' ')}...")

        # ── Latency ──────────────────────────────────────────────────────────
        print(f"\n  Latency  vector={t_vector:.3f}s  bm25={t_bm25:.3f}s  fusion={t_fusion:.4f}s  rerank={t_rerank:.3f}s  total={total:.3f}s")

        q_label = query[:30]

        if total > LATENCY_LIMIT_SECONDS:
            _fail(f"latency:{q_label}", f"Total latency {total:.2f}s exceeds {LATENCY_LIMIT_SECONDS}s limit")
        else:
            _pass(f"latency:{q_label}", f"Total latency {total:.2f}s within {LATENCY_LIMIT_SECONDS}s limit")

        # ── Rerank order strictly descending ─────────────────────────────────
        scores = [r.rerank_score for r in reranked if r.rerank_score is not None]
        if scores == sorted(scores, reverse=True):
            _pass(f"order:{q_label}", "rerank_score is strictly descending")
        else:
            _fail(f"order:{q_label}", "rerank_score is NOT in descending order")

        # ── RRF sourced from both retrievers ─────────────────────────────────
        has_vector = any(r.vector_score is not None for r in fused[:5])
        has_bm25   = any(r.bm25_score   is not None for r in fused[:5])

        if has_vector and has_bm25:
            _pass(f"diversity:{q_label}", "Top-5 fused results contain chunks from both vector and BM25")
        elif not has_vector:
            _fail(f"diversity:{q_label}", "No vector_score in top-5 fused results")
        else:
            _fail(f"diversity:{q_label}", "No bm25_score in top-5 fused results")


def print_summary() -> None:
    _header("SUMMARY")
    passed = sum(RESULTS.values())
    total = len(RESULTS)
    print(f"\n  {passed}/{total} checks passed\n")
    for label, ok in RESULTS.items():
        print(f"  {'[PASS]' if ok else '[FAIL]'} {label}")
    print()


def main() -> None:
    print("Loading retrievers...")
    vector   = VectorRetriever()
    bm25     = BM25Retriever()
    reranker = Reranker()

    run_pipeline_checks(vector, bm25, reranker)
    print_summary()


if __name__ == "__main__":
    main()

"""
Phase 1 pipeline health check.
Usage: python scripts/health_check.py
"""

import json
import logging
import pickle
import random
import statistics
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.disable(logging.CRITICAL)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
FAISS_PATH = Path("indexes/faiss.index")
BM25_PATH = Path("indexes/bm25.pkl")
METADATA_PATH = Path("indexes/metadata.jsonl")

REQUIRED_FIELDS = {"doc_id", "source", "title", "chunk_index", "text"}
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RESULTS: dict[str, bool] = {}


def _pass(label: str, msg: str = "") -> None:
    RESULTS[label] = True
    print(f"  [PASS] {msg or label}")


def _fail(label: str, msg: str = "") -> None:
    RESULTS[label] = False
    print(f"  [FAIL] {msg or label}")


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ── 1. Chunks file integrity ──────────────────────────────────────────────────

def check_chunks_integrity(chunks: list[dict]) -> None:
    _header("1. Chunks File Integrity")

    print(f"  Total chunks: {len(chunks)}")

    missing_fields = [c for c in chunks if not REQUIRED_FIELDS.issubset(c.keys())]
    if missing_fields:
        _fail("fields", f"{len(missing_fields)} chunks missing required fields")
    else:
        _pass("fields", "All chunks have required fields")

    empty_content = [c for c in chunks if not c.get("text", "").strip()]
    if empty_content:
        print(f"  WARNING: {len(empty_content)} chunks have empty/whitespace text")
    else:
        print("  Empty content: 0")

    lengths = [len(c["text"]) for c in chunks]
    print(f"  Char length  min={min(lengths)}  max={max(lengths)}  mean={statistics.mean(lengths):.0f}")

    if not missing_fields and not empty_content:
        _pass("integrity", "Integrity check passed")
    elif not missing_fields:
        _pass("integrity", "Integrity check passed (empty content warnings noted)")


# ── 2. Chunk size sanity ──────────────────────────────────────────────────────

def check_chunk_sizes(chunks: list[dict], model: SentenceTransformer) -> None:
    _header("2. Chunk Size Sanity")

    sample = random.sample(chunks, min(200, len(chunks)))
    tokenizer = model.tokenizer
    token_counts = [
        len(tokenizer.encode(c["text"], add_special_tokens=False))
        for c in sample
    ]

    buckets = {
        "under_50":   sum(1 for t in token_counts if t < 50),
        "50_200":     sum(1 for t in token_counts if 50 <= t < 200),
        "200_400":    sum(1 for t in token_counts if 200 <= t < 400),
        "400_512":    sum(1 for t in token_counts if 400 <= t <= 512),
        "over_512":   sum(1 for t in token_counts if t > 512),
    }
    total = len(token_counts)

    print(f"  Sample size: {total}")
    for label, count in buckets.items():
        print(f"  {label:12s}: {count:4d}  ({count / total * 100:.1f}%)")

    pct_small = buckets["under_50"] / total
    pct_large = buckets["over_512"] / total

    if pct_small > 0.05:
        _fail("chunk_sizes", f"{pct_small:.1%} of chunks under 50 tokens (threshold: 5%)")
    elif pct_large > 0.05:
        _fail("chunk_sizes", f"{pct_large:.1%} of chunks over 512 tokens (threshold: 5%)")
    else:
        _pass("chunk_sizes", "Token distribution within acceptable bounds")


# ── 3. FAISS index ────────────────────────────────────────────────────────────

def check_faiss_index(chunks: list[dict], metadata: list[dict], model: SentenceTransformer) -> None:
    _header("3. FAISS Index")

    index = faiss.read_index(str(FAISS_PATH))
    print(f"  Vectors in index : {index.ntotal}")
    print(f"  Chunks in file   : {len(chunks)}")

    if index.ntotal != len(chunks):
        _fail("faiss_count", f"Vector count mismatch: {index.ntotal} vs {len(chunks)} chunks")
    else:
        _pass("faiss_count", "Vector count matches chunk count")

    query = "what is retrieval augmented generation"
    embedding = model.encode(
        [f"{BGE_QUERY_PREFIX}{query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = index.search(embedding, 5)
    print(f"\n  Query: \"{query}\"")
    print(f"  {'Score':>6}  Title / Snippet")
    print(f"  {'-' * 55}")
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        snippet = chunk["text"][:100].replace("\n", " ")
        print(f"  {score:6.4f}  [{chunk['title'][:40]}]")
        print(f"          {snippet}...")


# ── 4. BM25 index ─────────────────────────────────────────────────────────────

def check_bm25_index(chunks: list[dict], metadata: list[dict]) -> None:
    _header("4. BM25 Index")

    with BM25_PATH.open("rb") as f:
        bm25 = pickle.load(f)

    print(f"  BM25 corpus size : {bm25.corpus_size}")
    print(f"  Chunks in file   : {len(chunks)}")

    if bm25.corpus_size != len(chunks):
        _fail("bm25_count", f"BM25 corpus size mismatch: {bm25.corpus_size} vs {len(chunks)}")
    else:
        _pass("bm25_count", "BM25 corpus size matches chunk count")

    query = "what is retrieval augmented generation"
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top5 = np.argsort(scores)[::-1][:5]

    print(f"\n  Query: \"{query}\"")
    print(f"  {'Score':>6}  Title / Snippet")
    print(f"  {'-' * 55}")
    for idx in top5:
        chunk = metadata[idx]
        snippet = chunk["text"][:100].replace("\n", " ")
        print(f"  {scores[idx]:6.4f}  [{chunk['title'][:40]}]")
        print(f"          {snippet}...")


# ── 5. Retrieval diversity ────────────────────────────────────────────────────

def check_retrieval_diversity(chunks: list[dict], metadata: list[dict], model: SentenceTransformer) -> None:
    _header("5. Retrieval Diversity Check")

    with BM25_PATH.open("rb") as f:
        bm25 = pickle.load(f)
    index = faiss.read_index(str(FAISS_PATH))

    queries = [
        "what is retrieval augmented generation",
        "BERT pretraining objective masked language model",
        "how does cross-encoder reranking work",
    ]

    all_identical = True

    for query in queries:
        embedding = model.encode(
            [f"{BGE_QUERY_PREFIX}{query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        _, faiss_indices = index.search(embedding, 3)
        faiss_top3 = set(faiss_indices[0].tolist())

        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_top3 = set(np.argsort(bm25_scores)[::-1][:3].tolist())

        overlap = faiss_top3 & bm25_top3
        identical = faiss_top3 == bm25_top3
        if not identical:
            all_identical = False

        status = "identical" if identical else f"{len(overlap)}/3 overlap"
        print(f"  Q: \"{query[:50]}\"")
        print(f"     FAISS top-3 : {[metadata[i]['title'][:30] for i in faiss_top3 if i != -1]}")
        print(f"     BM25  top-3 : {[metadata[i]['title'][:30] for i in bm25_top3]}")
        print(f"     Overlap     : {status}\n")

    if all_identical:
        _fail("diversity", "FAISS and BM25 returned identical results for all 3 queries")
    else:
        _pass("diversity", "FAISS and BM25 return different results — hybrid retrieval is working")


# ── 6. Metadata coverage ──────────────────────────────────────────────────────

def check_metadata_coverage(chunks: list[dict]) -> None:
    _header("6. Metadata Coverage")

    unique_docs = {c["doc_id"] for c in chunks}
    unique_titles = {c["title"] for c in chunks}

    print(f"  Unique doc_ids : {len(unique_docs)}")
    print(f"  Unique titles  : {len(unique_titles)}")

    if len(unique_docs) < 50:
        _fail("coverage", f"Only {len(unique_docs)} unique documents (minimum: 50)")
    else:
        _pass("coverage", f"{len(unique_docs)} unique source documents")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    _header("SUMMARY")
    passed = sum(RESULTS.values())
    total = len(RESULTS)
    print(f"\n  {passed}/{total} checks passed\n")
    for label, ok in RESULTS.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    for path in (CHUNKS_PATH, FAISS_PATH, BM25_PATH, METADATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
    metadata = [json.loads(l) for l in METADATA_PATH.read_text(encoding="utf-8").splitlines()]

    print("Loading BGE model...")
    model = SentenceTransformer(BGE_MODEL)

    check_chunks_integrity(chunks)
    check_chunk_sizes(chunks, model)
    check_faiss_index(chunks, metadata, model)
    check_bm25_index(chunks, metadata)
    check_retrieval_diversity(chunks, metadata, model)
    check_metadata_coverage(chunks)
    print_summary()


if __name__ == "__main__":
    main()

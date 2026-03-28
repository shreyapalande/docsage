"""Reciprocal Rank Fusion for combining dense and sparse retrieval results."""

import logging

from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_TOP_AFTER_FUSION = 20


def fuse(
    vector_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Merges vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = sum of 1 / (k + rank) across both ranked lists.
    Deduplication is by (doc_id, chunk_index). The winning chunk copy
    carries both the rrf_score and whichever individual scores were set.
    Returns the top 20 chunks sorted by rrf_score descending.
    """
    rrf_scores: dict[tuple[str, int], float] = {}
    chunk_map: dict[tuple[str, int], RetrievedChunk] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        key = (chunk.doc_id, chunk.chunk_index)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        key = (chunk.doc_id, chunk.chunk_index)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in chunk_map:
            chunk_map[key] = chunk
        else:
            existing = chunk_map[key]
            chunk_map[key] = RetrievedChunk(
                doc_id=existing.doc_id,
                title=existing.title,
                source=existing.source,
                chunk_index=existing.chunk_index,
                content=existing.content,
                vector_score=existing.vector_score,
                bm25_score=chunk.bm25_score,
                rrf_score=None,
                rerank_score=None,
            )

    fused = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:_TOP_AFTER_FUSION]

    results = []
    for key in fused:
        chunk = chunk_map[key]
        results.append(
            RetrievedChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                vector_score=chunk.vector_score,
                bm25_score=chunk.bm25_score,
                rrf_score=rrf_scores[key],
                rerank_score=None,
            )
        )

    logger.debug("RRF fusion: %d vector + %d bm25 -> %d fused", len(vector_results), len(bm25_results), len(results))
    return results

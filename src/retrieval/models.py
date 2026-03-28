"""Shared data models for the retrieval and reranking pipeline."""

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    source: str
    chunk_index: int
    content: str
    vector_score: float | None
    bm25_score: float | None
    rrf_score: float | None
    rerank_score: float | None

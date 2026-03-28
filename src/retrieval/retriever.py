"""Orchestrates vector retrieval, BM25 retrieval, RRF fusion, and cross-encoder reranking."""

import logging

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid import fuse
from src.retrieval.models import RetrievedChunk
from src.retrieval.reranker import Reranker
from src.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Full retrieval pipeline: dense + sparse → RRF fusion → cross-encoder reranking.

    Each stage operates on RetrievedChunk objects, progressively filling score fields:
      vector_score (FAISS) → bm25_score → rrf_score (fusion) → rerank_score (cross-encoder)
    """

    def __init__(self, fetch_k: int = 20) -> None:
        self._vector = VectorRetriever()
        self._bm25 = BM25Retriever()
        self._reranker = Reranker()
        self._fetch_k = fetch_k

    def retrieve(self, query: str, top_n: int = 5, doc_filter: str | None = None) -> list[RetrievedChunk]:
        """
        Runs the full retrieval pipeline for a query.

        If doc_filter is provided, only chunks whose doc_id matches are passed to the reranker.
        RRF fusion still runs over the full unfiltered candidate set.
        """
        vector_results = self._vector.retrieve(query, top_k=self._fetch_k)
        bm25_results = self._bm25.retrieve(query, top_k=self._fetch_k)

        fused = fuse(vector_results, bm25_results)

        candidates = (
            [c for c in fused if c.doc_id == doc_filter]
            if doc_filter
            else fused
        )

        if not candidates:
            logger.warning("doc_filter='%s' matched 0 chunks after fusion", doc_filter)
            return []

        return self._reranker.rerank(query, candidates, top_n=top_n)

"""Sparse BM25 retriever backed by a pre-built rank-bm25 corpus."""

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_DEFAULT_BM25_PATH = Path("indexes/bm25.pkl")
_DEFAULT_METADATA_PATH = Path("indexes/metadata.jsonl")


class BM25Retriever:
    """
    Retrieves chunks using BM25Okapi keyword scoring.
    Query is lowercased and whitespace-tokenized to match the indexing strategy.
    """

    def __init__(
        self,
        bm25_path: Path = _DEFAULT_BM25_PATH,
        metadata_path: Path = _DEFAULT_METADATA_PATH,
    ) -> None:
        with bm25_path.open("rb") as f:
            self._bm25 = pickle.load(f)
        self._metadata = self._load_metadata(metadata_path)
        logger.info("BM25Retriever loaded — %d documents", self._bm25.corpus_size)

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        return query.lower().split()

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Returns top_k chunks ranked by BM25 score."""
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] == 0.0:
                continue
            chunk = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    doc_id=chunk["doc_id"],
                    title=chunk["title"],
                    source=chunk["source"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["text"],
                    vector_score=None,
                    bm25_score=float(scores[idx]),
                    rrf_score=None,
                    rerank_score=None,
                )
            )

        return results

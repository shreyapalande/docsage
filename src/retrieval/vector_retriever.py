"""Dense retriever backed by a FAISS index and BAAI/bge-small-en-v1.5 embeddings."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_BGE_MODEL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_DEFAULT_FAISS_PATH = Path("indexes/faiss.index")
_DEFAULT_METADATA_PATH = Path("indexes/metadata.jsonl")


class VectorRetriever:
    """
    Retrieves chunks by dense vector similarity using FAISS IndexFlatIP.
    Scores are inner-product values over L2-normalised embeddings (equivalent to cosine similarity).
    """

    def __init__(
        self,
        faiss_path: Path = _DEFAULT_FAISS_PATH,
        metadata_path: Path = _DEFAULT_METADATA_PATH,
        model_name: str = _BGE_MODEL,
    ) -> None:
        self._index = faiss.read_index(str(faiss_path))
        self._metadata = self._load_metadata(metadata_path)
        self._model = SentenceTransformer(model_name)
        logger.info("VectorRetriever loaded — %d vectors", self._index.ntotal)

    @staticmethod
    def _load_metadata(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _encode_query(self, query: str) -> np.ndarray:
        return self._model.encode(
            [f"{_BGE_QUERY_PREFIX}{query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def retrieve(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        """Returns top_k chunks ranked by cosine similarity to the query."""
        embedding = self._encode_query(query)
        scores, indices = self._index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    doc_id=chunk["doc_id"],
                    title=chunk["title"],
                    source=chunk["source"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["text"],
                    vector_score=float(score),
                    bm25_score=None,
                    rrf_score=None,
                    rerank_score=None,
                )
            )

        return results

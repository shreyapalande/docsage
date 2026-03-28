"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2."""

import logging

from sentence_transformers import CrossEncoder

from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Scores (query, passage) pairs with a cross-encoder and returns the top_n chunks.
    Cross-encoders attend jointly to query and passage, giving higher precision than bi-encoders.
    """

    def __init__(self, model_name: str = _CROSS_ENCODER_MODEL) -> None:
        self._model = CrossEncoder(model_name)
        logger.info("Reranker loaded — %s", model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        """Scores each (query, chunk.content) pair and returns top_n by rerank_score descending."""
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self._model.predict(pairs)

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                vector_score=chunk.vector_score,
                bm25_score=chunk.bm25_score,
                rrf_score=chunk.rrf_score,
                rerank_score=float(score),
            )
            for score, chunk in ranked[:top_n]
        ]

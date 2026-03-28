"""Semantic chunker that splits documents on cosine similarity drops between consecutive sentences."""

import logging
from dataclasses import dataclass

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.loader import Document

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_SIMILARITY_THRESHOLD = 0.4
_MIN_TOKENS = 100
_MAX_TOKENS = 512


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    source: str
    title: str
    chunk_index: int
    text: str
    token_count: int


class SemanticChunker:
    """
    Splits document content into semantically coherent chunks using sentence embeddings.

    Sentences are grouped until a cosine similarity drop below `similarity_threshold`
    is detected and the current chunk meets `min_tokens`. Hard upper bound is `max_tokens`.
    """

    def __init__(
        self,
        model_name: str = _EMBEDDING_MODEL,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        min_tokens: int = _MIN_TOKENS,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.model = SentenceTransformer(model_name)
        self._ensure_nltk()

    def _ensure_nltk(self) -> None:
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(resource.split("/")[1], quiet=True)

    def _count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

    def _embed(self, sentences: list[str]) -> np.ndarray:
        return self.model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _build_chunks(self, sentences: list[str], similarities: np.ndarray) -> list[str]:
        """
        Greedily groups sentences into chunks, flushing when a semantic break is detected
        or the max token limit would be exceeded.
        """
        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_tokens = self._count_tokens(sentences[0])

        for i, sentence in enumerate(sentences[1:], start=1):
            sentence_tokens = self._count_tokens(sentence)
            sim = similarities[i - 1]
            would_exceed = (current_tokens + sentence_tokens) > self.max_tokens
            semantic_break = sim < self.similarity_threshold and current_tokens >= self.min_tokens

            if would_exceed or semantic_break:
                chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = sentence_tokens
            else:
                current.append(sentence)
                current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _merge_trailing_small_chunk(self, chunks: list[str]) -> list[str]:
        """Merges the final chunk into the previous one if it falls below min_tokens."""
        if len(chunks) < 2:
            return chunks
        if self._count_tokens(chunks[-1]) < self.min_tokens:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()
        return chunks

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Returns an ordered list of Chunks for a single Document."""
        sentences = self._split_sentences(doc.content)
        if not sentences:
            return []

        if len(sentences) == 1:
            return [
                Chunk(
                    doc_id=doc.arxiv_id,
                    source=doc.source,
                    title=doc.title,
                    chunk_index=0,
                    text=sentences[0],
                    token_count=self._count_tokens(sentences[0]),
                )
            ]

        embeddings = self._embed(sentences)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)

        raw_chunks = self._build_chunks(sentences, similarities)
        final_chunks = self._merge_trailing_small_chunk(raw_chunks)

        return [
            Chunk(
                doc_id=doc.arxiv_id,
                source=doc.source,
                title=doc.title,
                chunk_index=idx,
                text=text,
                token_count=self._count_tokens(text),
            )
            for idx, text in enumerate(final_chunks)
        ]

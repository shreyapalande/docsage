"""Parses inline [doc_id:chunk_index] citations from LLM answers and resolves them to chunk metadata."""

import re

from src.retrieval.models import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9.\-]+):(\d+)\]")
_SNIPPET_LENGTH = 150


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """
    Parses all [doc_id:chunk_index] references from the answer and resolves each
    to its source chunk. Duplicate references to the same chunk are collapsed to one entry.
    References that don't match any chunk in the provided list are silently ignored.
    """
    chunk_index: dict[tuple[str, int], RetrievedChunk] = {
        (c.doc_id, c.chunk_index): c for c in chunks
    }

    seen: set[tuple[str, int]] = set()
    citations: list[dict] = []

    for doc_id, raw_index in _CITATION_PATTERN.findall(answer):
        key = (doc_id, int(raw_index))
        if key in seen:
            continue
        seen.add(key)

        chunk = chunk_index.get(key)
        if chunk is None:
            continue

        citations.append({
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "cited_text_snippet": chunk.content[:_SNIPPET_LENGTH],
        })

    return citations

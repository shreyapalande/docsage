"""Tests for citation extraction."""

from src.context.citation_builder import extract_citations
from src.retrieval.models import RetrievedChunk


def _chunk(doc_id: str, chunk_index: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, title=f"Title {doc_id}", source="http://example.com",
        chunk_index=chunk_index, content=content,
        vector_score=None, bm25_score=None, rrf_score=None, rerank_score=None,
    )


CHUNKS = [
    _chunk("2101.00001", 0, "BERT uses masked language modeling for pretraining."),
    _chunk("2101.00002", 1, "The Transformer relies on self-attention mechanisms."),
    _chunk("2101.00003", 0, "RAG combines retrieval with generation."),
]


def test_extracts_single_citation():
    answer = "BERT is pretrained using MLM [2101.00001:0]."
    citations = extract_citations(answer, CHUNKS)
    assert len(citations) == 1
    assert citations[0]["doc_id"] == "2101.00001"
    assert citations[0]["chunk_index"] == 0


def test_extracts_multiple_citations():
    answer = "BERT [2101.00001:0] and Transformer [2101.00002:1] are both influential."
    citations = extract_citations(answer, CHUNKS)
    assert len(citations) == 2


def test_deduplicates_repeated_citation():
    answer = "BERT [2101.00001:0] is used here and here [2101.00001:0]."
    citations = extract_citations(answer, CHUNKS)
    assert len(citations) == 1


def test_ignores_unresolved_reference():
    answer = "Some claim [9999.99999:0]."
    citations = extract_citations(answer, CHUNKS)
    assert len(citations) == 0


def test_snippet_length():
    answer = "RAG is described here [2101.00003:0]."
    citations = extract_citations(answer, CHUNKS)
    assert len(citations[0]["cited_text_snippet"]) <= 150


def test_citation_contains_required_fields():
    answer = "See [2101.00001:0]."
    citations = extract_citations(answer, CHUNKS)
    assert all(k in citations[0] for k in ("doc_id", "title", "source", "chunk_index", "cited_text_snippet"))

"""Tests for the context packer."""

import tiktoken

from src.context.packer import pack
from src.retrieval.models import RetrievedChunk


def _chunk(doc_id: str, content: str, rerank_score: float) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, title=f"Title {doc_id}", source="http://example.com",
        chunk_index=0, content=content, vector_score=None, bm25_score=None,
        rrf_score=None, rerank_score=rerank_score,
    )


CHUNKS = [
    _chunk("a", "Transformers revolutionized NLP with self-attention.", 0.9),
    _chunk("b", "BERT uses masked language modeling for pretraining.", 0.8),
    _chunk("c", "RAG combines retrieval with language model generation.", 0.7),
]


def test_pack_returns_xml():
    result = pack("test query", CHUNKS)
    assert result.startswith("<documents>")
    assert result.endswith("</documents>")


def test_pack_contains_all_chunks():
    result = pack("test query", CHUNKS, max_tokens=3000)
    for chunk in CHUNKS:
        assert chunk.doc_id in result


def test_pack_respects_token_budget():
    enc = tiktoken.get_encoding("cl100k_base")
    result = pack("test query", CHUNKS, max_tokens=100)
    assert len(enc.encode(result)) <= 110


def test_pack_orders_by_rerank_score():
    result = pack("test query", CHUNKS)
    pos_a = result.index('doc_id="a"')
    pos_b = result.index('doc_id="b"')
    pos_c = result.index('doc_id="c"')
    assert pos_a < pos_b < pos_c


def test_pack_includes_relevance_score():
    result = pack("test query", CHUNKS)
    assert 'relevance="0.900"' in result


def test_pack_empty_chunks():
    result = pack("test query", [])
    assert result == "<documents>\n</documents>"

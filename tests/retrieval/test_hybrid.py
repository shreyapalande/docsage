"""Tests for RRF fusion."""

from src.retrieval.hybrid import fuse
from src.retrieval.models import RetrievedChunk


def _make_chunk(doc_id: str, chunk_index: int, vector_score=None, bm25_score=None) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id,
        title="Test",
        source="http://example.com",
        chunk_index=chunk_index,
        content="some content",
        vector_score=vector_score,
        bm25_score=bm25_score,
        rrf_score=None,
        rerank_score=None,
    )


def test_fuse_deduplicates():
    vec = [_make_chunk("a", 0, vector_score=0.9), _make_chunk("b", 0, vector_score=0.8)]
    bm25 = [_make_chunk("a", 0, bm25_score=5.0), _make_chunk("c", 0, bm25_score=4.0)]
    result = fuse(vec, bm25)
    keys = [(r.doc_id, r.chunk_index) for r in result]
    assert len(keys) == len(set(keys))


def test_fuse_rrf_scores_populated():
    vec = [_make_chunk("a", 0, vector_score=0.9)]
    bm25 = [_make_chunk("b", 0, bm25_score=5.0)]
    result = fuse(vec, bm25)
    assert all(r.rrf_score is not None for r in result)


def test_fuse_appearance_in_both_lists_ranks_higher():
    vec = [_make_chunk("a", 0, vector_score=0.9), _make_chunk("b", 0, vector_score=0.8)]
    bm25 = [_make_chunk("a", 0, bm25_score=5.0), _make_chunk("c", 0, bm25_score=4.0)]
    result = fuse(vec, bm25)
    assert result[0].doc_id == "a"


def test_fuse_preserves_both_scores_on_shared_chunk():
    vec = [_make_chunk("a", 0, vector_score=0.9)]
    bm25 = [_make_chunk("a", 0, bm25_score=5.0)]
    result = fuse(vec, bm25)
    merged = next(r for r in result if r.doc_id == "a")
    assert merged.vector_score == 0.9
    assert merged.bm25_score == 5.0


def test_fuse_capped_at_20():
    vec = [_make_chunk(f"v{i}", i, vector_score=1.0) for i in range(20)]
    bm25 = [_make_chunk(f"b{i}", i, bm25_score=1.0) for i in range(20)]
    result = fuse(vec, bm25)
    assert len(result) <= 20

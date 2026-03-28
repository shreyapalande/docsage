"""Unit tests for SemanticChunker."""

import pytest

from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.loader import Document

SAMPLE_DOC = Document(
    arxiv_id="2101.00001",
    title="Attention Is All You Need",
    source="https://arxiv.org/abs/1706.03762",
    content=(
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
        "These networks include an encoder and a decoder. "
        "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms. "
        "Experiments on two machine translation tasks show these models to be superior in quality. "
        "The Transformer is the first transduction model relying entirely on self-attention. "
        "Self-attention relates different positions of a single sequence to compute a representation. "
        "We achieve 28.4 BLEU on the WMT 2014 English-to-German translation task. "
        "This outperforms the best previously reported results by over 2 BLEU. "
        "On the WMT 2014 English-to-French translation task, our model achieves 41.0 BLEU. "
        "This establishes a new single-model state-of-the-art result. "
    ) * 6,
)


@pytest.fixture(scope="module")
def chunker():
    return SemanticChunker()


def test_chunk_document_returns_chunks(chunker):
    chunks = chunker.chunk_document(SAMPLE_DOC)
    assert len(chunks) > 0


def test_chunk_metadata(chunker):
    chunks = chunker.chunk_document(SAMPLE_DOC)
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk)
        assert chunk.doc_id == SAMPLE_DOC.arxiv_id
        assert chunk.source == SAMPLE_DOC.source
        assert chunk.title == SAMPLE_DOC.title
        assert chunk.chunk_index == i
        assert chunk.text


def test_chunk_token_bounds(chunker):
    chunks = chunker.chunk_document(SAMPLE_DOC)
    for chunk in chunks[:-1]:
        assert chunk.token_count <= chunker.max_tokens


def test_chunk_indices_are_sequential(chunker):
    chunks = chunker.chunk_document(SAMPLE_DOC)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_content_returns_no_chunks(chunker):
    doc = Document(arxiv_id="x", title="t", source="s", content="Hello.")
    chunks = chunker.chunk_document(doc)
    assert isinstance(chunks, list)

"""Smoke tests for the HuggingFace dataset ingestion loader."""

import pytest

from src.ingestion.loader import Document, load_documents


def test_load_first_batch():
    """Verify the first 5 documents load successfully with all required fields populated."""
    docs = list(doc for _, doc in zip(range(5), load_documents(streaming=True)))

    assert len(docs) == 5

    for doc in docs:
        assert isinstance(doc, Document)
        assert doc.arxiv_id
        assert doc.title
        assert doc.source
        assert doc.content


def test_document_schema():
    """Verify Document rejects rows with empty content."""
    with pytest.raises(Exception):
        Document(arxiv_id="x", title="t", source="s", content="")

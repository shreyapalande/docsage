"""Unit tests for the FAISS + BM25 indexer."""

import json
import pickle
import tempfile
from pathlib import Path

import faiss
import pytest

from src.indexing.indexer import build_indexes

SAMPLE_CHUNKS = [
    {
        "doc_id": "2101.00001",
        "source": "http://arxiv.org/pdf/2101.00001",
        "title": "Attention Is All You Need",
        "chunk_index": 0,
        "text": "The Transformer model relies entirely on self-attention mechanisms to draw global dependencies.",
        "token_count": 18,
    },
    {
        "doc_id": "2101.00002",
        "source": "http://arxiv.org/pdf/2101.00002",
        "title": "BERT Pre-training",
        "chunk_index": 0,
        "text": "BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
        "token_count": 16,
    },
    {
        "doc_id": "2101.00003",
        "source": "http://arxiv.org/pdf/2101.00003",
        "title": "GPT Language Models",
        "chunk_index": 0,
        "text": "GPT uses a unidirectional transformer trained with a language modeling objective.",
        "token_count": 14,
    },
]


@pytest.fixture(scope="module")
def index_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        chunks_file = Path(tmpdir) / "chunks.jsonl"
        with chunks_file.open("w") as f:
            for chunk in SAMPLE_CHUNKS:
                f.write(json.dumps(chunk) + "\n")

        import src.indexing.indexer as idx_module
        original_dir = idx_module._INDEXES_DIR
        original_faiss = idx_module._FAISS_INDEX_PATH
        original_bm25 = idx_module._BM25_PATH
        original_meta = idx_module._METADATA_PATH

        idx_module._INDEXES_DIR = Path(tmpdir) / "indexes"
        idx_module._FAISS_INDEX_PATH = idx_module._INDEXES_DIR / "faiss.index"
        idx_module._BM25_PATH = idx_module._INDEXES_DIR / "bm25.pkl"
        idx_module._METADATA_PATH = idx_module._INDEXES_DIR / "metadata.jsonl"

        build_indexes(chunks_path=chunks_file)

        yield idx_module._INDEXES_DIR

        idx_module._INDEXES_DIR = original_dir
        idx_module._FAISS_INDEX_PATH = original_faiss
        idx_module._BM25_PATH = original_bm25
        idx_module._METADATA_PATH = original_meta


def test_faiss_index_created(index_dir):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    assert index.ntotal == len(SAMPLE_CHUNKS)


def test_faiss_index_dimension(index_dir):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    assert index.d == 384


def test_bm25_index_created(index_dir):
    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    assert bm25.corpus_size == len(SAMPLE_CHUNKS)


def test_metadata_written(index_dir):
    lines = (index_dir / "metadata.jsonl").read_text().strip().split("\n")
    assert len(lines) == len(SAMPLE_CHUNKS)
    first = json.loads(lines[0])
    assert first["doc_id"] == SAMPLE_CHUNKS[0]["doc_id"]

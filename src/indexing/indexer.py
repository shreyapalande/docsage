"""Builds and persists a FAISS dense index and a BM25 sparse index from chunk records."""

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_INDEXES_DIR = Path("indexes")
_FAISS_INDEX_PATH = _INDEXES_DIR / "faiss.index"
_BM25_PATH = _INDEXES_DIR / "bm25.pkl"
_METADATA_PATH = _INDEXES_DIR / "metadata.jsonl"


def _load_chunks(chunks_path: Path) -> list[dict]:
    """Loads all chunk records from a JSONL file."""
    with chunks_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Embeds texts using BGE with its recommended passage prefix.
    Returns L2-normalized float32 embeddings for inner-product similarity.
    """
    prefixed = [f"{_BGE_QUERY_PREFIX}{t}" for t in texts]
    embeddings = model.encode(
        prefixed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def build_indexes(chunks_path: Path = Path("data/processed/chunks.jsonl"), limit: int | None = None) -> None:
    """
    Reads chunks from disk, builds a FAISS IndexFlatIP and a BM25Okapi index,
    and saves both along with chunk metadata to ./indexes/.
    """
    _INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    chunks = _load_chunks(chunks_path)
    if limit:
        chunks = chunks[:limit]

    logger.info("Indexing %d chunks", len(chunks))

    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(_EMBEDDING_MODEL)
    embeddings = _embed(model, texts)

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, str(_FAISS_INDEX_PATH))
    logger.info("FAISS index saved — %d vectors, dim=%d", faiss_index.ntotal, dimension)

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with _BM25_PATH.open("wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 index saved")

    with _METADATA_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    logger.info("Metadata saved -> %s", _METADATA_PATH)

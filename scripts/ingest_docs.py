"""
Builds FAISS and BM25 indexes from the processed chunks.
Usage: python scripts/ingest_docs.py [--limit N]
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from src.indexing.indexer import build_indexes

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max chunks to index (default: all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_PATH}. Run scripts/run_chunking.py first.")
    build_indexes(chunks_path=CHUNKS_PATH, limit=args.limit)
    print("Indexing complete.")


if __name__ == "__main__":
    main()

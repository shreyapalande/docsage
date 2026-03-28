"""
Runs the full ingestion + semantic chunking pipeline and writes chunks to JSONL.
Usage: python scripts/run_chunking.py [--limit N]
"""

import argparse
import itertools
import json
import logging
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from src.ingestion.chunker import SemanticChunker
from src.ingestion.loader import load_documents

OUTPUT_PATH = Path("data/processed/chunks.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    chunker = SemanticChunker()
    documents = load_documents(streaming=True)

    if args.limit:
        documents = itertools.islice(documents, args.limit)

    total_chunks = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk)) + "\n")
            total_chunks += len(chunks)
            logger.info("doc=%s chunks=%d", doc.arxiv_id, len(chunks))

    logger.info("Done. Total chunks written: %d -> %s", total_chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()

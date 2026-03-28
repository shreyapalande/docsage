"""Quick smoke test for the ingestion loader. Run from project root: python scripts/test_ingestion.py"""

import itertools
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from src.ingestion.loader import load_documents


def main():
    print("Loading first 3 documents...\n")
    docs = list(itertools.islice(load_documents(), 3))

    for doc in docs:
        print(f"arxiv_id : {doc.arxiv_id}")
        print(f"title    : {doc.title[:80]}")
        print(f"source   : {doc.source}")
        print(f"content  : {doc.content[:120]}...")
        print("-" * 60)

    print(f"\nLoaded {len(docs)} documents successfully.")


if __name__ == "__main__":
    main()

"""
Prints the full prompt sent to the LLM for a single query so you can inspect context quality.
Usage: python scripts/debug_rag.py
"""

import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

from src.context.injection_guard import filter_chunks
from src.context.packer import pack
from src.retrieval.retriever import HybridRetriever

QUERY = "what is retrieval augmented generation?"


def main() -> None:
    retriever = HybridRetriever()
    chunks = retriever.retrieve(QUERY, top_n=5)
    chunks = filter_chunks(chunks)

    print(f"Chunks retrieved: {len(chunks)}\n")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] doc_id={c.doc_id} chunk_index={c.chunk_index}")
        print(f"       rerank={c.rerank_score:.4f}  rrf={c.rrf_score:.4f}")
        print(f"       title={c.title[:60]}")
        print(f"       content={c.content[:150].replace(chr(10), ' ')}...")
        print()

    context = pack(QUERY, chunks)
    print("=" * 65)
    print("PACKED CONTEXT SENT TO LLM:")
    print("=" * 65)
    print(context)


if __name__ == "__main__":
    main()

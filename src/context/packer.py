"""Assembles retrieved chunks into a token-bounded XML context string."""

import logging

import tiktoken

from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_ENCODING = "cl100k_base"


def _count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _render_doc(index: int, chunk: RetrievedChunk) -> str:
    """Renders a single chunk as an XML <doc> element."""
    relevance = f"{chunk.rerank_score:.3f}" if chunk.rerank_score is not None else "n/a"
    return (
        f'  <doc id="{index}"'
        f' doc_id="{chunk.doc_id}"'
        f' title="{chunk.title}"'
        f' source="{chunk.source}"'
        f' chunk_index="{chunk.chunk_index}"'
        f' relevance="{relevance}">\n'
        f"    {chunk.content}\n"
        f"  </doc>"
    )


def pack(
    query: str,
    chunks: list[RetrievedChunk],
    max_tokens: int = 3000,
) -> str:
    """
    Packs chunks into a token-bounded XML context block.

    Chunks are added in rerank_score order (highest first).
    Stops before adding a chunk that would push the total over max_tokens.
    Token counts use tiktoken cl100k_base to match OpenAI-compatible models.
    """
    enc = tiktoken.get_encoding(_ENCODING)

    wrapper_overhead = _count_tokens("<documents>\n</documents>", enc)
    budget = max_tokens - wrapper_overhead

    doc_blocks: list[str] = []
    used_tokens = 0

    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.rerank_score if c.rerank_score is not None else float("-inf"),
        reverse=True,
    )

    for i, chunk in enumerate(sorted_chunks, start=1):
        block = _render_doc(i, chunk)
        block_tokens = _count_tokens(block, enc)

        if used_tokens + block_tokens > budget:
            logger.debug("Token budget reached at chunk %d — stopping context packing", i)
            break

        doc_blocks.append(block)
        used_tokens += block_tokens

    context = "<documents>\n" + ("\n".join(doc_blocks) + "\n" if doc_blocks else "") + "</documents>"
    logger.debug("Packed %d/%d chunks, ~%d tokens", len(doc_blocks), len(chunks), used_tokens + wrapper_overhead)
    return context

"""End-to-end RAG chain: retrieve → guard → pack → generate → cite."""

import logging

from src.context.citation_builder import extract_citations
from src.context.injection_guard import filter_chunks, guard_query
from src.context.packer import pack
from src.generation.llm_client import LLMClient
from src.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question by synthesizing information from the provided documents.

Rules:
1. Cite every factual claim using [doc_id:chunk_index] format inline, e.g. "BERT uses masked language modeling [2104.08378:3]."
2. Synthesize across documents — if no single document fully answers the question, combine relevant information from multiple documents to form a complete answer.
3. Only say "I don't have enough information in the provided documents to answer this." if the documents contain absolutely no relevant information.
4. Only cite doc IDs that appear in the <documents> block. Never invent or guess doc IDs.
5. Never fabricate information not present in the documents.
6. Be concise and technical."""

_LOW_CONFIDENCE_PREFIX = "⚠️ Low retrieval confidence — treat this answer with caution.\n\n"


class RAGChain:
    """
    Orchestrates the full RAG pipeline:
      query → injection guard → hybrid retrieval → context packing → LLM generation → citation extraction
    """

    def __init__(self, retriever: HybridRetriever, llm: LLMClient) -> None:
        self._retriever = retriever
        self._llm = llm

    def query(self, question: str, doc_filter: str | None = None) -> dict:
        """
        Runs the full RAG pipeline for a question.

        Raises PromptInjectionError if the question is flagged.
        Returns a structured dict with answer, citations, usage stats, and confidence signal.
        """
        guard_query(question)

        chunks = self._retriever.retrieve(question, top_n=5, doc_filter=doc_filter)
        chunks = filter_chunks(chunks)

        rerank_scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
        low_confidence = bool(rerank_scores) and all(s < 0.0 for s in rerank_scores)

        context = pack(question, chunks)
        user_prompt = f"{context}\n\nQuestion: {question}"

        answer, usage = self._llm.complete(system=_SYSTEM_PROMPT, user=user_prompt)

        if low_confidence:
            answer = _LOW_CONFIDENCE_PREFIX + answer

        citations = extract_citations(answer, chunks)

        return {
            "answer":            answer,
            "citations":         citations,
            "chunks_used":       len(chunks),
            "retrieved_chunks":  chunks,
            "rerank_scores":     rerank_scores,
            "usage":             usage,
            "low_confidence":    low_confidence,
        }

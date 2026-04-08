"""RAGAS-style faithfulness and context_recall metrics implemented with Gemini Flash as judge."""

import logging
import time

import google.generativeai as genai
import nltk

from src.config import settings
from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.0-flash"
_CALL_INTERVAL = 1.0


def _get_model() -> genai.GenerativeModel:
    if not settings.gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in .env")
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(_MODEL_NAME)


def _split_sentences(text: str) -> list[str]:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{c.doc_id}:{c.chunk_index}] {c.content}" for c in chunks
    )


def _ask_yes_no(model: genai.GenerativeModel, prompt: str) -> bool:
    """Returns True if Gemini responds 'yes', False otherwise."""
    try:
        response = model.generate_content(prompt)
        return response.text.strip().lower().startswith("yes")
    except Exception as e:
        logger.warning("Gemini call failed: %s", e)
        return False
    finally:
        time.sleep(_CALL_INTERVAL)


def faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float:
    """
    Measures whether every sentence in the answer is supported by the retrieved chunks.
    Score = supported_sentences / total_sentences. Range: 0.0 - 1.0.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    context = _build_context(chunks)
    model = _get_model()
    supported = 0

    for sentence in sentences:
        prompt = (
            f"Given these source documents:\n{context}\n\n"
            f"Is this sentence supported by the documents? Answer only yes or no:\n{sentence}"
        )
        if _ask_yes_no(model, prompt):
            supported += 1

    score = supported / len(sentences)
    logger.info("Faithfulness: %d/%d sentences supported = %.3f", supported, len(sentences), score)
    return score


def context_recall(reference_answer: str, chunks: list[RetrievedChunk]) -> float:
    """
    Measures whether the retrieved chunks contain the information needed to produce the reference answer.
    Score = covered_sentences / total_sentences. Range: 0.0 - 1.0.
    """
    sentences = _split_sentences(reference_answer)
    if not sentences:
        return 0.0

    context = _build_context(chunks)
    model = _get_model()
    covered = 0

    for sentence in sentences:
        prompt = (
            f"Given these retrieved chunks:\n{context}\n\n"
            f"Is this information present in the chunks? Answer only yes or no:\n{sentence}"
        )
        if _ask_yes_no(model, prompt):
            covered += 1

    score = covered / len(sentences)
    logger.info("Context recall: %d/%d sentences covered = %.3f", covered, len(sentences), score)
    return score

"""LLM-as-judge evaluator using Gemini Flash for correctness, groundedness, and citation quality."""

import json
import logging
import time

import google.generativeai as genai

from src.config import settings

logger = logging.getLogger(__name__)

_MODEL_NAME = "gemini-2.0-flash"
_CALL_INTERVAL = 1.0

_RUBRIC_PROMPT = """You are an expert evaluator for RAG systems. Score the answer on three dimensions.

Question: {question}
Reference answer: {reference_answer}
Generated answer: {answer}

Score each dimension 0.0 to 1.0:
1. correctness: Does the answer convey the same facts as the reference?
2. groundedness: Does the answer avoid speculation and stick to stated facts?
3. citation_quality: Does the answer cite sources inline rather than just listing them?

Respond in JSON only:
{{"correctness": 0.0, "groundedness": 0.0, "citation_quality": 0.0, "reasoning": "..."}}"""

_FALLBACK = {"correctness": 0.0, "groundedness": 0.0, "citation_quality": 0.0, "reasoning": "parse_error"}


def _get_model() -> genai.GenerativeModel:
    if not settings.gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in .env")
    genai.configure(api_key=settings.gemini_api_key)
    return genai.GenerativeModel(_MODEL_NAME)


def judge(question: str, answer: str, reference_answer: str) -> dict:
    """
    Scores a generated answer against a reference on correctness, groundedness, and citation quality.
    Uses a single Gemini Flash call with a structured JSON rubric.
    Returns a dict with all four fields; falls back to zeros on parse failure.
    """
    model = _get_model()
    prompt = _RUBRIC_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        answer=answer,
    )

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores = json.loads(raw)
        return {
            "correctness":      float(scores.get("correctness", 0.0)),
            "groundedness":     float(scores.get("groundedness", 0.0)),
            "citation_quality": float(scores.get("citation_quality", 0.0)),
            "reasoning":        str(scores.get("reasoning", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("LLM judge parse failed: %s | raw=%s", e, raw[:200] if "raw" in dir() else "")
        return _FALLBACK.copy()
    except Exception as e:
        logger.error("LLM judge call failed: %s", e)
        return _FALLBACK.copy()
    finally:
        time.sleep(_CALL_INTERVAL)

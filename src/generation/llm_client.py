"""LiteLLM client for Groq llama-3.3-70b-versatile with retry and usage tracking."""

import logging
import os
import time

import litellm

from src.config import settings

logger = logging.getLogger(__name__)

_MODEL = "groq/llama-3.3-70b-versatile"
_MAX_RETRIES = 3
_BACKOFF_SECONDS = [2, 4, 8]
_RATE_LIMIT_BUFFER = 2.0


class LLMClient:
    """
    Wraps LiteLLM completion for the Groq backend.

    Applies a fixed inter-call sleep buffer and exponential backoff on 429s.
    usage_stats always includes estimated_cost_usd for forward-compatibility
    with paid model swaps.
    """

    def __init__(self) -> None:
        os.environ["GROQ_API_KEY"] = settings.groq_api_key

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        """
        Sends a chat completion request and returns (response_text, usage_stats).
        Retries up to 3 times on rate limit errors with exponential backoff.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                time.sleep(_RATE_LIMIT_BUFFER)
                response = litellm.completion(model=_MODEL, messages=messages)

                usage = response.usage
                usage_stats = {
                    "input_tokens":        usage.prompt_tokens,
                    "output_tokens":       usage.completion_tokens,
                    "total_tokens":        usage.total_tokens,
                    "estimated_cost_usd":  0.0,
                }

                return response.choices[0].message.content, usage_stats

            except litellm.RateLimitError as e:
                wait = _BACKOFF_SECONDS[attempt]
                logger.warning("Rate limit hit (attempt %d/%d) — retrying in %ds", attempt + 1, _MAX_RETRIES, wait)
                time.sleep(wait)
                last_error = e

            except Exception as e:
                logger.error("LLM completion failed: %s", e)
                raise

        raise RuntimeError(f"LLM request failed after {_MAX_RETRIES} retries") from last_error

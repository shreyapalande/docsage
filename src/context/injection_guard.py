"""Prompt injection detection for user queries and retrieved chunk content."""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PATH = Path("logs/injection_attempts.log")

_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(previous|above|all)\s+instructions?", "ignore_instructions"),
    (r"you\s+are\s+now\s+(a|an)\b",                   "persona_override"),
    (r"system\s*:",                                     "system_prompt_injection"),
    (r"<\s*instructions?\s*>",                          "xml_instruction_tag"),
    (r"forget\s+(everything|all|your)",                 "forget_instructions"),
    (r"new\s+persona",                                  "new_persona"),
    (r"jailbreak",                                      "jailbreak"),
    (r"disregard\s+(your|all|previous)",                "disregard_instructions"),
]

_COMPILED = [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in _PATTERNS]


class PromptInjectionError(ValueError):
    """Raised when a user query contains a detected injection attempt."""


def _log_attempt(threat_type: str, text: str) -> None:
    """Appends a timestamped injection attempt record to the log file."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = f"[{timestamp}] threat={threat_type} preview={repr(text[:100])}\n"
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(entry)
    logger.warning("Injection attempt detected: type=%s", threat_type)


def scan(text: str) -> tuple[bool, str | None]:
    """
    Scans text for prompt injection patterns.

    Returns (True, None) if the text is safe.
    Returns (False, threat_type) on the first match found, and logs the attempt.
    """
    for pattern, label in _COMPILED:
        if pattern.search(text):
            _log_attempt(label, text)
            return False, label
    return True, None


def guard_query(query: str) -> None:
    """Raises PromptInjectionError if the user query contains an injection attempt."""
    is_safe, threat_type = scan(query)
    if not is_safe:
        raise PromptInjectionError(
            f"Query blocked due to detected injection pattern: {threat_type}"
        )


def filter_chunks(chunks: list) -> list:
    """
    Removes any chunks whose content triggers an injection pattern.
    Flagged chunks are silently excluded and logged.
    """
    safe = []
    for chunk in chunks:
        is_safe, _ = scan(chunk.content)
        if is_safe:
            safe.append(chunk)
    return safe

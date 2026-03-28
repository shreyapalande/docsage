"""Loads the jamescalam/ai-arxiv dataset from HuggingFace and yields normalized Document objects."""

import logging
from collections.abc import Iterator
from typing import Literal

from datasets import load_dataset
from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

DATASET_NAME = "jamescalam/ai-arxiv"
REQUIRED_FIELDS = {"id", "title", "summary", "source"}


class Document(BaseModel):
    """Canonical document schema used throughout the pipeline."""

    arxiv_id: str
    title: str
    source: str
    content: str = Field(..., min_length=1)


def _validate_schema(column_names: list[str]) -> None:
    """Raises ValueError if the dataset is missing any required fields."""
    missing = REQUIRED_FIELDS - set(column_names)
    if missing:
        raise ValueError(f"Dataset missing expected fields: {missing}")


def load_documents(
    split: Literal["train", "test", "validation"] = "train",
    streaming: bool = True,
) -> Iterator[Document]:
    """
    Streams documents from the ai-arxiv HuggingFace dataset.

    Uses streaming by default to avoid loading the full dataset into memory.
    Dataset fields are mapped: id -> arxiv_id, summary -> content.
    """
    logger.info("Loading dataset '%s' (split=%s, streaming=%s)", DATASET_NAME, split, streaming)

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=streaming,
        token=settings.hf_token,
    )

    _validate_schema(dataset.column_names)

    for row in dataset:
        try:
            yield Document(
                arxiv_id=row["id"],
                title=row["title"],
                source=row["source"],
                content=row["summary"],
            )
        except Exception:
            logger.warning("Skipping malformed row: id=%s", row.get("id", "unknown"))
            continue

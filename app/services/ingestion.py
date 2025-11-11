"""Document ingestion and chunking utilities."""

from __future__ import annotations

import os
import re
from typing import Iterable, List, Tuple


def clean_text(text: str) -> str:
    """Basic normalisation for input documents.

    Strips leading/trailing whitespace and collapses multiple spaces and newlines.
    """
    # Replace Windows line endings and collapse multiple spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, max_tokens: int) -> List[str]:
    """Split a string into chunks of roughly `max_tokens` words.

    Args:
        text: The full document text.
        max_tokens: Desired number of whitespace‑delimited tokens per chunk.

    Returns:
        A list of chunk strings.  The final chunk may be shorter.
    """
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i : i + max_tokens])
        chunks.append(chunk)
    return chunks
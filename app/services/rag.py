"""Answer composition for retrieval augmented QA."""

from __future__ import annotations

from typing import Callable, List, Tuple


def compose_answer(
    query: str,
    retrieved: List[Tuple[int, float]],
    fetch_chunks: Callable[[List[int]], List[Tuple[int, int, int, str]]],
) -> dict:
    """Compose an answer from retrieved chunk IDs.

    Args:
        query: The user’s question. (Currently unused but left for future heuristics.)
        retrieved: A list of `(chunk_id, score)` tuples sorted by score descending.
        fetch_chunks: Function to fetch chunk rows given IDs.

    Returns:
        A dictionary with an `answer` string and a list of `contexts` where each
        context contains the chunk ID, parent document ID, chunk index, text and
        score.
    """
    if not retrieved:
        return {"answer": "No relevant information found.", "contexts": []}
    chunk_ids = [cid for cid, _ in retrieved]
    rows = fetch_chunks(chunk_ids)
    # Map id to row for quick lookup
    row_map = {row[0]: row for row in rows}
    contexts = []
    answer = ""
    for cid, score in retrieved:
        row = row_map.get(cid)
        if not row:
            continue
        _, doc_id, chunk_idx, text = row
        contexts.append(
            {
                "chunk_id": cid,
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "text": text,
                "score": score,
            }
        )
    # Choose the highest scoring context as the answer
    answer = contexts[0]["text"] if contexts else ""
    return {"answer": answer, "contexts": contexts}
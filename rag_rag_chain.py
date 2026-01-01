from typing import List, Dict, Tuple, Optional

from rag_llm import chat_completion
from rag_vectorstore import query_similar

# Tune this:
# Lower distance = more similar. Typical good matches are often < 1.0 depending on embeddings.
# Start with 1.0 then tune (0.8 stricter, 1.2 looser).
RAG_DISTANCE_THRESHOLD = 1.0

SYSTEM_RAG = """You are a helpful assistant.
Answer using ONLY the provided context. If the context does not contain the answer, say:
"I don't know based on the provided documents."
"""

SYSTEM_GENERAL = """You are a helpful assistant. Answer normally and concisely."""

def _format_context(contexts: List[Dict]) -> str:
    # contexts items expected like: {"text": "...", "meta": {...}, "distance": 0.42}
    lines = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("meta", {}) or {}
        src = meta.get("source", "document")
        chunk_id = meta.get("chunk_id", i)
        lines.append(f"[{i}] source={src} chunk={chunk_id}\n{c['text']}")
    return "\n\n".join(lines)

def rag_answer(question: str, k: int = 4) -> Tuple[str, List[Dict]]:
    """
    Hybrid behavior:
    - If top retrieval is relevant => RAG answer + return contexts for Sources.
    - Else => General LLM answer + return [] (no Sources).
    """
    contexts = query_similar(question, k=k)

    # If nothing retrieved, go general
    if not contexts:
        answer = chat_completion(
            system=SYSTEM_GENERAL,
            user=question,
        )
        return answer, []

    # Determine best distance (smaller is better)
    best_distance = None
    for c in contexts:
        d = c.get("distance", None)
        if d is not None:
            best_distance = d if best_distance is None else min(best_distance, d)

    # If we can't read distances, fall back to RAG (or you can go general)
    if best_distance is None:
        context_text = _format_context(contexts)
        answer = chat_completion(
            system=SYSTEM_RAG,
            user=f"Question:\n{question}\n\nContext:\n{context_text}",
        )
        # If the model says "I don't know..." then return general instead (optional)
        if answer.strip().lower().startswith("i don't know"):
            answer2 = chat_completion(system=SYSTEM_GENERAL, user=question)
            return answer2, []
        return answer, contexts

    # If distance is too large => not relevant => general answer and no sources
    if best_distance > RAG_DISTANCE_THRESHOLD:
        answer = chat_completion(system=SYSTEM_GENERAL, user=question)
        return answer, []

    # Otherwise use RAG
    context_text = _format_context(contexts)
    answer = chat_completion(
        system=SYSTEM_RAG,
        user=f"Question:\n{question}\n\nContext:\n{context_text}",
    )

    # Optional extra safety: if RAG says "I don't know", switch to general
    if answer.strip().lower().startswith("i don't know"):
        answer2 = chat_completion(system=SYSTEM_GENERAL, user=question)
        return answer2, []

    return answer, contexts

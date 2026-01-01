from typing import List, Dict, Tuple

from rag_llm import chat_completion
from rag_vectorstore import query_similar

RAG_DISTANCE_THRESHOLD = 1.3

SYSTEM_RAG = """You are a document-grounded assistant.

RULES:
- Use ONLY the provided context.
- If the answer is not explicitly in the context, reply exactly:
  I don't know based on the provided documents.
- Do NOT add outside knowledge.
- Keep the answer short and factual.
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
    contexts = query_similar(question, k=k)

    if not contexts:
        answer = chat_completion(SYSTEM_GENERAL, question)
        return answer, []

    best_distance = None
    for c in contexts:
        d = c.get("distance", None)
        if d is not None:
            best_distance = d if best_distance is None else min(best_distance, d)

    if best_distance is None:
        context_text = _format_context(contexts)
        answer = chat_completion(SYSTEM_RAG, f"Question:\n{question}\n\nContext:\n{context_text}")
        if answer.strip().lower().startswith("i don't know"):
            answer2 = chat_completion(SYSTEM_GENERAL, question)
            return answer2, []
        return answer, contexts

    if best_distance > RAG_DISTANCE_THRESHOLD:
        answer = chat_completion(SYSTEM_GENERAL, question)
        return answer, []

    context_text = _format_context(contexts)
    answer = chat_completion(SYSTEM_RAG, f"Question:\n{question}\n\nContext:\n{context_text}")

    if answer.strip().lower().startswith("i don't know"):
        answer2 = chat_completion(SYSTEM_GENERAL, question)
        return answer2, []

    return answer, contexts

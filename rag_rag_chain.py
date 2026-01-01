from typing import List, Dict, Tuple

from rag_llm import chat_completion
from rag_vectorstore import query_similar

# Tune this:
# Lower distance = more similar (cosine distance). For MiniLM embeddings,
# ~1.0–1.4 is a reasonable range to start, then tune based on your document.
RAG_DISTANCE_THRESHOLD = 1.3

SYSTEM_RAG = """You are a document-grounded assistant.

You MUST follow these rules:
1) Use ONLY the provided context. Do not use outside knowledge.
2) If the answer is not explicitly stated in the context, reply exactly:
I don't know based on the provided documents.
3) Answer in 2-5 short sentences maximum.
4) Do not guess or infer anything that is not written in the context.
"""

SYSTEM_GENERAL = """You are a helpful assistant. Answer normally and concisely."""


def _format_context(contexts: List[Dict]) -> str:
    """
    contexts expected like:
      {"text": "...", "meta": {...}, "distance": 0.42}
    """
    lines: List[str] = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("meta", {}) or {}
        src = meta.get("source", "document")
        chunk_id = meta.get("chunk_id", i)
        lines.append(f"[{i}] source={src} chunk={chunk_id}\n{c.get('text','')}")
    return "\n\n".join(lines)


def rag_answer(question: str, k: int = 4) -> Tuple[str, List[Dict]]:
    """
    Hybrid behavior:
    - If retrieval is relevant => RAG answer + return contexts for Sources.
    - Else => General LLM answer + return [] (no Sources).
    """
    contexts = query_similar(question, k=k)

    # If nothing retrieved, go general
    if not contexts:
        answer = chat_completion(SYSTEM_GENERAL, question)
        return answer, []

    # Compute best (minimum) distance
    best_distance = None
    for c in contexts:
        d = c.get("distance", None)
        if d is not None:
            best_distance = d if best_distance is None else min(best_distance, d)

    # If we can't read distances, do RAG but still enforce refusal
    if best_distance is None:
        # keep only top contexts to reduce noise
        contexts = contexts[:3]
        for c in contexts:
            c["text"] = (c.get("text") or "")[:1200]

        context_text = _format_context(contexts)
        answer = chat_completion(
            SYSTEM_RAG,
            f"Question:\n{question}\n\nContext:\n{context_text}",
        )

        # If doc doesn't contain answer, switch to general (and hide sources)
        if answer.strip().lower().startswith("i don't know"):
            answer2 = chat_completion(SYSTEM_GENERAL, question)
            return answer2, []

        return answer, contexts

    # If not relevant, go general (no sources)
    if best_distance > RAG_DISTANCE_THRESHOLD:
        answer = chat_completion(SYSTEM_GENERAL, question)
        return answer, []

    # Relevant => use RAG
    contexts = contexts[:3]
    for c in contexts:
        c["text"] = (c.get("text") or "")[:1200]

    context_text = _format_context(contexts)
    answer = chat_completion(
        SYSTEM_RAG,
        f"Question:\n{question}\n\nContext:\n{context_text}",
    )

    # If RAG refuses, go general (no sources)
    if answer.strip().lower().startswith("i don't know"):
        answer2 = chat_completion(SYSTEM_GENERAL, question)
        return answer2, []

    return answer, contexts

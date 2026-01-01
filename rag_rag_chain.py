from typing import Dict, List, Tuple

from rag_config import TOP_K
from rag_llm import generate_answer
from rag_vectorstore import query_chunks


SYSTEM_PROMPT = """You are a company document assistant.

RULES:
- You MUST answer using ONLY the provided CONTEXT from company documents.
- Do NOT use outside knowledge.
- Do NOT browse the internet.
- If the answer is not found in the CONTEXT, reply exactly:
"I don't know based on the provided company documents."
- Add citations like [1], [2] referencing the context items.

Be concise and accurate.
"""


def build_prompt(question: str, contexts: List[Dict]) -> str:
    ctx = ""
    for i, c in enumerate(contexts, start=1):
        meta = c.get("meta", {})
        src = meta.get("source", "unknown")
        page = meta.get("page", None)
        chunk_id = meta.get("chunk_id", None)

        label = f"[{i}] source={src}"
        if page:
            label += f", page={page}"
        if chunk_id:
            label += f", chunk={chunk_id}"

        ctx += f"{label}\n{c['text']}\n\n"

    return f"""{SYSTEM_PROMPT}

CONTEXT:
{ctx}

QUESTION:
{question}

Answer:
"""


def rag_answer(question: str, k: int = TOP_K) -> Tuple[str, List[Dict]]:
    contexts = query_chunks(question, k=k)
    prompt = build_prompt(question, contexts)
    answer = generate_answer(prompt)
    return answer, contexts
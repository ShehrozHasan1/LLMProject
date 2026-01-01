import requests
from rag_config import PERPLEXITY_API_KEY, PERPLEXITY_MODEL


def generate_answer(prompt: str) -> str:
    """
    Calls Perplexity API using the chat completions endpoint.
    """
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY is missing. Add it to .env")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system", "content": "Follow the user's instructions carefully."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    return (data["choices"][0]["message"]["content"] or "").strip()


def chat_completion(system: str, user: str) -> str:
    """
    Small wrapper used by rag_rag_chain.py.
    We combine system + user into one prompt to keep your current API style.
    """
    prompt = f"System:\n{system}\n\nUser:\n{user}"
    return generate_answer(prompt)

def chat_completion(system: str, user: str) -> str:
    """
    Wrapper used by rag_rag_chain.py.
    Sends a normal chat-style request with system + user messages.
    """
    if not PERPLEXITY_API_KEY:
        raise ValueError("PERPLEXITY_API_KEY is missing. Add it to Streamlit Secrets.")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": PERPLEXITY_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()

# DEBUG: prove what functions exist
__ALL_FUNCS__ = [name for name in globals().keys() if "chat" in name or "generate" in name]

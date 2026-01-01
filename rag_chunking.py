from typing import Dict, List, Tuple


def chunk_text(
    text: str,
    base_metadata: Dict,
    chunk_size: int,
    overlap: int
) -> Tuple[List[str], List[Dict]]:
    """
    Simple character-based chunking.
    This is easy to understand and works for a demo project.
    """
    text = (text or "").strip()
    if not text:
        return [], []

    chunks: List[str] = []
    metas: List[Dict] = []

    start = 0
    chunk_id = 1

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
            md = dict(base_metadata)
            md["chunk_id"] = chunk_id
            metas.append(md)
            chunk_id += 1

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks, metas
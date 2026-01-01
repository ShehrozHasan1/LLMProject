import os
from typing import Dict, List, Tuple

from pypdf import PdfReader
from docx import Document


def load_pdf(path: str) -> Tuple[str, List[Dict]]:
    reader = PdfReader(path)
    parts: List[str] = []
    metas: List[Dict] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        parts.append(text)
        metas.append({"source": os.path.basename(path), "page": i + 1})

    return "\n\n".join(parts), metas


def load_docx(path: str) -> Tuple[str, List[Dict]]:
    doc = Document(path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n".join(lines)
    metas = [{"source": os.path.basename(path), "page": None}]
    return text, metas


def load_document(path: str) -> Tuple[str, List[Dict]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)

    raise ValueError(f"Unsupported file type: {ext}. Please upload PDF or DOCX.")
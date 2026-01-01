import os
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from rag_config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
)


def get_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: List[str], metadatas: List[Dict], ids: List[str]) -> None:
    col = get_collection()
    col.upsert(documents=chunks, metadatas=metadatas, ids=ids)


def query_chunks(query: str, k: int = 5) -> List[Dict]:
    col = get_collection()
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    out: List[Dict] = []
    for d, m, dist in zip(docs, metas, dists):
        out.append({"text": d, "meta": m, "distance": dist})
    return out
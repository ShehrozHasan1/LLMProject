import os
import uuid
import streamlit as st

from rag_config import UPLOADS_DIR, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, PROJECT_DIR
from rag_document_loaders import load_document
from rag_chunking import chunk_text
from rag_vectorstore import upsert_chunks, get_doc_count
from rag_rag_chain import rag_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.title("Chatbot by Muhammad Shehroz.")

# -----------------------------
# AUTO-INGEST (from repo PDF)
# -----------------------------
DEFAULT_PDF_NAME = "Company-Profile.pdf"
DEFAULT_PDF_PATH = os.path.join(os.path.dirname(__file__), DEFAULT_PDF_NAME)

def ingest_file(path: str, display_name: str):
    text, _ = load_document(path)
    if not text.strip():
        st.warning(f"No text extracted from: {display_name}")
        return 0

    base_meta = {"source": display_name, "page": ""}
    chunks, metas = chunk_text(
        text=text,
        base_metadata=base_meta,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    ids = [f"{display_name}-{i+1}" for i in range(len(chunks))]
    upsert_chunks(chunks, metas, ids)
    return len(chunks)

# Run once per session (not every rerun)
if "auto_ingested" not in st.session_state:
    st.session_state.auto_ingested = False

if not st.session_state.auto_ingested:
    existing = get_doc_count()
    if existing == 0 and os.path.exists(DEFAULT_PDF_PATH):
        with st.spinner(f"Auto-ingesting {DEFAULT_PDF_NAME} (first time setup)..."):
            stored = ingest_file(DEFAULT_PDF_PATH, DEFAULT_PDF_NAME)
        st.session_state.auto_ingested = True
        st.success(f"Auto-ingest complete. Stored {stored} chunks.")
    else:
        st.session_state.auto_ingested = True

with st.sidebar:
    st.header("Upload file/documents (optional)")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    st.caption("Click ingest to build/update (optional).")

    if st.button("Ingest uploaded documents"):
        if not uploaded_files:
            st.warning("Upload at least one PDF/DOCX file first.")
        else:
            with st.spinner("Ingesting..."):
                total_chunks = 0

                for uf in uploaded_files:
                    file_id = str(uuid.uuid4())
                    save_path = os.path.join(UPLOADS_DIR, f"{file_id}_{uf.name}")
                    with open(save_path, "wb") as f:
                        f.write(uf.getbuffer())

                    total_chunks += ingest_file(save_path, uf.name)

                st.success(f"Ingestion completed. Stored {total_chunks} chunks in ChromaDB.")

st.subheader("Ask Me Anything 😉")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask your querry here...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, contexts = rag_answer(question)
            st.markdown(answer if answer else "(No response)")

            if contexts:
                with st.expander("Sources (retrieved chunks)"):
                    for i, c in enumerate(contexts, start=1):
                        meta = c.get("meta", {})
                        st.write(
                            f"**[{i}]** source={meta.get('source')} "
                            f"page={meta.get('page')} chunk={meta.get('chunk_id')} "
                            f"distance={c.get('distance')}"
                        )
                        st.code(c["text"][:1500])

    st.session_state.messages.append({"role": "assistant", "content": answer})

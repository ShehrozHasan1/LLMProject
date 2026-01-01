# Company RAG Chatbot (Perplexity API + Streamlit)

A **RAG-based LLM chatbot**:
- Upload **PDF/DOCX** company documents
- Store them in a local vector DB (**ChromaDB**)
- Answer questions using **retrieval + Perplexity LLM generation**
- Runs on localhost via Streamlit

## 1) Setup

### A) Clone
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

### B) Create virtual environment
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### C) Install dependencies
```bash
pip install -r requirements.txt
```

### D) Add Perplexity API key
Copy the example:
```bash
# macOS/Linux
cp .env.example .env
```

Windows PowerShell:
```powershell
copy .env.example .env
```

Edit `.env` and set:
- `PERPLEXITY_API_KEY=...`
- `PERPLEXITY_MODEL=sonar-pro`

## 2) Run
```bash
streamlit run app.py
```

Open:
- http://localhost:8501

## 3) How the RAG pipeline works
1. Upload PDF/DOCX
2. Extract text from document
3. Split into chunks
4. Convert chunks to embeddings (SentenceTransformers)
5. Store embeddings in ChromaDB
6. When user asks a question:
   - retrieve top matching chunks from ChromaDB
   - send question + chunks to Perplexity model
   - display answer + sources

## Notes
- Answers are forced to use ONLY the provided context.
- If scanned PDFs have no text, you will need OCR (not included).
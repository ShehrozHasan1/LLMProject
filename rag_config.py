import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_DIR, "data")
UPLOADS_DIR = os.path.join(PROJECT_DIR, "uploads")

# Vector DB (Chroma)
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
CHROMA_COLLECTION = "company_docs"

# Embeddings model (runs locally; free)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval
TOP_K = 5

# Perplexity
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache HuggingFace models for instant backend startup
RUN python -c "\
from llama_index.embeddings.huggingface import HuggingFaceEmbedding; \
HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-reranker-base')\
"

# Copy parent files needed for backend
COPY config.py utils.py 3_query_graph.py .env ./

# Copy backend files
COPY backend/ ./backend/

# Set Python path to look in /app for root module imports
ENV PYTHONPATH=/app

EXPOSE 5005

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "5005"]

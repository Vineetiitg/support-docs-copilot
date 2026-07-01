FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip && pip install --default-timeout=1000 -r requirements.txt

# Copy entire project
COPY . .

# Pre-download embedding model weights during image build for instant cloud startup
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')"

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Launch FastAPI backend on port 8000 in background, wait 5 seconds, then start Streamlit UI on port 7860
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 & sleep 5 && BACKEND_BASE_URL=http://localhost:8000 streamlit run ui/app.py --server.port 7860 --server.address 0.0.0.0"

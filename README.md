# Support Docs Copilot

Python-only advanced RAG support copilot using Ollama, Qdrant hybrid retrieval, local reranking, LangGraph Self-RAG, Guardrails AI, Ragas evaluation, FastAPI, and Streamlit.

## Quick Start

1. Start Ollama and pull the model:

```bash
ollama run llama3
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Ingest the sample docs:

```bash
python app/engine/ingestion.py
```

4. Start the backend:

```bash
uvicorn app.main:app --reload
```

5. Start the UI in another terminal:

```bash
streamlit run ui/app.py
```

Backend docs run at `http://127.0.0.1:8000/docs`; the Streamlit app runs at `http://localhost:8501`.

## Docker

```bash
docker-compose up --build -d
docker exec -it $(docker-compose ps -q ollama) ollama run llama3
docker exec -it $(docker-compose ps -q backend) python app/engine/ingestion.py
```

## Suggested Commit Roadmap

1. `init: setup fastapi boilerplate and environment config for ollama and qdrant`
2. `feat: implement hybrid search ingestion pipeline with qdrant and fastembed`
3. `feat: integrate cross-encoder reranking for context refinement`
4. `feat: build self-rag decision graph with evaluation nodes`
5. `feat: add input validation and output verification guardrails`
6. `test: implement automated ragas evaluation pipeline`
7. `feat: complete streamlit chat interface and integrate backend streaming api`
8. `deploy: containerize complete architecture with docker compose for production`


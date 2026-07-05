# Support Docs Copilot - REST Endpoints Guide

This guide describes the main REST endpoints in the backend application of the Support Docs Copilot.

## 1. Chat & Query Endpoints

### `POST /chat`
The primary conversational RAG endpoint. It accepts a user query and conversation history, performs query expansion, retrieves relevant chunks from Qdrant, reranks them using Cohere or FlashRank, and generates a grounded response using the LLM with strict source citations.

### `POST /chat/stream`
A streaming variant of the chat endpoint that streams tokens back to the client in real time using Server-Sent Events (SSE). It performs full RAG pipeline evaluation, including semantic cache checks, relevance grading, and guardrail interception.

### `POST /query`
A direct retrieval and generation endpoint designed for single-turn queries without conversation history.

## 2. Ingestion & Document Management Endpoints

### `POST /ingest`
Triggers asynchronous document ingestion. This endpoint scans the `data/docs` directory (or uploaded files), chunks documents, computes embeddings using FastEmbed (`BAAI/bge-small-en-v1.5`), and indexes them into the Qdrant vector database (`support_docs` collection). It dispatches an asynchronous background task to the Redis worker.

### `DELETE /ingest/files`
Allows users or administrators to delete files and remove their corresponding embeddings from Qdrant in a single click.

## 3. Feedback & Observability Endpoints

### `POST /feedback`
Records user thumbs-up or thumbs-down feedback for generated responses along with comment logs for evaluation and continuous improvement.

### `GET /health`
Returns the health status of the API server, Redis connection, and Qdrant vector store.

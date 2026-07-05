# Redis Worker & Asynchronous Background Tasks Architecture

This document explains how the Redis worker handles asynchronous background tasks and document indexing in the Support Docs Copilot architecture.

## 1. Asynchronous Background Task Queue
The Support Docs Copilot uses **ARQ (Async Redis Queue)** paired with a Redis instance to manage heavy background workloads without blocking the asynchronous FastAPI event loop. When long-running tasks such as document ingestion or batch evaluation are initiated via the API, the server pushes a task job to Redis and immediately returns a job ID to the client.

## 2. Document Indexing & Ingestion Workflow
When an ingestion request (`POST /ingest`) is received:
1. **Task Dispatch**: The API server enqueues an `ingest_documents_task` into Redis.
2. **Worker Execution**: The standalone ARQ worker process picks up the task from Redis asynchronously.
3. **Chunking & Embedding**: The worker loads files from `data/docs`, applies recursive character chunking with overlap, and generates dense vector embeddings using local ONNX models (`FastEmbedEmbeddings` with `BAAI/bge-small-en-v1.5`) as well as sparse embeddings (`Qdrant/bm25`) for hybrid retrieval.
4. **Vector Storage**: The processed points and metadata are batched and upserted into the Qdrant vector database under the `support_docs` collection.

## 3. Reliability & Scalability
By offloading document indexing and semantic cache pre-warming to the Redis worker queue, the backend ensures zero degradation in chat latency for active user sessions. The queue supports automatic retries, job status polling, and concurrency control.

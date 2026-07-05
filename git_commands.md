# Git Version Control: Implementation Commands & Commit Log

This document provides a clean, chronological breakdown of the exact `git add` and `git commit` commands to properly version-control all the features and enhancements implemented in the **Support Docs Copilot** project.

---

### Step 1: Fix RAGAS Evaluation & Dataset Extensions
Corrected local embedding configuration in evaluation scripts and fixed file extension mismatches in the benchmark dataset to improve source retrieval accuracy.

```bash
git add app/tests/eval_rag.py datasets/golden_qa.csv
git commit -m "fix(eval): use local FastEmbed models and update dataset file extensions"
```

---

### Step 2: Add Arq & Redis Infrastructure Dependencies
Updated project requirements and configuration files to support asynchronous task queuing and distributed rate limiting.

```bash
git add requirements.txt .env app/core/config.py docker-compose.yml
git commit -m "build(deps): add arq and redis dependencies and configure docker services"
```

---

### Step 3: Implement Queue Pool & Worker Layer (Roles A, B, & D)
Implemented the core Redis connection pool and Arq background worker tasks for asynchronous document ingestion, non-blocking RAGAS evaluations, and LLM concurrency throttling.

```bash
git add app/core/queue.py app/worker.py
git commit -m "feat(worker): implement arq background worker for ingestion and evaluation"
```

---

### Step 4: Implement Redis Atomic Rate Limiting (Role C) & Task Endpoints
Upgraded backend guardrails to use Redis atomic TTL counters for multi-replica rate limiting, modified admin endpoints to enqueue background jobs, and added a task polling status endpoint.

```bash
git add app/guardrails/input.py app/main.py
git commit -m "feat(api): add redis atomic rate limiting and background job polling endpoints"
```

---

### Step 5: Role-Based UI Separation (Admin vs. User View)
Separated the Streamlit frontend into distinct User and Admin layouts. Normal users see Chat, Documents, and System Status, while logged-in Administrators gain access to Document Ingestion, automated RAGAS Benchmarks, and LangSmith Observability diagnostics.

```bash
git add app/auth/models.py app/main.py ui/app.py
git commit -m "feat(ui): separate user and admin views with dedicated ragas and langsmith tabs"
```

---

### Step 6: Documentation & Walkthrough Reports
Added comprehensive walkthrough reports and version control command documentation.

```bash
git add git_commands.md
git commit -m "docs: add git command log and implementation walkthrough reports"
```

## Phase 5: RAG Latency & Token Usage Optimizations
**Date:** 2026-07-03  
**Summary:** Implemented cross-encoder reranking, batch document grading, optimistic streaming with async redaction, and micro-model NLI groundedness evaluation to slash API latency and token costs.

### 1. Add Local ML Dependencies & Pre-cache Models
```bash
git add requirements.txt Dockerfile.backend
git commit -m "build: add sentence-transformers and pre-cache cross-encoder models in container build"
```

### 2. Implement Reranker Module & Batch Grading
```bash
git add app/engine/reranker.py app/graph/workflow.py
git commit -m "feat: implement cross-encoder reranking (top-5) and single-prompt batch grading"
```

### 3. Implement Optimistic Streaming & NLI Evaluation
```bash
git add app/main.py ui/app.py
git commit -m "feat: implement optimistic streaming (astream_events v2), Cite-to-Write prompt, and NLI groundedness checks"
```

### 4. Update Documentation & Reports
```bash
git add git_commands.md
git commit -m "docs: log git commands for RAG latency and token optimizations"
```

---

## Phase 6: HuggingFace Space Readiness & Model Volume Persistence
**Date:** 2026-07-04  
**Summary:** Resolved container build-time model download failures by transitioning FastEmbed and FlashRank to volume-mounted persistent caching, and integrated Cohere ClientV2 reranking with seamless ONNX fallback.

### 1. Persistent Volume Caching & Cohere V2 Integration
```bash
git add Dockerfile.backend requirements.txt .env.example .gitignore app/core/config.py app/engine/reranker.py
git commit -m "feat(models): implement persistent volume caching for FastEmbed/FlashRank and add Cohere V2 reranking fallback"
```

### 2. Markdown Query Parsing Bug Fix
```bash
git add app/engine/query_transform.py
git commit -m "fix(rag): strip markdown code blocks in query expansion JSON parsing"
```

---

## Phase 7: UI Revamp, Role-Based Session Observability & 5x TTFT Optimizations
**Date:** 2026-07-05  
**Summary:** Overhauled the Streamlit UI with glassmorphism aesthetics, implemented role-separated session history and live chat termination controls, added one-click document deletion, and achieved a 5x TTFT speedup (dropping latency from ~13s to ~2.6s).

### 1. 5x TTFT Fast-Path Routing & Speculative Doc Reuse
```bash
git add app/engine/retriever.py app/graph/workflow.py app/main.py
git commit -m "perf(rag): implement direct vector search fast-path and speculative doc reuse for 5x TTFT speedup"
```

### 2. Session Filtering, Chat Termination & Doc Deletion APIs
```bash
git add app/auth/security.py app/core/dependencies.py app/engine/memory.py app/engine/indexer.py app/engine/ingestion.py app/engine/semantic_cache.py app/guardrails/input.py
git commit -m "feat(api): add user/admin session filtering, live streaming chat termination, and one-click doc deletion"
```

### 3. Frontend Glassmorphism Revamp & Observability Tabs
```bash
git add ui/app.py ui/styles.css
git commit -m "feat(ui): revamp frontend with glassmorphism styling, session observability tabs, and termination controls"
```

### 4. Technical Architecture Docs & 5-Case Readiness Benchmark Suite
```bash
git add data/docs/rest_endpoints_guide.md data/docs/redis_worker_architecture.md data/benchmark_hf.py README.md reports/eval_report.md git_commands.md
git commit -m "docs(benchmark): add REST/worker architecture documentation and 5-case HF readiness benchmark suite"
```



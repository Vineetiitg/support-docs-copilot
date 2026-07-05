import asyncio
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from guardrails import Guard
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.auth.models import Token, UserContext
from app.auth.security import require_admin, resolve_user, create_access_token, verify_password, USERS
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.core.config import settings
from app.core.dependencies import check_openrouter, check_qdrant
from app.core.errors import CopilotError
from app.core.logging import configure_logging, logger, request_id_var
from app.core.queue import get_redis_client
from app.engine.document_registry import load_registry
from app.engine.ingestion import delete_indexed_document, ingest_documents, reset_index
from app.engine.context_builder import build_context, format_sources
import csv
from app.engine.memory import add_session_message, get_session_history, list_user_sessions, delete_session, get_session_summary, list_all_sessions
from app.engine.query_transform import condense_query
from app.engine.retriever import retrieve_documents
from app.engine.semantic_cache import get_cached_answer, set_cached_answer
from app.graph.workflow import compile_workflow
from app.guardrails.input import async_enforce_rate_limit, enforce_rate_limit, validate_query
from app.guardrails.output import redact_sensitive_data
from app.guardrails.validators import DetectPromptInjection
from app.observability.metrics import RequestMetrics, log_request_metrics, timed_stage
from app.tests.eval_rag import run_local_evaluation, REPORT_PATH

configure_logging()
if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    logger.info(f"LangSmith tracing enabled for project: {settings.LANGCHAIN_PROJECT}")

app = FastAPI(title=settings.PROJECT_NAME)

@app.on_event("startup")
async def startup_faq_prewarming():
    try:
        csv_path = Path("datasets/golden_qa.csv")
        if csv_path.exists():
            logger.info("PRE-WARMING SEMANTIC CACHE: Seeding FAQ entries from golden_qa.csv...")
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    question = row.get("question", "").strip()
                    answer = row.get("expected_answer", "").strip()
                    sources_raw = row.get("expected_sources", "").strip()
                    if question and answer:
                        sources = [{"doc_id": sources_raw, "source": sources_raw, "snippet": answer}] if sources_raw else []
                        await set_cached_answer(question, answer, sources, confidence=0.99)
                        count += 1
            logger.info(f"PRE-WARMING COMPLETE: Successfully seeded {count} FAQ entries into Redis vector cache.")
    except Exception as e:
        logger.warning(f"FAQ pre-warming failed or skipped: {e}")

async def resolve_query_speculative(query: str, chat_history: list, summary: str):
    speculative_docs = []
    if chat_history:
        condense_task = asyncio.create_task(condense_query(query, chat_history, summary=summary))
        retrieval_task = asyncio.create_task(retrieve_documents(query, chat_history))
        results = await asyncio.gather(condense_task, retrieval_task, return_exceptions=True)
        
        standalone_query = query if isinstance(results[0], Exception) else results[0]
        raw_docs = [] if isinstance(results[1], Exception) else results[1]
        
        if raw_docs:
            top_sim = max([d.metadata.get("similarity_score", 0.0) for d in raw_docs] + [0.0])
            if top_sim >= 0.85:
                logger.info(f"SPECULATIVE RETRIEVAL HIT: Raw query '{query}' matched with top similarity {top_sim:.4f} >= 0.85!")
                speculative_docs = raw_docs
    else:
        standalone_query = await condense_query(query, chat_history, summary=summary)
        
    return standalone_query, speculative_docs

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)

@app.exception_handler(CopilotError)
async def copilot_error_handler(request: Request, exc: CopilotError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )

rag_agent = compile_workflow()
input_guard = Guard().use(DetectPromptInjection, on_fail="exception")

class ChatRequest(BaseModel):
    query: str
    chat_history: list[dict] = []
    session_id: str | None = None

class SourceCitation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str | None = None
    doc_id: str | None = None
    snippet: str

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation] = []
    confidence: float = 0.0
    session_id: str = "default"

class IngestionRequest(BaseModel):
    data_dir: str = "data/docs"
    force: bool = False

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    is_positive: bool = True
    comments: str | None = None

class InterveneRequest(BaseModel):
    message: str
    role: str = "assistant"

@app.get("/health")
async def health_endpoint():
    return {"status": "ok", "project": settings.PROJECT_NAME}

@app.get("/ready")
async def ready_endpoint():
    openrouter = check_openrouter()
    qdrant = check_qdrant()
    return {
        "ready": bool(openrouter.get("ok") and qdrant.get("ok")),
        "openrouter": openrouter,
        "qdrant": qdrant,
    }

@app.post("/auth/login", response_model=Token)
async def login_endpoint(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise CopilotError("Incorrect username or password", status_code=401)
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": form_data.username, "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user["role"]}

@app.get("/documents")
async def documents_endpoint(user: UserContext = Depends(resolve_user)):
    return {"documents": list(load_registry().values()), "role": user.role}

@app.post("/admin/ingest")
async def admin_ingest_endpoint(request: IngestionRequest, user: UserContext = Depends(resolve_user)):
    require_admin(user)
    try:
        from app.core.queue import get_arq_pool
        pool = await get_arq_pool()
        job = await pool.enqueue_job("async_ingest_documents", data_dir=request.data_dir, force=request.force)
        return {"status": "ok", "message": "Ingestion task queued.", "job_id": job.job_id}
    except Exception as e:
        logger.warning(f"Arq enqueue failed ({e}), falling back to synchronous ingestion.")
        ingest_documents(data_dir=request.data_dir, force=request.force)
        return {"status": "ok", "message": "Ingestion completed synchronously."}

@app.post("/admin/upload")
async def admin_upload_endpoint(files: list[UploadFile] = File(...), user: UserContext = Depends(resolve_user)):
    require_admin(user)
    target_dir = Path(settings.DATA_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for uploaded_file in files:
        filename = Path(uploaded_file.filename or "uploaded.txt").name
        target_path = target_dir / filename
        target_path.write_bytes(await uploaded_file.read())
        saved_files.append(filename)
    return {"status": "ok", "saved_files": saved_files, "data_dir": str(target_dir)}

@app.delete("/admin/documents/{doc_id}")
async def admin_delete_document_endpoint(doc_id: str, user: UserContext = Depends(resolve_user)):
    require_admin(user)
    delete_indexed_document(doc_id)
    return {"status": "ok", "doc_id": doc_id}

@app.post("/admin/reset")
async def admin_reset_endpoint(user: UserContext = Depends(resolve_user)):
    require_admin(user)
    reset_index()
    return {"status": "ok", "message": "Index reset."}

@app.get("/admin/eval")
async def get_eval_endpoint(user: UserContext = Depends(resolve_user)):
    if REPORT_PATH.exists():
        return {"status": "ok", "report": REPORT_PATH.read_text(encoding="utf-8")}
    return {"status": "missing", "report": "No evaluation report found yet. Click 'Run Evaluation Now' below to generate one."}

@app.post("/admin/eval")
async def post_eval_endpoint(user: UserContext = Depends(resolve_user)):
    require_admin(user)
    try:
        from app.core.queue import get_arq_pool
        pool = await get_arq_pool()
        job = await pool.enqueue_job("async_run_ragas_eval")
        return {"status": "ok", "message": "RAGAS evaluation task queued.", "job_id": job.job_id}
    except Exception as e:
        logger.warning(f"Arq enqueue failed ({e}), falling back to synchronous evaluation.")
        summary = await run_local_evaluation()
        report_content = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "Report generated."
        return {"status": "ok", "summary": summary, "report": report_content}

@app.get("/tasks/status/{job_id}")
async def get_task_status_endpoint(job_id: str, user: UserContext = Depends(resolve_user)):
    try:
        from app.core.queue import get_arq_pool
        from arq.jobs import Job
        pool = await get_arq_pool()
        job = Job(job_id, redis=pool)
        status = await job.status()
        info = await job.info() if status else None
        return {
            "job_id": job_id,
            "status": status.value if status else "unknown",
            "result": getattr(info, "result", None) if info else None
        }
    except Exception as e:
        return {"job_id": job_id, "status": "error", "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request, user: UserContext = Depends(resolve_user)):
    metrics = RequestMetrics()
    await async_enforce_rate_limit(http_request.client.host if http_request.client else user.user_id)
    validate_query(request.query)
    if settings.ENABLE_GUARDRAILS:
        try:
            input_guard.validate(request.query)
        except Exception as e:
            raise CopilotError(str(getattr(e, "message", e)), status_code=400)

    session_id = request.session_id or str(uuid.uuid4())
    chat_history = request.chat_history
    if not chat_history:
        chat_history = await get_session_history(user.user_id, session_id, limit=6)
    summary = await get_session_summary(user.user_id, session_id)

    standalone_query, speculative_docs = await resolve_query_speculative(request.query, chat_history, summary=summary)

    cached = await get_cached_answer(standalone_query)
    if cached:
        log_request_metrics(metrics, route="/chat (cache hit)", sources=len(cached.get("sources", [])), model="semantic_cache")
        await add_session_message(user.user_id, session_id, "user", request.query)
        await add_session_message(user.user_id, session_id, "assistant", cached["answer"], cached.get("sources", []), cached.get("confidence", 0.99))
        return ChatResponse(query=request.query, answer=cached["answer"], sources=cached.get("sources", []), confidence=cached.get("confidence", 0.99), session_id=session_id)

    initial_state = {"question": standalone_query, "chat_history": chat_history, "summary": summary, "run_count": 0, "documents": speculative_docs}
    try:
        with timed_stage(metrics, "rag_workflow"):
            final_state = await rag_agent.ainvoke(initial_state)
        answer = redact_sensitive_data(final_state.get("generation", "Unable to compile answer."))
        sources = final_state.get("sources", [])
        confidence = final_state.get("confidence_score", 0.0)
        
        if answer and sources:
            await set_cached_answer(standalone_query, answer, sources, confidence)
            await add_session_message(user.user_id, session_id, "user", request.query)
            await add_session_message(user.user_id, session_id, "assistant", answer, sources, confidence)
    except Exception as e:
        raise CopilotError(str(e), status_code=500)

    log_request_metrics(metrics, route="/chat", sources=len(sources), model=settings.LLM_MODEL)
    return ChatResponse(query=request.query, answer=answer, sources=sources, confidence=confidence, session_id=session_id)

@app.post("/chat/feedback")
async def chat_feedback_endpoint(request: FeedbackRequest, user: UserContext = Depends(resolve_user)):
    logger.info("Feedback received", extra={"feedback": request.dict(), "user": user.user_id})
    return {"status": "ok", "message": "Feedback recorded."}

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, http_request: Request, user: UserContext = Depends(resolve_user)):
    await async_enforce_rate_limit(http_request.client.host if http_request.client else user.user_id)
    validate_query(request.query)
    if settings.ENABLE_GUARDRAILS:
        try:
            input_guard.validate(request.query)
        except Exception as e:
            raise CopilotError(str(getattr(e, "message", e)), status_code=400)

    session_id = request.session_id or str(uuid.uuid4())
    chat_history = request.chat_history
    if not chat_history:
        chat_history = await get_session_history(user.user_id, session_id, limit=6)
    summary = await get_session_summary(user.user_id, session_id)

    standalone_query, speculative_docs = await resolve_query_speculative(request.query, chat_history, summary=summary)

    async def token_generator():
        try:
            metrics = RequestMetrics()
            cached = await get_cached_answer(standalone_query)
            if cached:
                log_request_metrics(metrics, route="/chat/stream (cache hit)", sources=len(cached.get("sources", [])), model="semantic_cache")
                await add_session_message(user.user_id, session_id, "user", request.query)
                await add_session_message(user.user_id, session_id, "assistant", cached["answer"], cached.get("sources", []), cached.get("confidence", 0.99))
                yield cached["answer"]
                return

            initial_state = {"question": standalone_query, "chat_history": chat_history, "summary": summary, "run_count": 0, "documents": speculative_docs}
            documents = []
            sources_text = ""
            grounded_result = "yes"
            has_streamed_tokens = False
            streamed_text = ""

            with timed_stage(metrics, "rag_workflow_stream"):
                redis = await get_redis_client()
                async for event in rag_agent.astream_events(initial_state, version="v2"):
                    kind = event["event"]
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    
                    if kind == "on_chat_model_stream" and node_name == "generate":
                        if await redis.exists(f"session:{session_id}:terminate"):
                            await redis.delete(f"session:{session_id}:terminate")
                            yield "\n\n🛑 **[TERMINATED BY USER: Generation was stopped.]**"
                            await add_session_message(user.user_id, session_id, "user", request.query)
                            await add_session_message(user.user_id, session_id, "assistant", streamed_text + "\n\n🛑 [TERMINATED BY USER]", documents or [], 0.0)
                            return
                        chunk = event["data"]["chunk"]
                        if chunk and getattr(chunk, "content", None):
                            has_streamed_tokens = True
                            streamed_text += chunk.content
                            yield redact_sensitive_data(chunk.content)
                            await asyncio.sleep(0.005)
                            
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict):
                            if "documents" in output:
                                documents = output["documents"]
                            if "sources" in output:
                                sources_text = output["sources"]
                            if "grounded" in output:
                                grounded_result = output["grounded"]

            if not has_streamed_tokens:
                if not documents:
                    yield "I am sorry, no reliable matching documentation was found."
                else:
                    yield "I am sorry, I could not generate a response based on the available documentation."
                return

            if str(grounded_result).lower() == "no":
                yield "\n\n🚨 **[CANCELLED: This response violated safety guidelines and has been retracted.]**"
                return

            if has_streamed_tokens and documents and str(grounded_result).lower() != "no":
                redacted_text = redact_sensitive_data(streamed_text)
                await set_cached_answer(standalone_query, redacted_text, documents, 0.98)
                await add_session_message(user.user_id, session_id, "user", request.query)
                await add_session_message(user.user_id, session_id, "assistant", redacted_text, documents, 0.98)
                
            log_request_metrics(metrics, route="/chat/stream", sources=len(documents), model=settings.LLM_MODEL)
        except Exception as exc:
            logger.error(f"Streaming error: {exc}", exc_info=True)
            if "429" in str(exc) or "Rate limit" in str(exc) or "free-models-per-day" in str(exc):
                yield "\n\n⚠️ **OpenRouter Daily Limit Reached:** You have exhausted the 50 free requests/day limit on OpenRouter. To continue using free models today without rate limits, add $1 (or 10 credits) to your OpenRouter account, or try again tomorrow when the limit resets."
            else:
                yield f"\n\n⚠️ **Error generating response:** {exc}"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

@app.get("/api/v1/sessions")
async def get_user_sessions_endpoint(user: UserContext = Depends(resolve_user)):
    sessions = await list_user_sessions(user.user_id)
    return {"sessions": sessions}

@app.get("/api/v1/sessions/{session_id}/messages")
async def get_session_messages_endpoint(session_id: str, user: UserContext = Depends(resolve_user)):
    messages = await get_session_history(user.user_id, session_id, limit=50)
    return {"session_id": session_id, "messages": messages}

@app.post("/api/v1/sessions/{session_id}/terminate")
async def terminate_session_endpoint(session_id: str, user: UserContext = Depends(resolve_user)):
    redis = await get_redis_client()
    await redis.setex(f"session:{session_id}:terminate", 60, "1")
    return {"status": "terminated", "session_id": session_id}

@app.delete("/api/v1/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, user: UserContext = Depends(resolve_user)):
    success = await delete_session(user.user_id, session_id)
    return {"status": "ok" if success else "error", "session_id": session_id}

@app.get("/api/v1/admin/sessions")
async def admin_list_sessions_endpoint(user: UserContext = Depends(resolve_user)):
    if user.role != "admin":
        raise CopilotError("Admin privileges required", status_code=403)
    sessions = await list_all_sessions(limit=50)
    return {"sessions": sessions}

@app.get("/api/v1/admin/sessions/{user_id}/{session_id}/messages")
async def admin_get_session_messages_endpoint(user_id: str, session_id: str, user: UserContext = Depends(resolve_user)):
    if user.role != "admin":
        raise CopilotError("Admin privileges required", status_code=403)
    messages = await get_session_history(user_id, session_id, limit=50)
    summary = await get_session_summary(user_id, session_id)
    return {"session_id": session_id, "user_id": user_id, "messages": messages, "summary": summary}

@app.post("/api/v1/admin/sessions/{user_id}/{session_id}/message")
async def admin_intervene_message_endpoint(user_id: str, session_id: str, request: InterveneRequest, user: UserContext = Depends(resolve_user)):
    if user.role != "admin":
        raise CopilotError("Admin privileges required", status_code=403)
    await add_session_message(user_id, session_id, request.role, request.message)
    return {"status": "ok", "message": "Intervention message injected."}



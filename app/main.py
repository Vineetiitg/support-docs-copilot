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
from app.engine.document_registry import load_registry
from app.engine.ingestion import delete_indexed_document, ingest_documents, reset_index
from app.engine.context_builder import build_context, format_sources
from app.graph.workflow import compile_workflow
from app.guardrails.input import enforce_rate_limit, validate_query
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

class IngestionRequest(BaseModel):
    data_dir: str = "data/docs"
    force: bool = False

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    is_positive: bool
    comments: str | None = None

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
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/documents")
async def documents_endpoint(user: UserContext = Depends(resolve_user)):
    return {"documents": list(load_registry().values()), "role": user.role}

@app.post("/admin/ingest")
async def admin_ingest_endpoint(request: IngestionRequest, user: UserContext = Depends(resolve_user)):
    require_admin(user)
    ingest_documents(data_dir=request.data_dir, force=request.force)
    return {"status": "ok", "message": "Ingestion completed."}

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
    summary = await run_local_evaluation()
    report_content = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "Report generated."
    return {"status": "ok", "summary": summary, "report": report_content}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, http_request: Request, user: UserContext = Depends(resolve_user)):
    metrics = RequestMetrics()
    enforce_rate_limit(http_request.client.host if http_request.client else user.user_id)
    validate_query(request.query)
    if settings.ENABLE_GUARDRAILS:
        try:
            input_guard.validate(request.query)
        except Exception as e:
            raise CopilotError(str(getattr(e, "message", e)), status_code=400)

    initial_state = {"question": request.query, "chat_history": request.chat_history, "run_count": 0}
    try:
        with timed_stage(metrics, "rag_workflow"):
            final_state = await rag_agent.ainvoke(initial_state)
        answer = redact_sensitive_data(final_state.get("generation", "Unable to compile answer."))
        sources = final_state.get("sources", [])
        confidence = final_state.get("confidence_score", 0.0)
    except Exception as e:
        raise CopilotError(str(e), status_code=500)

    log_request_metrics(metrics, route="/chat", sources=len(sources), model=settings.LLM_MODEL)
    return ChatResponse(query=request.query, answer=answer, sources=sources, confidence=confidence)

@app.post("/chat/feedback")
async def chat_feedback_endpoint(request: FeedbackRequest, user: UserContext = Depends(resolve_user)):
    logger.info("Feedback received", extra={"feedback": request.dict(), "user": user.user_id})
    return {"status": "ok", "message": "Feedback recorded."}

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, http_request: Request, user: UserContext = Depends(resolve_user)):
    enforce_rate_limit(http_request.client.host if http_request.client else user.user_id)
    validate_query(request.query)
    if settings.ENABLE_GUARDRAILS:
        try:
            input_guard.validate(request.query)
        except Exception as e:
            raise CopilotError(str(getattr(e, "message", e)), status_code=400)

    async def token_generator():
        try:
            metrics = RequestMetrics()
            initial_state = {"question": request.query, "chat_history": request.chat_history, "run_count": 0}
            with timed_stage(metrics, "rag_workflow"):
                final_state = await rag_agent.ainvoke(initial_state)
            documents = final_state.get("documents", [])
            
            if not documents:
                yield "I am sorry, no reliable matching documentation was found."
                return

            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.chat_history[-5:]])
            context = build_context(documents)
            prompt = PromptTemplate(
                template="""You are a Support Docs Copilot. Use only the retrieved context to answer the question concisely. If you don't know the answer, say "I don't know".
                
                Chat History:
                {chat_history}
                
                Question: {question} 
                Context: {context} \n\nAnswer:""",
                input_variables=["question", "context", "chat_history"],
            )
            async_llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=0,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base=settings.OPENROUTER_BASE_URL,
                default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
            )
            rag_chain = prompt | async_llm

            async for chunk in rag_chain.astream({"context": context, "question": request.query, "chat_history": history_str}):
                if chunk.content:
                    yield redact_sensitive_data(chunk.content)
                    await asyncio.sleep(0.01)
            yield format_sources(documents)
            log_request_metrics(metrics, route="/chat/stream", sources=len(documents), model=settings.LLM_MODEL)
        except Exception as exc:
            logger.error(f"Streaming error: {exc}", exc_info=True)
            if "429" in str(exc) or "Rate limit" in str(exc) or "free-models-per-day" in str(exc):
                yield "\n\n⚠️ **OpenRouter Daily Limit Reached:** You have exhausted the 50 free requests/day limit on OpenRouter. To continue using free models today without rate limits, add $1 (or 10 credits) to your OpenRouter account, or try again tomorrow when the limit resets."
            else:
                yield f"\n\n⚠️ **Error generating response:** {exc}"

    return StreamingResponse(token_generator(), media_type="text/event-stream")

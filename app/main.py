import asyncio
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from guardrails import Guard
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.dependencies import check_ollama, check_qdrant
from app.core.logging import configure_logging, logger
from app.engine.context_builder import build_context, format_sources
from app.graph.workflow import compile_workflow
from app.guardrails.validators import DetectPromptInjection

configure_logging()
app = FastAPI(title=settings.PROJECT_NAME)
rag_agent = compile_workflow()
input_guard = Guard().use(DetectPromptInjection, on_fail="exception")

class ChatRequest(BaseModel):
    query: str

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

@app.get("/health")
async def health_endpoint():
    return {"status": "ok", "project": settings.PROJECT_NAME}

@app.get("/ready")
async def ready_endpoint():
    ollama = check_ollama()
    qdrant = check_qdrant()
    return {
        "ready": bool(ollama.get("ok") and qdrant.get("ok")),
        "ollama": ollama,
        "qdrant": qdrant,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    started_at = time.perf_counter()
    if len(request.query) > settings.MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail="Query is too long.")
    if settings.ENABLE_GUARDRAILS:
        try:
            input_guard.validate(request.query)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(getattr(e, "message", e)))

    initial_state = {"question": request.query, "run_count": 0}
    try:
        final_state = rag_agent.invoke(initial_state)
        answer = final_state.get("generation", "Unable to compile answer.")
        sources = final_state.get("sources", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("chat completed latency_ms=%s", round((time.perf_counter() - started_at) * 1000, 2))
    return ChatResponse(query=request.query, answer=answer, sources=sources)

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    if len(request.query) > settings.MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail="Query is too long.")
    if settings.ENABLE_GUARDRAILS:
        try:
            input_guard.validate(request.query)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(getattr(e, "message", e)))

    async def token_generator():
        started_at = time.perf_counter()
        initial_state = {"question": request.query, "run_count": 0}
        final_state = rag_agent.invoke(initial_state)
        documents = final_state.get("documents", [])
        
        if not documents:
            yield "I am sorry, no reliable matching documentation was found."
            return

        context = build_context(documents)
        prompt = PromptTemplate(
            template="""You are a Support Docs Copilot. Use only the retrieved context to answer the question concisely. If you don't know the answer, say "I don't know".
            Question: {question} 
            Context: {context} \n\nAnswer:""",
            input_variables=["question", "context"],
        )
        async_llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0, base_url=settings.OLLAMA_BASE_URL)
        rag_chain = prompt | async_llm

        async for chunk in rag_chain.astream({"context": context, "question": request.query}):
            if chunk.content:
                yield chunk.content
                await asyncio.sleep(0.01)
        yield format_sources(documents)
        logger.info("stream completed latency_ms=%s", round((time.perf_counter() - started_at) * 1000, 2))

    return StreamingResponse(token_generator(), media_type="text/event-stream")

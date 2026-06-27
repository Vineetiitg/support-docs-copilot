import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from guardrails import Guard
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.graph.workflow import compile_workflow
from app.guardrails.validators import DetectPromptInjection

app = FastAPI(title=settings.PROJECT_NAME)
rag_agent = compile_workflow()
input_guard = Guard().use(DetectPromptInjection, on_fail="exception")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        input_guard.validate(request.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(getattr(e, "message", e)))

    initial_state = {"question": request.query, "run_count": 0}
    try:
        final_state = rag_agent.invoke(initial_state)
        answer = final_state.get("generation", "Unable to compile answer.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(query=request.query, answer=answer)

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    try:
        input_guard.validate(request.query)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(getattr(e, "message", e)))

    async def token_generator():
        initial_state = {"question": request.query, "run_count": 0}
        final_state = rag_agent.invoke(initial_state)
        documents = final_state.get("documents", [])
        
        if not documents:
            yield "I am sorry, no reliable matching documentation was found."
            return

        context = "\n\n".join(doc.page_content for doc in documents)
        prompt = PromptTemplate(
            template="""You are a Support Docs Copilot. Use the retrieved context to answer the question concisely. If you don't know the answer, say "I don't know".
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

    return StreamingResponse(token_generator(), media_type="text/event-stream")

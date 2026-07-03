import json
from typing import List, Optional, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from app.core.config import settings
from app.core.logging import logger
from app.engine.context_builder import build_context, source_citations
from app.engine.retriever import retrieve_documents
from app.engine.reranker import rerank_documents, evaluate_nli_groundedness

class GraphState(TypedDict):
    question: str
    chat_history: List[dict]
    generation: str
    documents: List[Document]
    sources: Optional[list[dict]]
    run_count: int
    confidence_score: float
    grounded: str

import httpx

_http_client = httpx.AsyncClient(
    http2=True,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
    timeout=httpx.Timeout(60.0, connect=10.0),
)

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=0,
    openai_api_key=settings.OPENROUTER_API_KEY,
    openai_api_base=settings.OPENROUTER_BASE_URL,
    default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
    http_async_client=_http_client,
)
llm_json = ChatOpenAI(
    model=getattr(settings, "FAST_LLM_MODEL", settings.LLM_MODEL),
    temperature=0,
    openai_api_key=settings.OPENROUTER_API_KEY,
    openai_api_base=settings.OPENROUTER_BASE_URL,
    default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
    http_async_client=_http_client,
)
llm_slow = ChatOpenAI(
    model=getattr(settings, "SLOW_LLM_MODEL", settings.LLM_MODEL),
    temperature=0,
    openai_api_key=settings.OPENROUTER_API_KEY,
    openai_api_base=settings.OPENROUTER_BASE_URL,
    default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
    http_async_client=_http_client,
)

async def retrieve(state: GraphState):
    logger.info("NODE: RETRIEVE DOCS")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    run_count = state.get("run_count", 0)
    documents = await retrieve_documents(question, chat_history)
    return {"documents": documents, "sources": source_citations(documents), "question": question, "run_count": run_count}

async def grade_documents(state: GraphState):
    logger.info("NODE: GRADE DOCUMENT RELEVANCE")
    question = state["question"]
    documents = state.get("documents", [])
    
    reranked_docs = rerank_documents(question, documents, top_k=3)
    if not reranked_docs:
        return {"documents": []}
        
    top_score = reranked_docs[0].metadata.get("rerank_score", -10.0)
    if top_score >= 0.0:
        logger.info(f"High confidence Cross-Encoder score ({top_score:.4f} >= 0.0). Skipping LLM grader.")
        return {"documents": reranked_docs}
        
    docs_text = "\n\n".join([f"[{idx+1}] ID: {d.metadata.get('doc_id', idx+1)}\nContent: {d.page_content}" for idx, d in enumerate(reranked_docs)])
    prompt = PromptTemplate(
        template="""You are a strict grader assessing relevance of retrieved documents to a user question.
        User Question: {question}
        
        Retrieved Documents:
        {docs_text}
        
        For each document [1] to [{count}], assess if it contains keywords or semantic meaning relevant to the question.
        Return ONLY a JSON object with a key 'results' containing a list of objects: [{{"id": 1, "relevant": true}}, ...].""",
        input_variables=["question", "docs_text", "count"],
    )
    grader = prompt | llm_json
    result = await grader.ainvoke({"question": question, "docs_text": docs_text, "count": len(reranked_docs)})
    
    filtered_docs = []
    try:
        parsed = json.loads(result.content)
        results = parsed.get("results", [])
        relevant_indices = set()
        for r in results:
            if r.get("relevant") is True or str(r.get("relevant")).lower() == "true":
                idx_val = r.get("id")
                if isinstance(idx_val, int) and 1 <= idx_val <= len(reranked_docs):
                    relevant_indices.add(idx_val - 1)
        for i, d in enumerate(reranked_docs):
            if i in relevant_indices:
                filtered_docs.append(d)
    except Exception as e:
        logger.warning(f"Batch grading parse failed ({e}), keeping all {len(reranked_docs)} reranked docs.")
        filtered_docs = reranked_docs
        
    if not filtered_docs and reranked_docs:
        top_score = reranked_docs[0].metadata.get("rerank_score", 0)
        if top_score > 0.0:
            filtered_docs = [reranked_docs[0]]
            
    return {"documents": filtered_docs}

async def generate(state: GraphState):
    logger.info("NODE: GENERATE ANSWER")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    run_count = state.get("run_count", 0) + 1
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
    context = build_context(documents)
    prompt = PromptTemplate(
        template="""You are a Support Docs Copilot. Use only the retrieved context to answer the question concisely.
        
        CRITICAL INSTRUCTION (Cite-to-Write):
        You must append [doc_id] to the end of every sentence. Do not write a sentence if you cannot cite a source from the retrieved context. If the context does not contain the answer, say "I don't know".
        
        Chat History:
        {chat_history}
        
        Question: {question} 
        Context: {context} 
        Answer:""",
        input_variables=["question", "context", "chat_history"],
    )
    selected_llm = llm_slow if run_count > 1 else llm
    if run_count > 1:
        logger.info(f"Using slow reasoning model ({getattr(settings, 'SLOW_LLM_MODEL', 'default')}) for retry attempt #{run_count}")
    rag_chain = prompt | selected_llm
    generation = await rag_chain.ainvoke({"context": context, "question": question, "chat_history": history_str})
    return {"generation": generation.content, "sources": source_citations(documents), "run_count": run_count}

async def decide_to_generate(state: GraphState):
    if not state["documents"]:
        logger.info("ROUTE: ALL DOCS IRRELEVANT")
        return "end"
    logger.info("ROUTE: RELEVANT DOCS FOUND")
    return "generate"

async def evaluate_answer(state: GraphState):
    logger.info("NODE: EVALUATE ANSWER")
    documents = state["documents"]
    generation = state["generation"]
    
    context = build_context(documents)
    grade, confidence = evaluate_nli_groundedness(context, generation)
        
    return {"grounded": grade, "confidence_score": confidence}

async def check_hallucinations(state: GraphState):
    run_count = state["run_count"]
    
    if run_count >= 3:
        logger.info("ROUTE: MAX RETRIES REACHED")
        return "end"
        
    grade = state.get("grounded", "yes")
        
    if grade.lower() == "yes":
        logger.info("ROUTE: GROUNDED")
        return "end"
    logger.info("ROUTE: HALLUCINATION DETECTED")
    return "regenerate"

def compile_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate_answer", evaluate_answer)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "end": END})
    workflow.add_edge("generate", "evaluate_answer")
    workflow.add_conditional_edges("evaluate_answer", check_hallucinations, {"end": END, "regenerate": "generate"})
    return workflow.compile()

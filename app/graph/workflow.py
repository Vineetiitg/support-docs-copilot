import json
from typing import List, Optional, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from app.core.config import settings
from app.core.logging import logger
from app.engine.context_builder import build_context, source_citations
from app.engine.retriever import retrieve_documents

class GraphState(TypedDict):
    question: str
    chat_history: List[dict]
    generation: str
    documents: List[Document]
    sources: Optional[list[dict]]
    run_count: int
    confidence_score: float
    grounded: str

llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0, base_url=settings.OLLAMA_BASE_URL)
llm_json = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0, format="json", base_url=settings.OLLAMA_BASE_URL)

def retrieve(state: GraphState):
    logger.info("NODE: RETRIEVE DOCS")
    question = state["question"]
    chat_history = state.get("chat_history", [])
    run_count = state.get("run_count", 0)
    documents = retrieve_documents(question, chat_history)
    return {"documents": documents, "sources": source_citations(documents), "question": question, "run_count": run_count}

def grade_documents(state: GraphState):
    logger.info("NODE: GRADE DOCUMENT RELEVANCE")
    question = state["question"]
    documents = state.get("documents", [])
    
    prompt = PromptTemplate(
        template="""You are a strict grader assessing relevance of a retrieved document to a user question.
        Document: \n\n {document} \n\n
        Question: {question} \n
        If the document contains keywords or semantic meaning related to the question, grade it as 'yes'. Otherwise, 'no'.
        Provide a JSON with a single key 'score' and value 'yes' or 'no'.""",
        input_variables=["question", "document"],
    )
    grader = prompt | llm_json
    
    filtered_docs = []
    for d in documents:
        result = grader.invoke({"question": question, "document": d.page_content})
        try:
            grade = json.loads(result.content).get("score", "no")
        except:
            grade = "no"
        if grade.lower() == "yes":
            filtered_docs.append(d)
            
    return {"documents": filtered_docs}

def generate(state: GraphState):
    logger.info("NODE: GENERATE ANSWER")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    run_count = state.get("run_count", 0) + 1
    
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-5:]])
    context = build_context(documents)
    prompt = PromptTemplate(
        template="""You are a Support Docs Copilot. Use only the retrieved context to answer the question concisely. If the context does not contain the answer, say "I don't know".
        
        Chat History:
        {chat_history}
        
        Question: {question} 
        Context: {context} 
        Answer:""",
        input_variables=["question", "context", "chat_history"],
    )
    rag_chain = prompt | llm
    generation = rag_chain.invoke({"context": context, "question": question, "chat_history": history_str})
    return {"generation": generation.content, "sources": source_citations(documents), "run_count": run_count}

def decide_to_generate(state: GraphState):
    if not state["documents"]:
        logger.info("ROUTE: ALL DOCS IRRELEVANT")
        return "end"
    logger.info("ROUTE: RELEVANT DOCS FOUND")
    return "generate"

def evaluate_answer(state: GraphState):
    logger.info("NODE: EVALUATE ANSWER")
    documents = state["documents"]
    generation = state["generation"]
    
    context = build_context(documents)
    prompt = PromptTemplate(
        template="""You are evaluating whether a generated answer is fully grounded in the retrieved facts.
        Facts: \n\n {context} \n\n
        Answer: {generation} \n
        If the answer is supported by the facts, return 'yes'. If it contains hallucinations, return 'no'.
        Provide a JSON with keys 'score' (yes/no) and 'confidence' (float 0.0-1.0).""",
        input_variables=["context", "generation"],
    )
    grader = prompt | llm_json
    
    result = grader.invoke({"context": context, "generation": generation})
    try:
        parsed = json.loads(result.content)
        grade = parsed.get("score", "yes")
        confidence = float(parsed.get("confidence", 0.8))
    except:
        grade = "yes"
        confidence = 0.5
        
    return {"grounded": grade, "confidence_score": confidence}

def check_hallucinations(state: GraphState):
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

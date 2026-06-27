import json
from typing import List, Optional, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

from app.core.config import settings
from app.engine.context_builder import build_context, source_citations
from app.engine.retriever import retrieve_documents

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    sources: Optional[list[dict]]
    run_count: int

llm = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0, base_url=settings.OLLAMA_BASE_URL)
llm_json = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0, format="json", base_url=settings.OLLAMA_BASE_URL)

def retrieve(state: GraphState):
    print("--- NODE: RETRIEVE DOCS ---")
    question = state["question"]
    run_count = state.get("run_count", 0)
    documents = retrieve_documents(question)
    return {"documents": documents, "sources": source_citations(documents), "question": question, "run_count": run_count}

def grade_documents(state: GraphState):
    print("--- NODE: GRADE DOCUMENT RELEVANCE ---")
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
    print("--- NODE: GENERATE ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    run_count = state.get("run_count", 0) + 1
    
    context = build_context(documents)
    prompt = PromptTemplate(
        template="""You are a Support Docs Copilot. Use only the retrieved context to answer the question concisely. If the context does not contain the answer, say "I don't know".
        Question: {question} 
        Context: {context} 
        Answer:""",
        input_variables=["question", "context"],
    )
    rag_chain = prompt | llm
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"generation": generation.content, "sources": source_citations(documents), "run_count": run_count}

def decide_to_generate(state: GraphState):
    if not state["documents"]:
        print("--- ROUTE: ALL DOCS IRRELEVANT ---")
        return "end"
    print("--- ROUTE: RELEVANT DOCS FOUND ---")
    return "generate"

def check_hallucinations(state: GraphState):
    documents = state["documents"]
    generation = state["generation"]
    run_count = state["run_count"]
    
    if run_count >= 3:
        print("--- ROUTE: MAX RETRIES REACHED ---")
        return "end"
        
    context = build_context(documents)
    prompt = PromptTemplate(
        template="""You are evaluating whether a generated answer is fully grounded in the retrieved facts.
        Facts: \n\n {context} \n\n
        Answer: {generation} \n
        If the answer is supported by the facts, return 'yes'. If it contains hallucinations, return 'no'.
        Provide a JSON with a single key 'score' and value 'yes' or 'no'.""",
        input_variables=["context", "generation"],
    )
    grader = prompt | llm_json
    
    result = grader.invoke({"context": context, "generation": generation})
    try:
        grade = json.loads(result.content).get("score", "yes")
    except:
        grade = "yes"
        
    if grade.lower() == "yes":
        print("--- ROUTE: GROUNDED ---")
        return "end"
    print("--- ROUTE: HALLUCINATION DETECTED ---")
    return "regenerate"

def compile_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "end": END})
    workflow.add_conditional_edges("generate", check_hallucinations, {"end": END, "regenerate": "generate"})
    return workflow.compile()

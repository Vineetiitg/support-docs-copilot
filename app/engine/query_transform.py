import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.logging import logger


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


async def query_variants(query: str, chat_history: list[dict] = None) -> list[str]:
    normalized = normalize_query(query)
    variants = [normalized]
    history_str = ""
    if chat_history:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-3:]])
    
    try:
        llm = ChatOpenAI(
            model=getattr(settings, "FAST_LLM_MODEL", settings.LLM_MODEL),
            temperature=0,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
        )
        prompt = PromptTemplate(
            template="""You are an expert technical support assistant. 
Your goal is to generate 2 alternative phrasing variants for the user's question to improve retrieval accuracy.
Return ONLY a JSON object with a single key 'variants' containing a list of strings.

Chat History:
{chat_history}

User Question: {question}""",
            input_variables=["question", "chat_history"],
        )
        chain = prompt | llm
        result = await chain.ainvoke({"question": normalized, "chat_history": history_str})
        
        parsed = json.loads(result.content)
        new_variants = parsed.get("variants", [])
        
        if isinstance(new_variants, list):
            for variant in new_variants:
                if isinstance(variant, str) and variant.strip():
                    variants.append(variant.strip())
                    
    except Exception as e:
        logger.warning(f"Failed to generate query variants with LLM: {e}")
        
    return list(dict.fromkeys(variants))


async def condense_query(query: str, chat_history: list[dict] = None) -> str:
    if not chat_history:
        return normalize_query(query)
        
    normalized = normalize_query(query)
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-6:]])
    
    try:
        llm = ChatOpenAI(
            model=getattr(settings, "FAST_LLM_MODEL", settings.LLM_MODEL),
            temperature=0,
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
        )
        prompt = PromptTemplate(
            template="""Given the following chat history and a follow-up question, rewrite the follow-up question into a standalone query that can be understood without the chat history. Do not answer the question, just reformulate it. If the follow-up question is already standalone, return it unchanged.

Chat History:
{chat_history}

Follow Up Question: {question}
Standalone Query:""",
            input_variables=["question", "chat_history"],
        )
        chain = prompt | llm
        result = await chain.ainvoke({"question": normalized, "chat_history": history_str})
        standalone = result.content.strip()
        if standalone:
            logger.debug(f"Condensed query: '{query}' -> '{standalone}'")
            return standalone
    except Exception as e:
        logger.warning(f"Failed to condense query with LLM: {e}")
        
    return normalized

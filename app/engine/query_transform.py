import json
import re

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.logging import logger


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def query_variants(query: str) -> list[str]:
    normalized = normalize_query(query)
    variants = [normalized]
    
    try:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL, 
            temperature=0, 
            format="json", 
            base_url=settings.OLLAMA_BASE_URL
        )
        prompt = PromptTemplate(
            template="""You are an expert technical support assistant. 
Your goal is to generate 2 alternative phrasing variants for the user's question to improve retrieval accuracy.
Return ONLY a JSON object with a single key 'variants' containing a list of strings.

User Question: {question}""",
            input_variables=["question"],
        )
        chain = prompt | llm
        result = chain.invoke({"question": normalized})
        
        parsed = json.loads(result.content)
        new_variants = parsed.get("variants", [])
        
        if isinstance(new_variants, list):
            for variant in new_variants:
                if isinstance(variant, str) and variant.strip():
                    variants.append(variant.strip())
                    
    except Exception as e:
        logger.warning(f"Failed to generate query variants with LLM: {e}")
        
    return list(dict.fromkeys(variants))

import asyncio
from typing import Any
from arq.connections import RedisSettings
from app.core.config import settings
from app.core.logging import logger
from app.engine.ingestion import ingest_documents


async def async_ingest_documents(ctx: dict[str, Any], data_dir: str = "data/docs", force: bool = False) -> dict[str, Any]:
    logger.info(f"[Arq Worker] Starting document ingestion for dir: {data_dir}, force={force}")
    # Run CPU-heavy synchronous chunking & ONNX embedding in a thread pool so we don't block the asyncio event loop
    await asyncio.to_thread(ingest_documents, data_dir=data_dir, force=force)
    logger.info("[Arq Worker] Document ingestion completed successfully.")
    return {"status": "SUCCESS", "message": "Documents ingested successfully.", "data_dir": data_dir}


async def async_run_ragas_eval(ctx: dict[str, Any]) -> dict[str, Any]:
    logger.info("[Arq Worker] Starting RAGAS benchmark evaluation...")
    # Import inside task to avoid circular imports or heavy startup load
    from app.tests.eval_rag import run_local_evaluation
    result_path = await run_local_evaluation()
    logger.info(f"[Arq Worker] RAGAS evaluation completed. Report saved at: {result_path}")
    return {"status": "SUCCESS", "report_path": str(result_path)}


async def async_llm_throttle_call(ctx: dict[str, Any], prompt: str) -> dict[str, Any]:
    """
    Role D: Concurrency Throttled LLM call via Arq queue to prevent OpenRouter HTTP 429 errors.
    """
    logger.info("[Arq Worker] Processing throttled LLM request...")
    from app.core.dependencies import get_llm
    llm = get_llm()
    response = await llm.ainvoke(prompt)
    return {"status": "SUCCESS", "content": response.content}


class WorkerSettings:
    functions = [async_ingest_documents, async_run_ragas_eval, async_llm_throttle_call]
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_jobs = 5  # Role D: Concurrency limit per worker container to smooth traffic spikes
    job_timeout = 600  # 10 minutes max for heavy ingestion or evaluation jobs

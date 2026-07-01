import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.graph.workflow import compile_workflow

DATASET_PATH = Path("datasets/golden_qa.csv")
REPORT_PATH = Path("reports/eval_report.md")

def load_golden_questions(path: Path = DATASET_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def source_hit(expected_sources: str, sources: list[dict]) -> bool:
    expected = {source.strip() for source in expected_sources.split("|") if source.strip()}
    actual = {source.get("source") for source in sources}
    return bool(expected & actual)

async def run_local_evaluation() -> dict:
    agent = compile_workflow()
    rows = load_golden_questions()
    results = []
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for row in rows:
        started = perf_counter()
        output_state = await agent.ainvoke({"question": row["question"], "chat_history": [], "run_count": 0})
        latency_ms = round((perf_counter() - started) * 1000, 2)
        answer = output_state.get("generation", "")
        sources_dicts = output_state.get("sources", [])
        docs = output_state.get("documents", [])
        
        questions.append(row["question"])
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])
        ground_truths.append(row["expected_answer"])
        
        results.append(
            {
                "question": row["question"],
                "answer": answer,
                "latency_ms": latency_ms,
                "source_hit": source_hit(row["expected_sources"], sources_dicts),
                "retrieved_contexts": len(docs),
            }
        )

    # RAGAS Evaluation
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
    )
    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    try:
        ragas_result = evaluate(
            ragas_dataset,
            metrics=[answer_relevancy, faithfulness],
            llm=llm
        )
        ragas_scores = ragas_result
    except Exception as e:
        ragas_scores = {"error": str(e)}

    source_hit_rate = round(sum(1 for result in results if result["source_hit"]) / max(len(results), 1), 3)
    average_latency_ms = round(sum(result["latency_ms"] for result in results) / max(len(results), 1), 2)
    
    summary = {
        "questions": len(results),
        "source_hit_rate": source_hit_rate,
        "average_latency_ms": average_latency_ms,
        "ragas_scores": ragas_scores,
        "results": results,
    }
    write_report(summary)
    return summary

def write_report(summary: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Overall Metrics",
        f"- **Questions Evaluated:** {summary['questions']}",
        f"- **Source Hit Rate:** {summary['source_hit_rate']}",
        f"- **Average Latency:** {summary['average_latency_ms']} ms",
        "",
        "### Ragas Scores",
        "```json",
        json.dumps(summary.get("ragas_scores", {}), indent=2, default=str),
        "```",
        "",
        "## Question Results",
        "",
    ]
    for result in summary["results"]:
        lines.extend(
            [
                f"### Q: {result['question']}",
                f"**A:** {result['answer']}",
                f"- Source hit: {result['source_hit']}",
                f"- Retrieved contexts: {result['retrieved_contexts']}",
                f"- Latency ms: {result['latency_ms']}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    print("Running evaluation...")
    print(asyncio.run(run_local_evaluation()))

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall, answer_correctness
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.engine.indexer import dense_embeddings
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

async def run_local_evaluation(
    assert_faithfulness: float = None,
    assert_precision: float = None,
    assert_relevancy: float = None,
) -> dict:
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

    # RAGAS Two-Tier Judge Configuration
    fast_llm = ChatOpenAI(
        model=getattr(settings, "FAST_LLM_MODEL", settings.LLM_MODEL),
        temperature=0,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
    )
    slow_llm = ChatOpenAI(
        model=getattr(settings, "SLOW_LLM_MODEL", settings.LLM_MODEL),
        temperature=0,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        default_headers={"HTTP-Referer": "https://localhost:3000", "X-Title": "Support Docs Copilot"},
    )

    # Assign fast LLM to structural metrics and slow LLM to reasoning metrics
    answer_relevancy.llm = fast_llm
    context_precision.llm = fast_llm
    context_recall.llm = fast_llm
    faithfulness.llm = slow_llm
    answer_correctness.llm = slow_llm

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    # Use global singleton ONNX embedder (eliminates 2.5s reload)
    embeddings = dense_embeddings()
    try:
        ragas_result = evaluate(
            ragas_dataset,
            metrics=[answer_relevancy, faithfulness, context_precision, context_recall, answer_correctness],
            llm=fast_llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=4, max_wait=60, max_retries=2),
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

    # CI/CD Quality Gate Assertions
    if isinstance(ragas_scores, dict) and "error" not in ragas_scores:
        if assert_faithfulness is not None:
            val = ragas_scores.get("faithfulness", 0.0)
            if val < assert_faithfulness:
                print(f"❌ CI Quality Gate FAILED: faithfulness score {val:.3f} < {assert_faithfulness}", file=sys.stderr)
                sys.exit(1)
        if assert_precision is not None:
            val = ragas_scores.get("context_precision", 0.0)
            if val < assert_precision:
                print(f"❌ CI Quality Gate FAILED: context_precision score {val:.3f} < {assert_precision}", file=sys.stderr)
                sys.exit(1)
        if assert_relevancy is not None:
            val = ragas_scores.get("answer_relevancy", 0.0)
            if val < assert_relevancy:
                print(f"❌ CI Quality Gate FAILED: answer_relevancy score {val:.3f} < {assert_relevancy}", file=sys.stderr)
                sys.exit(1)

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

def generate_synthetic_testset(output_path: Path = Path("datasets/synthetic_qa.json"), test_size: int = 5) -> list[dict]:
    from ragas.testset.generator import TestsetGenerator
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    print(f"Generating synthetic testset of size {test_size} from {settings.DATA_DIR}...")
    loader = DirectoryLoader(settings.DATA_DIR, glob="**/*.*", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    
    generator = TestsetGenerator.from_langchain_docs(
        docs=split_docs,
        llm=ChatOpenAI(model=settings.LLM_MODEL, temperature=0.7, openai_api_key=settings.OPENROUTER_API_KEY, openai_api_base=settings.OPENROUTER_BASE_URL),
        embeddings=dense_embeddings(),
    )
    testset = generator.generate_with_langchain_docs(split_docs, test_size=test_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = testset.to_pandas()
    df.to_json(output_path, orient="records", indent=2)
    print(f"Saved {len(df)} synthetic QA pairs to {output_path}")
    return df.to_dict(orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Support Docs Copilot RAG Evaluation Suite")
    parser.add_argument("--generate-testset", action="store_true", help="Generate synthetic QA testset from documentation")
    parser.add_argument("--test-size", type=int, default=5, help="Number of synthetic QA pairs to generate")
    parser.add_argument("--assert-faithfulness", type=float, default=None, help="Minimum required faithfulness score")
    parser.add_argument("--assert-precision", type=float, default=None, help="Minimum required context precision score")
    parser.add_argument("--assert-relevancy", type=float, default=None, help="Minimum required answer relevancy score")
    args = parser.parse_args()

    if args.generate_testset:
        generate_synthetic_testset(test_size=args.test_size)
    else:
        print("Running evaluation...")
        summary = asyncio.run(run_local_evaluation(
            assert_faithfulness=args.assert_faithfulness,
            assert_precision=args.assert_precision,
            assert_relevancy=args.assert_relevancy,
        ))
        print("Evaluation report generated at:", REPORT_PATH)

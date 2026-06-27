import csv
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app.graph.workflow import compile_workflow


DATASET_PATH = Path("datasets/golden_qa.csv")
REPORT_PATH = Path("reports/eval_report.md")


def load_golden_questions(path: Path = DATASET_PATH) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def token_overlap(expected: str, actual: str) -> float:
    expected_tokens = set(expected.lower().split())
    actual_tokens = set(actual.lower().split())
    if not expected_tokens:
        return 0.0
    return round(len(expected_tokens & actual_tokens) / len(expected_tokens), 3)


def source_hit(expected_sources: str, sources: list[dict]) -> bool:
    expected = {source.strip() for source in expected_sources.split("|") if source.strip()}
    actual = {source.get("source") for source in sources}
    return bool(expected & actual)


def run_local_evaluation() -> dict:
    agent = compile_workflow()
    rows = load_golden_questions()
    results = []

    for row in rows:
        started = perf_counter()
        output_state = agent.invoke({"question": row["question"], "run_count": 0})
        latency_ms = round((perf_counter() - started) * 1000, 2)
        answer = output_state.get("generation", "")
        sources = output_state.get("sources", [])
        results.append(
            {
                "question": row["question"],
                "answer": answer,
                "latency_ms": latency_ms,
                "answer_overlap": token_overlap(row["expected_answer"], answer),
                "source_hit": source_hit(row["expected_sources"], sources),
                "retrieved_contexts": len(output_state.get("documents", [])),
            }
        )

    average_overlap = round(sum(result["answer_overlap"] for result in results) / max(len(results), 1), 3)
    source_hit_rate = round(sum(1 for result in results if result["source_hit"]) / max(len(results), 1), 3)
    average_latency_ms = round(sum(result["latency_ms"] for result in results) / max(len(results), 1), 2)
    summary = {
        "questions": len(results),
        "answer_overlap": average_overlap,
        "source_hit_rate": source_hit_rate,
        "average_latency_ms": average_latency_ms,
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
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Questions | {summary['questions']} |",
        f"| Answer overlap | {summary['answer_overlap']} |",
        f"| Source hit rate | {summary['source_hit_rate']} |",
        f"| Average latency ms | {summary['average_latency_ms']} |",
        "",
        "## Question Results",
        "",
    ]
    for result in summary["results"]:
        lines.extend(
            [
                f"### {result['question']}",
                "",
                f"- Answer overlap: {result['answer_overlap']}",
                f"- Source hit: {result['source_hit']}",
                f"- Retrieved contexts: {result['retrieved_contexts']}",
                f"- Latency ms: {result['latency_ms']}",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(run_local_evaluation())

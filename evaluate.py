"""Measure how well the chatbot answers the questions in eval/questions.jsonl.

Examples:
    python evaluate.py --config both            # compare baseline vs improved
    python evaluate.py --config improved        # run one config
    python evaluate.py --config both --provider mock   # offline dry run, no API key

For each config it reports:
    * answer accuracy     - share of questions answered correctly
    * retrieval hit rate  - share of answerable questions where the right
                            document was among the retrieved chunks
    * refusal accuracy    - share of unanswerable questions where the bot
                            said it didn't know instead of guessing

Full per-question results are saved in results/ so you can see exactly
which questions failed and why.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from qa.llm import make_llm
from qa.pipeline import CONFIGS, QAPipeline
from qa.scoring import score_answer

ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "results"


def load_questions(path: Path = QUESTIONS_FILE) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def run_config(config_name: str, provider: str, questions: list[dict], sleep: float) -> dict:
    pipeline = QAPipeline(CONFIGS[config_name], make_llm(provider))
    rows = []
    for item in questions:
        result = pipeline.ask(item["question"])
        sources = [chunk.source for chunk in result.sources]
        correct = score_answer(result.answer, item)
        retrieval_hit = item["source"] in sources if item.get("answerable", True) else None
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answerable": item.get("answerable", True),
                "expected_keywords": item["expected_keywords"],
                "answer": result.answer,
                "retrieved_sources": sources,
                "retrieval_hit": retrieval_hit,
                "correct": correct,
            }
        )
        mark = "PASS" if correct else "FAIL"
        print(f"  [{mark}] Q{item['id']:>2}: {item['question']}")
        if sleep:
            time.sleep(sleep)  # stay under free-tier rate limits

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "config": config_name,
        "provider": provider,
        "model": getattr(pipeline.llm, "model", "unknown"),
        "questions": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy": percent(sum(r["correct"] for r in rows), len(rows)),
        "retrieval_hit_rate": percent(sum(bool(r["retrieval_hit"]) for r in answerable), len(answerable)),
        "refusal_accuracy": percent(sum(r["correct"] for r in unanswerable), len(unanswerable)),
    }
    return {"summary": summary, "results": rows}


def write_report(runs: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for run in runs:
        name = run["summary"]["config"]
        (RESULTS_DIR / f"{stamp}_{name}.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    lines = [
        "| Config | Model | Accuracy | Correct | Retrieval hit rate | Refusal accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        s = run["summary"]
        lines.append(
            f"| {s['config']} | {s['model']} | {s['accuracy']} | {s['correct']}/{s['questions']} "
            f"| {s['retrieval_hit_rate']} | {s['refusal_accuracy']} |"
        )
    summary_path = RESULTS_DIR / f"{stamp}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the document Q&A chatbot.")
    parser.add_argument("--config", choices=[*CONFIGS, "both"], default="both")
    parser.add_argument("--provider", choices=["llm", "mock"], default="llm")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds to wait between API calls")
    args = parser.parse_args()

    questions = load_questions()[: args.limit]
    names = list(CONFIGS) if args.config == "both" else [args.config]
    sleep = 0.0 if args.provider == "mock" else args.sleep

    runs = []
    for name in names:
        print(f"\nRunning config '{name}' ({len(questions)} questions)...")
        runs.append(run_config(name, args.provider, questions, sleep))

    summary_path = write_report(runs)
    print("\n" + summary_path.read_text(encoding="utf-8"))
    print(f"Saved detailed results to {RESULTS_DIR}/")
    if args.provider == "mock":
        print("Note: the mock provider does not really answer questions. Use --provider llm for real scores.")


if __name__ == "__main__":
    main()

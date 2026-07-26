from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.metrics.citation_accuracy import aggregate, score_citations
from src.copilot.generation.prompts import VERSION as PROMPT_VERSION
from src.copilot.pipeline import answer_question

load_dotenv()

logger = logging.getLogger(__name__)


def load_golden(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def build_eval_rows(golden: list[dict], top_k: int) -> list[dict]:
    rows = []
    for i, item in enumerate(golden, 1):
        logger.info("[%d/%d] %s", i, len(golden), item["question"][:70])
        try:
            result = await answer_question(item["question"], k=top_k)
            rows.append({
                "question": item["question"],
                "answer": result.generation.answer.answer,
                "contexts": [c.text for c in result.chunks],
                "ground_truth": " ".join(item["expected_answer_contains"]),
                "id": item["id"],
                "citations": [c.model_dump() for c in result.generation.answer.citations],
            })
        except Exception:
            logger.exception("Skipping %s", item["id"])
    return rows


async def main_async(dataset_path: Path, top_k: int, run_name: str,
                     output_path: Path | None) -> dict:
    golden = load_golden(dataset_path)
    logger.info("Loaded %d questions", len(golden))

    rows = await build_eval_rows(golden, top_k)

    # Citation accuracy — cheap, deterministic
    citation_scores = [
        score_citations(
            predicted_citations=row["citations"],
            expected_sources=item["expected_sources"],
        )
        for row, item in zip(rows, golden)
    ]
    citation_metrics = aggregate(citation_scores)

    # Ragas — for the JSON dataset, drop "citations" and "id" first
    ragas_rows = [
        {k: v for k, v in row.items() if k not in ("citations", "id")}
        for row in rows
    ]
    ds = Dataset.from_list(ragas_rows)

    logger.info("Running Ragas on %d rows", len(rows))
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {k: float(v) for k, v in result._repr_dict.items()}
    all_metrics = {**scores, **citation_metrics}
    logger.info("Scores: %s", all_metrics)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ibs-cbs-copilot")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("prompt_version", PROMPT_VERSION)
        mlflow.log_param("golden_size", len(golden))
        mlflow.log_param("dataset", dataset_path.name)
        for name, value in all_metrics.items():
            mlflow.log_metric(name, value)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_metrics, indent=2))
        logger.info("Wrote %s", output_path)

    return all_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("evals/golden/golden_v1.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-name", type=str, default="v0-baseline")
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY missing")
        return 1

    asyncio.run(main_async(args.dataset, args.top_k, args.run_name, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
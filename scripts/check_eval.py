from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowflow.agent import RagAgent
from knowflow.chunking import load_documents_from_path
from knowflow.evaluation import compare_retrieval_strategies, evaluate
from knowflow.store_factory import create_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline RAG evaluation with quality thresholds.")
    parser.add_argument("--docs", default="sample_docs", help="Directory with source documents.")
    parser.add_argument("--eval-set", default="evals/rag_eval_set.jsonl", help="Primary JSONL evaluation set.")
    parser.add_argument("--holdout-set", default="evals/rag_holdout.jsonl", help="Independent JSONL holdout set.")
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-mrr", type=float, default=0.90)
    parser.add_argument("--min-citation-accuracy", type=float, default=0.95)
    parser.add_argument("--min-faithfulness", type=float, default=0.95)
    parser.add_argument("--max-permission-leaks", type=int, default=0)
    parser.add_argument("--skip-experiment", action="store_true", help="Skip the four-strategy comparison.")
    args = parser.parse_args()

    store_dir = Path(tempfile.mkdtemp(prefix="knowflow-eval-"))
    try:
        store = create_store(str(store_dir), backend="jsonl")
        store.reset()
        documents = load_documents_from_path(Path(args.docs))
        store.add_documents(documents)
        agent = RagAgent(store)
        result = evaluate(agent, Path(args.eval_set))
        holdout = evaluate(agent, Path(args.holdout_set))
        experiment = None if args.skip_experiment else compare_retrieval_strategies(agent, Path(args.eval_set))
        summary = {
            "total": result.total,
            "recall_at_k": result.recall_at_k,
            "mrr": result.mrr,
            "citation_accuracy": result.citation_accuracy,
            "faithfulness": result.faithfulness,
            "permission_leaks": result.permission_leaks,
            "scenarios": result.scenario_summary,
            "holdout": {
                "total": holdout.total,
                "recall_at_k": holdout.recall_at_k,
                "mrr": holdout.mrr,
                "citation_accuracy": holdout.citation_accuracy,
                "faithfulness": holdout.faithfulness,
                "permission_leaks": holdout.permission_leaks,
                "scenarios": holdout.scenario_summary,
            },
            "retrieval_experiment": experiment.strategies if experiment else [],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        failures = _threshold_failures(args, summary)
        failures.extend(f"holdout {item}" for item in _threshold_failures(args, summary["holdout"]))
        if failures:
            print(json.dumps({"ok": False, "failures": failures, "result": asdict(result)}, ensure_ascii=False, indent=2))
            raise SystemExit(1)
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)


def _threshold_failures(args: argparse.Namespace, summary: dict[str, float | int]) -> list[str]:
    checks = [
        ("recall_at_k", summary["recall_at_k"], ">=", args.min_recall),
        ("mrr", summary["mrr"], ">=", args.min_mrr),
        ("citation_accuracy", summary["citation_accuracy"], ">=", args.min_citation_accuracy),
        ("faithfulness", summary["faithfulness"], ">=", args.min_faithfulness),
        ("permission_leaks", summary["permission_leaks"], "<=", args.max_permission_leaks),
    ]
    failures = []
    for name, actual, operator, expected in checks:
        failed = actual < expected if operator == ">=" else actual > expected
        if failed:
            failures.append(f"{name}={actual} expected {operator} {expected}")
    return failures


if __name__ == "__main__":
    main()

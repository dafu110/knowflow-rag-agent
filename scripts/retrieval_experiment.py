from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from knowflow.agent import RagAgent
from knowflow.chunking import load_documents_from_path
from knowflow.evaluation import compare_retrieval_strategies
from knowflow.store_factory import create_store


def main() -> None:
    _configure_stdout()
    workdir = Path(tempfile.mkdtemp(prefix="knowflow-retrieval-experiment-"))
    try:
        store = create_store(workdir, backend="jsonl")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        experiment = compare_retrieval_strategies(RagAgent(store), Path("evals/rag_eval_set.jsonl"))
        print(json.dumps({"corpus": "sample_docs", "eval_set": "evals/rag_eval_set.jsonl", "strategies": experiment.strategies}, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()

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
from knowflow.models import Principal
from knowflow.store_factory import create_store


CASES = [
    {
        "name": "sales_allowed",
        "question": "销售合同审批需要哪些材料？",
        "principal": Principal(user="alice", roles={"sales"}),
        "expected_type": "grounded",
    },
    {
        "name": "security_denied_to_sales",
        "question": "客户敏感数据的临时授权需要记录哪些信息？",
        "principal": Principal(user="alice", roles={"sales"}),
        "expected_type": "refusal",
    },
    {
        "name": "security_allowed",
        "question": "客户敏感数据的临时授权需要记录哪些信息？",
        "principal": Principal(user="ciso", roles={"security"}),
        "expected_type": "grounded",
    },
]


def main() -> None:
    _configure_stdout()
    store_dir = Path(tempfile.mkdtemp(prefix="knowflow-demo-"))
    try:
        store = create_store(str(store_dir), backend="jsonl")
        store.add_documents(load_documents_from_path(Path("sample_docs")))
        agent = RagAgent(store)
        results = []
        failures = []
        for case in CASES:
            answer = agent.ask(case["question"], principal=case["principal"], session_id=f"demo-{case['name']}")
            row = {
                "name": case["name"],
                "question": case["question"],
                "answer_type": answer.answer_type,
                "confidence": answer.confidence,
                "citations": [citation.source for citation in answer.citations],
                "answer": answer.answer,
            }
            results.append(row)
            if answer.answer_type != case["expected_type"]:
                failures.append(
                    f"{case['name']} expected {case['expected_type']} got {answer.answer_type}"
                )
        print(json.dumps({"ok": not failures, "failures": failures, "cases": results}, ensure_ascii=False, indent=2))
        if failures:
            raise SystemExit(1)
    finally:
        shutil.rmtree(store_dir, ignore_errors=True)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    main()

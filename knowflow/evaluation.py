from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import RagAgent
from .models import Principal
from .retrieval import tokenize


@dataclass(slots=True)
class EvalResult:
    total: int
    recall_at_k: float
    mrr: float
    citation_accuracy: float
    faithfulness: float
    permission_leaks: int
    scenario_summary: dict[str, dict[str, int]]
    cases: list[dict[str, Any]]


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def evaluate(agent: RagAgent, eval_path: Path, top_k: int = 6) -> EvalResult:
    cases = load_eval_set(eval_path)
    details: list[dict[str, Any]] = []
    recall_hits = 0
    reciprocal_ranks = 0.0
    citation_hits = 0
    retrieval_cases = 0
    citation_cases = 0
    faithfulness_scores = 0.0
    permission_leaks = 0
    scenario_summary: dict[str, dict[str, int]] = {}
    for case in cases:
        scenario = str(case.get("scenario", "core"))
        scenario_row = scenario_summary.setdefault(scenario, {"total": 0, "passed": 0, "permission_leaks": 0})
        scenario_row["total"] += 1
        principal = Principal(user=case.get("user", "eval"), roles=set(case.get("roles", [])))
        answer = agent.ask(case["question"], principal=principal, session_id=f"eval-{len(details)}", top_k=top_k)
        expected_sources = set(case.get("expected_sources", []))
        retrieved_sources = [Path(item["source"]).name for item in answer.retrieval_debug]
        cited_sources = [Path(citation.source).name for citation in answer.citations]
        first_rank = _first_rank(retrieved_sources, expected_sources)
        if expected_sources:
            retrieval_cases += 1
            citation_cases += 1
            if first_rank is not None:
                recall_hits += 1
                reciprocal_ranks += 1.0 / first_rank
            if expected_sources.intersection(cited_sources):
                citation_hits += 1
        allowed_unsupported = {"no_retrieved_evidence", "needs_clarification"}
        unsupported_ok = not set(answer.unsupported_claims) - allowed_unsupported
        if expected_sources:
            faithfulness = 1.0 if unsupported_ok else 0.0
        else:
            faithfulness = 1.0 if unsupported_ok and answer.answer_type in {"refusal", "clarify"} and not cited_sources else 0.0
        faithfulness_scores += faithfulness
        forbidden = set(case.get("forbidden_sources", []))
        if forbidden.intersection(retrieved_sources) or forbidden.intersection(cited_sources):
            permission_leaks += 1
            scenario_row["permission_leaks"] += 1
        expected_terms = set(tokenize(" ".join(case.get("expected_terms", []))))
        answer_terms = set(tokenize(answer.answer))
        term_coverage = len(expected_terms.intersection(answer_terms)) / max(len(expected_terms), 1)
        passed = bool(faithfulness) and not bool(forbidden.intersection(retrieved_sources) or forbidden.intersection(cited_sources))
        if expected_sources:
            passed = passed and first_rank is not None and bool(expected_sources.intersection(cited_sources))
        if passed:
            scenario_row["passed"] += 1
        details.append(
            {
                "scenario": scenario,
                "question": case["question"],
                "answer": answer.answer,
                "confidence": answer.confidence,
                "hallucination_risk": answer.hallucination_risk,
                "retrieved_sources": retrieved_sources,
                "cited_sources": cited_sources,
                "recall_hit": first_rank is not None,
                "first_rank": first_rank,
                "term_coverage": round(term_coverage, 3),
                "answer_type": answer.answer_type,
                "faithful": bool(faithfulness),
                "permission_leak": bool(forbidden.intersection(retrieved_sources) or forbidden.intersection(cited_sources)),
            }
        )
    retrieval_total = max(retrieval_cases, 1)
    citation_total = max(citation_cases, 1)
    total = max(len(cases), 1)
    return EvalResult(
        total=len(cases),
        recall_at_k=round(recall_hits / retrieval_total, 3),
        mrr=round(reciprocal_ranks / retrieval_total, 3),
        citation_accuracy=round(citation_hits / citation_total, 3),
        faithfulness=round(faithfulness_scores / total, 3),
        permission_leaks=permission_leaks,
        scenario_summary=scenario_summary,
        cases=details,
    )


def _first_rank(retrieved_sources: list[str], expected_sources: set[str]) -> int | None:
    if not expected_sources:
        return None
    for index, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return index
    return None

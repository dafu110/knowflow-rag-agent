from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Principal:
    user: str = "anonymous"
    roles: set[str] = field(default_factory=set)


@dataclass(slots=True)
class Document:
    id: str
    title: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_roles: set[str] = field(default_factory=set)
    allowed_users: set[str] = field(default_factory=set)
    created_at: str = field(default_factory=utc_now)

    def is_visible_to(self, principal: Principal) -> bool:
        if not self.allowed_roles and not self.allowed_users:
            return True
        if principal.user in self.allowed_users:
            return True
        return bool(self.allowed_roles.intersection(principal.roles))


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    title: str
    source: str
    text: str
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_roles: set[str] = field(default_factory=set)
    allowed_users: set[str] = field(default_factory=set)
    ordinal: int = 0
    created_at: str = field(default_factory=utc_now)

    def is_visible_to(self, principal: Principal) -> bool:
        if not self.allowed_roles and not self.allowed_users:
            return True
        if principal.user in self.allowed_users:
            return True
        return bool(self.allowed_roles.intersection(principal.roles))


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    bm25_score: float
    vector_score: float
    rerank_score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Citation:
    chunk_id: str
    document_id: str
    source: str
    title: str
    quote: str


@dataclass(slots=True)
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    hallucination_risk: str
    unsupported_claims: list[str]
    retrieval_debug: list[dict[str, Any]]
    answer_type: str = "grounded"
    evidence_summary: str = ""
    follow_up_questions: list[str] = field(default_factory=list)
    session_id: str | None = None

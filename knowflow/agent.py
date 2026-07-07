from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Answer, Citation, Principal, RetrievedChunk
from .providers import EvidenceComposer, composer_from_env, embedding_provider_from_env, reranker_from_env
from .retrieval import HybridRetriever, expand_query_tokens, tokenize
from .store import KnowledgeStore


@dataclass(slots=True)
class Turn:
    question: str
    answer: str
    answer_type: str = "grounded"
    evidence_summary: str = ""
    follow_up_questions: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)


class ConversationMemory:
    def __init__(self, max_turns: int = 4) -> None:
        self.max_turns = max_turns
        self.sessions: dict[str, list[Turn]] = {}

    def context_for(self, session_id: str | None) -> list[Turn]:
        if not session_id:
            return []
        return self.sessions.get(session_id, [])[-self.max_turns :]

    def add(self, session_id: str | None, turn: Turn) -> None:
        if not session_id:
            return
        turns = self.sessions.setdefault(session_id, [])
        turns.append(turn)
        del turns[: max(0, len(turns) - self.max_turns)]


class RagAgent:
    def __init__(
        self,
        store: KnowledgeStore,
        memory: ConversationMemory | None = None,
        composer: EvidenceComposer | None = None,
    ) -> None:
        self.store = store
        self.memory = memory or ConversationMemory()
        self.embedding_provider = embedding_provider_from_env()
        self.reranker = reranker_from_env()
        self.composer = composer if composer is not None else composer_from_env()

    def ask(
        self,
        question: str,
        principal: Principal | None = None,
        session_id: str | None = None,
        top_k: int = 6,
    ) -> Answer:
        principal = principal or Principal()
        contextual_query = self._contextualize(question, session_id)
        retriever = HybridRetriever(self.store.chunks(), embedding_provider=self.embedding_provider, reranker=self.reranker)
        retrieved = retriever.search(contextual_query, principal=principal, top_k=top_k)
        answer = self._compose_answer(question, retrieved, session_id)
        self.memory.add(
            session_id,
            Turn(
                question=question,
                answer=answer.answer,
                answer_type=answer.answer_type,
                evidence_summary=answer.evidence_summary,
                follow_up_questions=answer.follow_up_questions,
                cited_chunk_ids=[citation.chunk_id for citation in answer.citations],
            ),
        )
        return answer

    def _contextualize(self, question: str, session_id: str | None) -> str:
        history = self.memory.context_for(session_id)
        if not history:
            return question
        if len(question) >= 18 or _has_explicit_topic(question) or _looks_self_contained(question):
            return question
        previous = history[-1]
        parts = [previous.question, previous.answer]
        if previous.evidence_summary:
            parts.append(f"相关主题：{previous.evidence_summary}")
        if previous.follow_up_questions:
            parts.append("上轮追问：" + "；".join(previous.follow_up_questions[:2]))
        parts.append(f"追问：{question}")
        return "\n".join(parts)

    def _compose_answer(self, question: str, retrieved: list[RetrievedChunk], session_id: str | None) -> Answer:
        question_style = _question_style(question)
        strong_evidence = _strong_evidence(retrieved)
        if _needs_clarification(question, strong_evidence):
            return _clarify_answer(question, retrieved, session_id, strong_evidence, question_style)
        if not strong_evidence or not _intent_supported(question, strong_evidence):
            return _no_evidence_answer(question, retrieved, session_id, question_style)

        citations = [_citation_from_result(result) for result in strong_evidence[:4]]
        evidence_sentences = _best_sentences(question, strong_evidence, limit=5)
        if not evidence_sentences:
            evidence_sentences = [_trim(result.chunk.text, 220) for result in strong_evidence[:2]]
        answer_text = self._format_answer_with_composer(question, evidence_sentences, citations)
        unsupported_claims = _unsupported_claims(answer_text, strong_evidence)
        top_score = strong_evidence[0].score
        confidence = min(0.95, round(0.45 + top_score * 0.50 - 0.08 * len(unsupported_claims), 2))
        risk = "low" if not unsupported_claims and confidence >= 0.70 else "medium"
        if unsupported_claims:
            risk = "high" if confidence < 0.55 else "medium"

        evidence_summary = _evidence_summary(strong_evidence)
        return Answer(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=max(confidence, 0.0),
            hallucination_risk=risk,
            unsupported_claims=unsupported_claims,
            retrieval_debug=_debug_rows(retrieved),
            answer_type="grounded",
            evidence_summary=evidence_summary,
            follow_up_questions=_follow_up_questions(question, strong_evidence, question_style),
            session_id=session_id,
        )

    def _format_answer_with_composer(self, question: str, evidence_sentences: list[str], citations: list[Citation]) -> str:
        if not self.composer:
            return _format_answer(evidence_sentences, citations)
        try:
            return self.composer.compose(question, evidence_sentences, citations)
        except RuntimeError:
            return _format_answer(evidence_sentences, citations)


def _no_evidence_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    session_id: str | None,
    question_style: str,
) -> Answer:
    return Answer(
        question=question,
        answer="我没有在当前有权限访问的知识库中找到可靠依据，暂时不能给出确定答案。",
        citations=[],
        confidence=0.0,
        hallucination_risk="high",
        unsupported_claims=["no_retrieved_evidence"],
        retrieval_debug=_debug_rows(retrieved),
        answer_type="refusal",
        evidence_summary="",
        follow_up_questions=_clarification_prompts(question_style),
        session_id=session_id,
    )


def _clarify_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    session_id: str | None,
    strong_evidence: list[RetrievedChunk],
    question_style: str,
) -> Answer:
    prompts = _clarification_prompts(question_style, strong_evidence)
    topic = _evidence_topic(strong_evidence)
    answer_text = "这个问题还可以再收窄一点。你可以告诉我你想查的是哪类制度、哪份文档，或者具体流程节点。"
    if topic:
        answer_text = f"我能找到相关资料，但问题还偏宽。你想优先看{topic}里的哪一部分？"
    return Answer(
        question=question,
        answer=answer_text,
        citations=[_citation_from_result(result) for result in strong_evidence[:2]],
        confidence=0.45,
        hallucination_risk="medium",
        unsupported_claims=["needs_clarification"],
        retrieval_debug=_debug_rows(retrieved),
        answer_type="clarify",
        evidence_summary=_evidence_summary(strong_evidence),
        follow_up_questions=prompts,
        session_id=session_id,
    )


def _intent_supported(question: str, strong_evidence: list[RetrievedChunk]) -> bool:
    topic_terms = ("销售合同", "报销制度", "SLA", "客户敏感数据", "客户明细", "客户密钥", "权限回收")
    for term in topic_terms:
        if term in question and not any(term.lower() in _result_haystack(result) for result in strong_evidence):
            return False
    sensitive_terms = ("敏感数据", "客户数据", "临时授权", "密钥", "安全事件", "权限回收", "数据库", "密码", "root", "备份")
    if not any(term in question for term in sensitive_terms):
        return True
    security_hints = ("security", "access", "权限", "安全", "客户数据", "授权")
    for result in strong_evidence:
        haystack = _result_haystack(result)
        if any(hint.lower() in haystack for hint in security_hints):
            return True
    return False


def _result_haystack(result: RetrievedChunk) -> str:
    return f"{result.chunk.source} {result.chunk.title} {result.chunk.text}".lower()


def _strong_evidence(retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not retrieved or retrieved[0].score < 0.12:
        return []
    top_score = retrieved[0].score
    return [result for result in retrieved if _is_strong_evidence(result, top_score)]


def _debug_rows(retrieved: list[RetrievedChunk]) -> list[dict[str, object]]:
    top_score = retrieved[0].score if retrieved else 0.0
    return [
        {
            "chunk_id": result.chunk.id,
            "source": result.chunk.source,
            "score": round(result.score, 4),
            "bm25": round(result.bm25_score, 4),
            "vector": round(result.vector_score, 4),
            "rerank": round(result.rerank_score, 4),
            "evidence_grade": "strong" if _is_strong_evidence(result, top_score) else "weak",
            "reasons": result.reasons,
        }
        for result in retrieved
    ]


def _is_strong_evidence(result: RetrievedChunk, top_score: float) -> bool:
    return result.rerank_score >= 0.30 and result.score >= max(0.14, top_score * 0.35)


def _citation_from_result(result: RetrievedChunk) -> Citation:
    return Citation(
        chunk_id=result.chunk.id,
        document_id=result.chunk.document_id,
        source=result.chunk.source,
        title=result.chunk.title,
        quote=_trim(result.chunk.text, 180),
    )


def _best_sentences(question: str, retrieved: list[RetrievedChunk], limit: int) -> list[str]:
    query_tokens = set(expand_query_tokens(question))
    candidates: list[tuple[float, str]] = []
    for rank, result in enumerate(retrieved):
        for sentence in _sentences(result.chunk.text):
            sentence_tokens = set(tokenize(sentence))
            if not sentence_tokens:
                continue
            overlap = len(query_tokens.intersection(sentence_tokens)) / max(len(query_tokens), 1)
            score = overlap + _sentence_intent_boost(question, sentence) + result.score * 0.25 - rank * 0.03
            if overlap > 0:
                candidates.append((score, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen = set()
    for _, sentence in candidates:
        normalized = re.sub(r"\s+", "", sentence.lower())
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def _sentence_intent_boost(question: str, sentence: str) -> float:
    rules = [
        (("材料", "资料", "附件", "清单"), ("材料", "准备", "提供", "附件", "清单", "正文", "报价单"), 0.35),
        (("谁", "由谁", "负责人"), ("负责人", "经理", "专员", "审批", "复核"), 0.25),
        (("时间", "多久", "SLA", "响应", "恢复"), ("时间", "分钟", "小时", "工作日", "响应", "恢复"), 0.30),
        (("原因", "为什么", "退回"), ("原因", "因为", "不一致", "错误", "缺少"), 0.25),
        (("记录", "授权", "权限"), ("记录", "授权", "申请人", "审批人", "范围", "开始时间", "结束时间"), 0.30),
    ]
    boost = 0.0
    for query_terms, sentence_terms, value in rules:
        if any(term in question for term in query_terms) and any(term in sentence for term in sentence_terms):
            boost += value
    return min(boost, 0.5)


def _follow_up_questions(question: str, retrieved: list[RetrievedChunk], question_style: str) -> list[str]:
    prompts = _clarification_prompts(question_style, retrieved[:2])
    if _needs_clarification(question, retrieved):
        return prompts[:3]
    return prompts[:2]


def _clarification_prompts(question_style: str, retrieved: list[RetrievedChunk] | None = None) -> list[str]:
    retrieved = retrieved or []
    topic = _evidence_topic(retrieved)
    if question_style == "time":
        return [
            f"{topic} 的响应时间和恢复目标分别是什么？".strip(),
            f"{topic} 触发升级的条件是什么？".strip(),
        ]
    if question_style == "list":
        return [
            f"{topic} 需要准备哪些材料？".strip(),
            f"{topic} 的审批链路是什么？".strip(),
        ]
    if question_style == "who":
        return [
            f"{topic} 由谁审批？".strip(),
            f"{topic} 还需要哪些角色参与？".strip(),
        ]
    if question_style == "reason":
        return [
            f"{topic} 常见退回原因有哪些？".strip(),
            f"{topic} 如何避免被退回？".strip(),
        ]
    if topic:
        return [f"{topic} 的具体要求是什么？", f"{topic} 相关流程怎么走？"]
    return ["你想问的是哪份制度或哪类流程？", "你希望先看材料、责任人还是时间要求？"]


def _evidence_summary(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return ""
    top = retrieved[0]
    section = " / ".join(top.chunk.section_path) if top.chunk.section_path else top.chunk.title
    return f"{top.chunk.title} · {section}"


def _evidence_topic(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return ""
    top = retrieved[0].chunk
    if top.section_path:
        return top.section_path[-1]
    return top.title


def _question_style(question: str) -> str:
    if any(token in question for token in ("材料", "资料", "附件", "清单")):
        return "list"
    if any(token in question for token in ("谁", "由谁", "负责人")):
        return "who"
    if any(token in question for token in ("时间", "多久", "SLA", "响应", "恢复")):
        return "time"
    if any(token in question for token in ("原因", "为什么", "退回")):
        return "reason"
    return "general"


def _needs_clarification(question: str, retrieved: list[RetrievedChunk]) -> bool:
    stripped = question.strip()
    short_question = len(stripped) <= 8
    generic = not _has_explicit_topic(question) and not any(
        token in question for token in ("材料", "时间", "谁", "原因", "权限", "报销", "合同", "SLA")
    )
    weak_context = len(retrieved) >= 2 and retrieved[0].score < 0.55 and retrieved[0].rerank_score < 0.5
    return short_question and (generic or weak_context or not retrieved)


def _looks_self_contained(question: str) -> bool:
    return any(token in question for token in ("材料", "时间", "谁", "原因", "报销", "合同", "SLA", "权限", "审批"))


def _format_answer(sentences: list[str], citations: list[Citation]) -> str:
    cited = ", ".join(f"[{citation.chunk_id}]" for citation in citations[:3])
    summary = " ".join(_trim(sentence, 220) for sentence in sentences)
    return f"{summary}\n\n依据：{cited}"


def _unsupported_claims(answer_text: str, retrieved: list[RetrievedChunk]) -> list[str]:
    evidence_tokens = set()
    for result in retrieved:
        evidence_tokens.update(tokenize(result.chunk.text))
    unsupported: list[str] = []
    for sentence in _sentences(answer_text.split("依据：", 1)[0]):
        tokens = set(tokenize(sentence))
        if not tokens:
            continue
        overlap = len(tokens.intersection(evidence_tokens)) / max(len(tokens), 1)
        if overlap < 0.35:
            unsupported.append(sentence)
    return unsupported


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    sentences: list[str] = []
    for part in parts:
        stripped = part.strip(" -\t")
        if 8 <= len(stripped) <= 260:
            sentences.append(stripped)
    return sentences


def _has_explicit_topic(question: str) -> bool:
    return any(word in question for word in ("审批", "报销", "权限", "合同", "客户", "故障", "SLA", "知识库"))


def _trim(text: str, max_len: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"

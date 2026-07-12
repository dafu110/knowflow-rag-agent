from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .models import Chunk, Principal, RetrievedChunk
from .providers import EmbeddingProvider, Reranker


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
QUERY_STOP_TOKENS = {
    "哪些",
    "什么",
    "怎么",
    "如何",
    "是否",
    "需要",
    "可以",
    "一下",
    "信息",
    "内容",
    "情况",
    "说明",
}
DOMAIN_SYNONYMS = {
    "材料": ["资料", "附件", "文件", "清单"],
    "审批": ["审核", "批准", "复核", "负责人"],
    "合同": ["协议", "条款", "签约"],
    "报销": ["费用", "发票", "差旅", "付款"],
    "故障": ["事故", "不可用", "恢复", "响应"],
    "权限": ["授权", "访问", "角色", "密级"],
    "客户": ["企业客户", "客户主体", "客户数据"],
    "SLA": ["响应时间", "恢复目标", "升级机制"],
    "p0": ["生产系统", "大面积不可用", "15分钟", "4小时"],
}
RETRIEVAL_STRATEGIES = ("bm25", "vector", "hybrid", "rerank")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(_char_ngrams(token))
        else:
            tokens.append(token)
    return [token for token in tokens if len(token.strip()) > 0]


def expand_query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    expanded = list(tokens)
    lowered = query.lower()
    for term, synonyms in DOMAIN_SYNONYMS.items():
        term_key = term.lower()
        if term in query or term_key in lowered:
            expanded.extend(tokenize(" ".join(synonyms)))
    return _filter_query_noise(expanded)


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks = chunks
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.doc_tokens = [tokenize(_chunk_text(chunk)) for chunk in chunks]
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs = self._document_frequencies()
        self.avg_doc_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.idf = {
            term: math.log((len(self.chunks) - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.doc_freqs.items()
        }
        self._chunk_embeddings: list[list[float]] | None = None
        self._embedding_unavailable = False

    def search(
        self,
        query: str,
        principal: Principal,
        top_k: int = 6,
        strategy: str = "rerank",
    ) -> list[RetrievedChunk]:
        if strategy not in RETRIEVAL_STRATEGIES:
            raise ValueError(f"unsupported retrieval strategy: {strategy}")
        if not query.strip() or not self.chunks:
            return []
        visible_indices = [i for i, chunk in enumerate(self.chunks) if chunk.is_visible_to(principal)]
        query_tokens = expand_query_tokens(query)
        if not query_tokens:
            return []
        raw_results: list[tuple[int, float, float]] = []
        query_vector = self._tfidf_vector(query_tokens)
        semantic_scores = self._semantic_scores(query, visible_indices)
        for index in visible_indices:
            bm25_score = self._bm25(query_tokens, index)
            vector_score = semantic_scores.get(index)
            if vector_score is None:
                vector_score = self._cosine(query_vector, self._tfidf_vector(self.doc_tokens[index]))
            if bm25_score > 0 or vector_score > 0:
                raw_results.append((index, bm25_score, vector_score))
        if not raw_results:
            return []
        max_bm25 = max(score for _, score, _ in raw_results) or 1.0
        max_vector = max(score for _, _, score in raw_results) or 1.0
        results: list[RetrievedChunk] = []
        for index, bm25_score, vector_score in raw_results:
            chunk = self.chunks[index]
            bm25_norm = bm25_score / max_bm25
            vector_norm = vector_score / max_vector
            rerank_score, reasons = self._rerank(query, query_tokens, chunk)
            score = _strategy_score(strategy, bm25_norm, vector_norm, rerank_score)
            reasons.append(f"strategy={strategy}")
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    bm25_score=bm25_norm,
                    vector_score=vector_norm,
                    rerank_score=rerank_score,
                    reasons=reasons,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        if strategy == "rerank":
            self._apply_external_rerank(query, results)
        return results[:top_k]

    def _document_frequencies(self) -> dict[str, int]:
        doc_freqs: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                doc_freqs[token] += 1
        return dict(doc_freqs)

    def _bm25(self, query_tokens: list[str], index: int, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        freqs = self.term_freqs[index]
        doc_len = len(self.doc_tokens[index]) or 1
        for token in query_tokens:
            tf = freqs.get(token, 0)
            if tf == 0:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1))
            score += idf * ((tf * (k1 + 1)) / denom)
        return score

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        freqs = Counter(tokens)
        length = len(tokens) or 1
        return {term: (count / length) * self.idf.get(term, 0.1) for term, count in freqs.items()}

    def _semantic_scores(self, query: str, visible_indices: list[int]) -> dict[int, float]:
        if not self.embedding_provider or not visible_indices:
            return {}
        self.warm_embeddings()
        if not self._chunk_embeddings:
            return {}
        try:
            vectors = self.embedding_provider.embed([query])
        except RuntimeError:
            return {}
        if len(vectors) != 1:
            return {}
        query_vector = vectors[0]
        scores = {
            index: _cosine_dense(query_vector, self._chunk_embeddings[index])
            for index in visible_indices
        }
        max_score = max(scores.values()) if scores else 0.0
        if max_score <= 0:
            return scores
        return {index: score / max_score for index, score in scores.items()}

    def warm_embeddings(self) -> None:
        if not self.embedding_provider or self._chunk_embeddings is not None or self._embedding_unavailable:
            return
        try:
            vectors = self.embedding_provider.embed([_chunk_text(chunk) for chunk in self.chunks])
        except RuntimeError:
            self._embedding_unavailable = True
            return
        if len(vectors) != len(self.chunks):
            self._embedding_unavailable = True
            return
        self._chunk_embeddings = vectors

    def _apply_external_rerank(self, query: str, results: list[RetrievedChunk]) -> None:
        if not self.reranker or not results:
            return
        try:
            scores = self.reranker.rerank(query, results)
        except RuntimeError:
            return
        if len(scores) != len(results):
            return
        max_score = max(scores) if scores else 0.0
        normalized = [score / max_score if max_score > 0 else 0.0 for score in scores]
        for result, rerank_score in zip(results, normalized):
            result.rerank_score = max(result.rerank_score, rerank_score)
            result.score = 0.80 * result.score + 0.20 * rerank_score
            result.reasons.append(f"{self.reranker.name}={rerank_score:.2f}")
        results.sort(key=lambda item: item.score, reverse=True)

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        common = set(left).intersection(right)
        numerator = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _rerank(self, query: str, query_tokens: list[str], chunk: Chunk) -> tuple[float, list[str]]:
        chunk_text = _chunk_text(chunk).lower()
        chunk_tokens = set(tokenize(chunk_text))
        query_unique = set(query_tokens)
        matched_terms = query_unique.intersection(chunk_tokens)
        coverage = len(matched_terms) / max(len(query_unique), 1)
        score = coverage
        reasons = [f"term_coverage={coverage:.2f}", f"matched_terms={min(len(matched_terms), 12)}"]
        if query.lower() in chunk_text:
            score += 0.25
            reasons.append("exact_phrase")
        title_hits = len(query_unique.intersection(set(tokenize(chunk.title))))
        if title_hits:
            score += min(0.20, 0.06 * title_hits)
            reasons.append("title_match")
        if _has_nearby_terms(query_tokens, tokenize(chunk.text)):
            score += 0.15
            reasons.append("nearby_terms")
        freshness = _freshness_bonus(chunk.created_at)
        if freshness:
            score += freshness
            reasons.append("freshness")
        return min(score, 1.0), reasons


def _chunk_text(chunk: Chunk) -> str:
    section = " > ".join(chunk.section_path)
    return f"{chunk.title}\n{section}\n{chunk.text}"


def _strategy_score(strategy: str, bm25: float, vector: float, rerank: float) -> float:
    if strategy == "bm25":
        return bm25
    if strategy == "vector":
        return vector
    if strategy == "hybrid":
        return 0.55 * bm25 + 0.45 * vector
    return 0.45 * bm25 + 0.35 * vector + 0.20 * rerank


def _char_ngrams(text: str) -> list[str]:
    if len(text) <= 2:
        return [text]
    grams = set()
    for size in (2, 3):
        for i in range(0, len(text) - size + 1):
            grams.add(text[i : i + size])
    grams.add(text)
    return list(grams)


def _filter_query_noise(tokens: list[str]) -> list[str]:
    filtered = [token for token in tokens if token not in QUERY_STOP_TOKENS]
    return filtered or tokens


def _has_nearby_terms(query_tokens: list[str], doc_tokens: list[str], window: int = 16) -> bool:
    query_set = set(query_tokens)
    positions = [index for index, token in enumerate(doc_tokens) if token in query_set]
    if len(positions) < 2:
        return False
    return any(right - left <= window for left, right in zip(positions, positions[1:]))


def _freshness_bonus(created_at: str) -> float:
    try:
        created = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    age_days = (datetime.now(timezone.utc) - created).days
    if age_days < 90:
        return 0.05
    if age_days < 365:
        return 0.02
    return 0.0


def _cosine_dense(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)

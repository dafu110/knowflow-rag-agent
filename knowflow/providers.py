from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Callable, Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import load_model_settings, model_api_key, rerank_api_key
from .models import Citation, RetrievedChunk


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, results: list[RetrievedChunk]) -> list[float]:
        ...


class EvidenceComposer(Protocol):
    name: str

    def compose(self, question: str, evidence: list[str], citations: list[Citation]) -> str:
        ...


class ProviderTelemetry:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0
        self.failures = 0
        self.degradations = 0
        self.retries = 0
        self.total_latency_ms = 0.0
        self.last_error = ""
        self.open_until = 0.0

    def record_success(self, latency_ms: float) -> None:
        self.calls += 1
        self.total_latency_ms += latency_ms
        self.last_error = ""
        self.open_until = 0.0

    def record_failure(self, latency_ms: float, error: Exception) -> None:
        self.calls += 1
        self.failures += 1
        self.degradations += 1
        self.total_latency_ms += latency_ms
        self.last_error = type(error).__name__
        if self.failures >= 3:
            self.open_until = time.monotonic() + 30.0

    def status(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "degradations": self.degradations,
            "retries": self.retries,
            "avg_latency_ms": round(self.total_latency_ms / self.calls, 1) if self.calls else 0.0,
            "circuit": "open" if time.monotonic() < self.open_until else "closed",
            "last_error": self.last_error or None,
        }


class ReliableProvider:
    def __init__(self, name: str) -> None:
        self.telemetry = ProviderTelemetry(name)

    def status(self) -> dict[str, object]:
        return self.telemetry.status()

    def _request(self, call: Callable[[], dict[str, object]]) -> dict[str, object]:
        if time.monotonic() < self.telemetry.open_until:
            self.telemetry.degradations += 1
            raise RuntimeError("provider circuit is open")
        start = time.monotonic()
        last_error: RuntimeError | None = None
        for attempt in range(2):
            try:
                result = call()
                self.telemetry.record_success((time.monotonic() - start) * 1000)
                return result
            except RuntimeError as error:
                last_error = error
                if attempt == 0:
                    self.telemetry.retries += 1
        assert last_error is not None
        self.telemetry.record_failure((time.monotonic() - start) * 1000, last_error)
        raise last_error


class OpenAIEmbeddingProvider(ReliableProvider):
    name = "openai_embeddings"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 20.0) -> None:
        super().__init__(self.name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        data = self._request(lambda: _post_json(f"{self.base_url}/embeddings", payload, self.api_key, self.timeout))
        rows = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [list(map(float, row["embedding"])) for row in rows]


class OpenAIChatComposer(ReliableProvider):
    name = "openai_chat"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 20.0) -> None:
        super().__init__(self.name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def compose(self, question: str, evidence: list[str], citations: list[Citation]) -> str:
        evidence_block = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(evidence))
        citation_ids = ", ".join(f"[{citation.chunk_id}]" for citation in citations[:3])
        messages = [
            {
                "role": "system",
                "content": (
                    "你是企业知识库 RAG 助手。只能依据给定证据回答。"
                    "不要编造证据外的信息。回答必须简洁，并保留关键事实。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n证据：\n{evidence_block}\n\n请用中文回答，并在末尾写：依据：{citation_ids}",
            },
        ]
        payload = {"model": self.model, "messages": messages, "temperature": 0.1}
        data = self._request(lambda: _post_json(f"{self.base_url}/chat/completions", payload, self.api_key, self.timeout))
        content = data["choices"][0]["message"]["content"].strip()
        if "依据：" not in content:
            content = f"{content}\n\n依据：{citation_ids}"
        return content


class ExternalReranker(ReliableProvider):
    name = "external_reranker"

    def __init__(self, url: str, api_key: str = "", model: str = "", timeout: float = 20.0) -> None:
        super().__init__(self.name)
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, results: list[RetrievedChunk]) -> list[float]:
        payload = {
            "query": query,
            "model": self.model,
            "documents": [
                {
                    "index": index,
                    "text": result.chunk.text,
                    "metadata": {"source": result.chunk.source, "title": result.chunk.title},
                }
                for index, result in enumerate(results)
            ],
        }
        data = self._request(lambda: _post_json(self.url, payload, self.api_key, self.timeout))
        if isinstance(data.get("scores"), list):
            return [float(score) for score in data["scores"]]
        scores = [0.0 for _ in results]
        for item in data.get("results", []):
            scores[int(item["index"])] = float(item["score"])
        return scores


def embedding_provider_from_env() -> EmbeddingProvider | None:
    settings = load_model_settings()
    key = model_api_key()
    if settings.embedding_provider in {"openai", "openai-compatible"} and key:
        return OpenAIEmbeddingProvider(key, settings.embedding_base_url, settings.embedding_model, settings.request_timeout)
    return None


def composer_from_env() -> EvidenceComposer | None:
    settings = load_model_settings()
    key = model_api_key()
    if settings.llm_provider in {"openai", "openai-compatible"} and key:
        return OpenAIChatComposer(key, settings.llm_base_url, settings.llm_model, settings.request_timeout)
    return None


def reranker_from_env() -> Reranker | None:
    settings = load_model_settings()
    if settings.rerank_url:
        return ExternalReranker(settings.rerank_url, rerank_api_key(), settings.rerank_model, settings.request_timeout)
    return None


def _post_json(url: str, payload: dict[str, object], api_key: str = "", timeout: float = 20.0) -> dict[str, object]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as error:
        raise RuntimeError(f"model provider request failed: {error}") from error


def citation_dicts(citations: list[Citation]) -> list[dict[str, object]]:
    return [asdict(citation) for citation in citations]

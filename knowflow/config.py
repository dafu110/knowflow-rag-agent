from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSettings:
    embedding_provider: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    llm_provider: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    rerank_url: str = ""
    rerank_model: str = ""
    request_timeout: float = 20.0


def load_model_settings() -> ModelSettings:
    return ModelSettings(
        embedding_provider=os.environ.get("KNOWFLOW_EMBEDDING_PROVIDER", "").strip().lower(),
        embedding_model=os.environ.get("KNOWFLOW_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        embedding_base_url=os.environ.get("KNOWFLOW_EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        llm_provider=os.environ.get("KNOWFLOW_LLM_PROVIDER", "").strip().lower(),
        llm_model=os.environ.get("KNOWFLOW_LLM_MODEL", "gpt-4.1-mini").strip(),
        llm_base_url=os.environ.get("KNOWFLOW_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        rerank_url=os.environ.get("KNOWFLOW_RERANK_URL", "").strip(),
        rerank_model=os.environ.get("KNOWFLOW_RERANK_MODEL", "").strip(),
        request_timeout=float(os.environ.get("KNOWFLOW_MODEL_TIMEOUT", "20")),
    )


def model_api_key() -> str:
    return os.environ.get("KNOWFLOW_MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY", "")


def rerank_api_key() -> str:
    return os.environ.get("KNOWFLOW_RERANK_API_KEY", "")

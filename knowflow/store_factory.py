from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import Chunk, Document
from .sqlite_store import SQLiteKnowledgeStore
from .store import KnowledgeStore


class Store(Protocol):
    def reset(self) -> None:
        ...

    def add_documents(self, documents: list[Document]) -> int:
        ...

    def delete_document(self, document_id: str) -> bool:
        ...

    def documents(self) -> list[Document]:
        ...

    def chunks(self) -> list[Chunk]:
        ...

    def stats(self) -> dict[str, int]:
        ...


def create_store(root: str | Path, backend: str = "jsonl") -> Store:
    backend = backend.strip().lower()
    if backend == "jsonl":
        return KnowledgeStore(root)
    if backend == "sqlite":
        return SQLiteKnowledgeStore(root)
    raise ValueError("store backend must be jsonl or sqlite")

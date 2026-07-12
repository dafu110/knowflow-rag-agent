from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunking import chunk_document
from .models import Chunk, Document


class KnowledgeStore:
    def __init__(self, root: Path | str = "data/knowledge_store") -> None:
        self.root = Path(root)
        self.documents_path = self.root / "documents.jsonl"
        self.chunks_path = self.root / "chunks.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.documents_path.unlink(missing_ok=True)
        self.chunks_path.unlink(missing_ok=True)

    def add_documents(self, documents: list[Document]) -> int:
        existing_doc_ids = {document.id for document in self.documents()}
        new_documents = [document for document in documents if document.id not in existing_doc_ids]
        if not new_documents:
            return 0
        with self.documents_path.open("a", encoding="utf-8") as doc_file, self.chunks_path.open(
            "a", encoding="utf-8"
        ) as chunk_file:
            for document in new_documents:
                doc_file.write(json.dumps(_to_json(document), ensure_ascii=False) + "\n")
                for chunk in chunk_document(document):
                    chunk_file.write(json.dumps(_to_json(chunk), ensure_ascii=False) + "\n")
        return len(new_documents)

    def delete_document(self, document_id: str) -> bool:
        documents = self.documents()
        chunks = self.chunks()
        kept_documents = [document for document in documents if document.id != document_id]
        if len(kept_documents) == len(documents):
            return False
        kept_chunks = [chunk for chunk in chunks if chunk.document_id != document_id]
        _write_jsonl(self.documents_path, [_to_json(document) for document in kept_documents])
        _write_jsonl(self.chunks_path, [_to_json(chunk) for chunk in kept_chunks])
        return True

    def documents(self) -> list[Document]:
        return [_document_from_json(item) for item in _read_jsonl(self.documents_path)]

    def chunks(self) -> list[Chunk]:
        return [_chunk_from_json(item) for item in _read_jsonl(self.chunks_path)]

    def stats(self) -> dict[str, int]:
        return {"documents": len(self.documents()), "chunks": len(self.chunks())}

    def index_version(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks():
            digest.update(chunk.id.encode("utf-8"))
            digest.update(chunk.text.encode("utf-8"))
            digest.update(",".join(sorted(chunk.allowed_roles)).encode("utf-8"))
            digest.update(",".join(sorted(chunk.allowed_users)).encode("utf-8"))
        return digest.hexdigest()[:16]

    def update_document_permissions(
        self,
        document_id: str,
        *,
        allowed_roles: set[str],
        allowed_users: set[str],
    ) -> bool:
        documents = self.documents()
        target = next((document for document in documents if document.id == document_id), None)
        if target is None:
            return False
        target.allowed_roles = set(allowed_roles)
        target.allowed_users = set(allowed_users)
        chunks = self.chunks()
        for chunk in chunks:
            if chunk.document_id == document_id:
                chunk.allowed_roles = set(allowed_roles)
                chunk.allowed_users = set(allowed_users)
        _write_jsonl(self.documents_path, [_to_json(document) for document in documents])
        _write_jsonl(self.chunks_path, [_to_json(chunk) for chunk in chunks])
        return True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_json(item: Document | Chunk) -> dict[str, Any]:
    payload = asdict(item)
    payload["allowed_roles"] = sorted(payload["allowed_roles"])
    payload["allowed_users"] = sorted(payload["allowed_users"])
    return payload


def _document_from_json(payload: dict[str, Any]) -> Document:
    payload = dict(payload)
    payload["allowed_roles"] = set(payload.get("allowed_roles", []))
    payload["allowed_users"] = set(payload.get("allowed_users", []))
    return Document(**payload)


def _chunk_from_json(payload: dict[str, Any]) -> Chunk:
    payload = dict(payload)
    payload["allowed_roles"] = set(payload.get("allowed_roles", []))
    payload["allowed_users"] = set(payload.get("allowed_users", []))
    return Chunk(**payload)

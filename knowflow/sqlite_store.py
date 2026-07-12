from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunking import chunk_document
from .models import Chunk, Document


class SQLiteKnowledgeStore:
    def __init__(self, path: Path | str = "data/knowledge_store/knowflow.db") -> None:
        self.path = _database_path(Path(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def reset(self) -> None:
        with closing(self._connect()) as db:
            with db:
                db.execute("delete from chunks")
                db.execute("delete from documents")

    def add_documents(self, documents: list[Document]) -> int:
        added = 0
        with closing(self._connect()) as db:
            with db:
                for document in documents:
                    try:
                        db.execute(
                            "insert into documents(id, payload) values(?, ?)",
                            (document.id, json.dumps(_to_json(document), ensure_ascii=False)),
                        )
                    except sqlite3.IntegrityError:
                        continue
                    for chunk in chunk_document(document):
                        db.execute(
                            "insert into chunks(id, document_id, payload) values(?, ?, ?)",
                            (chunk.id, chunk.document_id, json.dumps(_to_json(chunk), ensure_ascii=False)),
                        )
                    added += 1
        return added

    def delete_document(self, document_id: str) -> bool:
        with closing(self._connect()) as db:
            with db:
                cursor = db.execute("delete from documents where id = ?", (document_id,))
                return cursor.rowcount > 0

    def documents(self) -> list[Document]:
        with closing(self._connect()) as db:
            rows = db.execute("select payload from documents order by rowid").fetchall()
        return [_document_from_json(json.loads(row[0])) for row in rows]

    def chunks(self) -> list[Chunk]:
        with closing(self._connect()) as db:
            rows = db.execute("select payload from chunks order by rowid").fetchall()
        return [_chunk_from_json(json.loads(row[0])) for row in rows]

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as db:
            documents = db.execute("select count(*) from documents").fetchone()[0]
            chunks = db.execute("select count(*) from chunks").fetchone()[0]
        return {"documents": int(documents), "chunks": int(chunks)}

    def index_version(self) -> str:
        digest = hashlib.sha256()
        with closing(self._connect()) as db:
            rows = db.execute("select payload from chunks order by id").fetchall()
        for row in rows:
            chunk = _chunk_from_json(json.loads(row[0]))
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
        with closing(self._connect()) as db:
            row = db.execute("select payload from documents where id = ?", (document_id,)).fetchone()
            if row is None:
                return False
            document = _document_from_json(json.loads(row[0]))
            document.allowed_roles = set(allowed_roles)
            document.allowed_users = set(allowed_users)
            chunks = [_chunk_from_json(json.loads(item[0])) for item in db.execute("select payload from chunks where document_id = ?", (document_id,))]
            for chunk in chunks:
                chunk.allowed_roles = set(allowed_roles)
                chunk.allowed_users = set(allowed_users)
            with db:
                db.execute("update documents set payload = ? where id = ?", (json.dumps(_to_json(document), ensure_ascii=False), document_id))
                for chunk in chunks:
                    db.execute("update chunks set payload = ? where id = ?", (json.dumps(_to_json(chunk), ensure_ascii=False), chunk.id))
        return True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("pragma foreign_keys = on")
        return connection

    def _init_schema(self) -> None:
        with closing(self._connect()) as db:
            with db:
                db.execute(
                    """
                    create table if not exists documents (
                        id text primary key,
                        payload text not null
                    )
                    """
                )
                db.execute(
                    """
                    create table if not exists chunks (
                        id text primary key,
                        document_id text not null references documents(id) on delete cascade,
                        payload text not null
                    )
                    """
                )
                db.execute("create index if not exists idx_chunks_document_id on chunks(document_id)")


def _database_path(path: Path) -> Path:
    if path.suffix:
        return path
    return path / "knowflow.db"


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

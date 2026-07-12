from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


MAX_SESSIONS_PER_USER = 100


class SessionStore:
    """SQLite-backed authenticated session store safe across web workers."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._initialize()

    def list_for(self, user: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
        return [_summary(record) for row in rows if _can_access(record := _record_from_row(row), user)]

    def get_for(self, session_id: str, user: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        record = _record_from_row(row) if row else None
        return record if record and _can_access(record, user) else None

    def save(self, record: dict[str, Any], user: str) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT owner, shared_with FROM sessions WHERE id = ?", (record["id"],)).fetchone()
            if row and row["owner"] != user:
                raise PermissionError("only the session owner can update it")
            if not row:
                owned = connection.execute("SELECT COUNT(*) FROM sessions WHERE owner = ?", (user,)).fetchone()[0]
                if owned >= MAX_SESSIONS_PER_USER:
                    raise OverflowError("session limit reached")
            record["owner"] = user
            record["shared_with"] = json.loads(row["shared_with"]) if row else []
            connection.execute(
                """
                INSERT INTO sessions (id, owner, title, turns, shared_with, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  title = excluded.title,
                  turns = excluded.turns,
                  updated_at = excluded.updated_at
                """,
                _row_values(record),
            )
        return dict(record)

    def share(self, session_id: str, user: str, collaborators: list[str]) -> dict[str, Any] | None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None or row["owner"] != user:
                return None
            record = _record_from_row(row)
            record["shared_with"] = sorted(set(collaborators) - {user})
            connection.execute("UPDATE sessions SET shared_with = ? WHERE id = ?", (json.dumps(record["shared_with"]), session_id))
        return record

    def delete(self, session_id: str, user: str) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute("DELETE FROM sessions WHERE id = ? AND owner = ?", (session_id, user))
        return result.rowcount > 0

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        legacy_records = self._legacy_records()
        if legacy_records is not None:
            backup = self.path.with_suffix(self.path.suffix + ".legacy.json")
            if backup.exists():
                raise RuntimeError(f"legacy session backup already exists: {backup}")
            self.path.replace(backup)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  id TEXT PRIMARY KEY,
                  owner TEXT NOT NULL,
                  title TEXT NOT NULL,
                  turns TEXT NOT NULL,
                  shared_with TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS sessions_owner_updated ON sessions(owner, updated_at DESC)")
            for record in (legacy_records or {}).values():
                if not isinstance(record, dict) or not {"id", "owner", "title", "turns", "updated_at"}.issubset(record):
                    continue
                record["shared_with"] = record.get("shared_with", [])
                connection.execute(
                    "INSERT OR REPLACE INTO sessions (id, owner, title, turns, shared_with, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    _row_values(record),
                )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _legacy_records(self) -> dict[str, dict[str, Any]] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


def _row_values(record: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(record["id"]),
        str(record["owner"]),
        str(record["title"]),
        json.dumps(record["turns"], ensure_ascii=False, separators=(",", ":")),
        json.dumps(record.get("shared_with", []), ensure_ascii=False, separators=(",", ":")),
        str(record["updated_at"]),
    )


def _record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "owner": row["owner"],
        "title": row["title"],
        "turns": json.loads(row["turns"]),
        "shared_with": json.loads(row["shared_with"]),
        "updated_at": row["updated_at"],
    }


def _can_access(record: dict[str, Any], user: str) -> bool:
    return record.get("owner") == user or user in record.get("shared_with", [])


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "title": record["title"],
        "owner": record["owner"],
        "shared_with": record.get("shared_with", []),
        "updated_at": record["updated_at"],
        "turn_count": len(record.get("turns", [])),
    }

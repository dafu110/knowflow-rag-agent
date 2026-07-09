from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def log(self, event: dict[str, Any]) -> None:
        if not self.path:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")


def audit_logger_from_env() -> AuditLogger:
    return AuditLogger(os.environ.get("KNOWFLOW_AUDIT_LOG", "").strip() or None)

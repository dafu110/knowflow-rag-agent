from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs

from .agent import RagAgent
from .evaluation import evaluate
from .models import Principal
from .server import CSP, HttpError, TokenAuthenticator, _clamp_int, _document_rows, _safe_eval_path
from .store_factory import create_store


STORE_ROOT = os.environ.get("KNOWFLOW_STORE", "data/knowledge_store")
STORE_BACKEND = os.environ.get("KNOWFLOW_STORE_BACKEND", "jsonl")
store = create_store(STORE_ROOT, STORE_BACKEND)
agent = RagAgent(store)
authenticator = TokenAuthenticator(os.environ.get("KNOWFLOW_AUTH_TOKENS", ""))


def application(environ: dict[str, object], start_response: Callable) -> Iterable[bytes]:
    try:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        if method == "GET" and path == "/health":
            return _json(start_response, {"ok": True, "stats": store.stats()})
        if method == "GET" and path == "/documents":
            return _json(start_response, {"documents": _document_rows(store)})
        if method == "POST" and path == "/ask":
            payload = _read_json(environ)
            principal = _principal_from_environ(environ, payload)
            answer = agent.ask(
                str(payload.get("question", "")),
                principal=principal,
                session_id=payload.get("session_id"),
                top_k=_clamp_int(payload.get("top_k", 6), minimum=1, maximum=20),
            )
            return _json(start_response, asdict(answer))
        if method == "POST" and path == "/eval":
            _principal_from_environ(environ, {})
            query = parse_qs(str(environ.get("QUERY_STRING", "")))
            eval_path = _safe_eval_path(query.get("path", ["rag_eval_set.jsonl"])[0])
            return _json(start_response, asdict(evaluate(agent, eval_path)))
        raise HttpError(404, "not found")
    except HttpError as error:
        return _json(start_response, {"ok": False, "error": {"status": error.status, "message": error.message}}, error.status)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _json(start_response, {"ok": False, "error": {"status": 400, "message": "invalid json"}}, 400)


def _read_json(environ: dict[str, object]) -> dict[str, object]:
    length = int(environ.get("CONTENT_LENGTH") or "0")
    body = environ["wsgi.input"].read(length).decode("utf-8") if length else "{}"
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise HttpError(400, "json body must be an object")
    return payload


def _principal_from_environ(environ: dict[str, object], payload: dict[str, object]) -> Principal:
    if authenticator.enabled:
        token = str(environ.get("HTTP_X_KNOWFLOW_TOKEN", ""))
        principal = authenticator.principal_for(token)
        if principal is None:
            raise HttpError(401, "missing or invalid api token")
        return principal
    return Principal(user=str(payload.get("user", "anonymous")), roles=set(payload.get("roles", [])))


def _json(start_response: Callable, payload: object, status: int = 200) -> list[bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    reason = "OK" if status < 400 else "Error"
    start_response(
        f"{status} {reason}",
        [
            ("content-type", "application/json; charset=utf-8"),
            ("content-length", str(len(body))),
            ("x-content-type-options", "nosniff"),
            ("referrer-policy", "same-origin"),
            ("x-frame-options", "DENY"),
            ("content-security-policy", CSP),
        ],
    )
    return [body]

from __future__ import annotations

import os
from typing import Callable, Iterable

from .server import WebApplication
from .store_factory import create_store


def create_application(
    store: object | None = None,
    *,
    auth_tokens: str | None = None,
) -> Callable[[dict[str, object], Callable], Iterable[bytes]]:
    store = store or create_store(
        os.environ.get("KNOWFLOW_STORE", "data/knowledge_store"),
        os.environ.get("KNOWFLOW_STORE_BACKEND", "jsonl"),
    )
    web = WebApplication(store, auth_tokens=auth_tokens)

    def application(environ: dict[str, object], start_response: Callable) -> Iterable[bytes]:
        method = str(environ.get("REQUEST_METHOD", "GET")).upper()
        path = str(environ.get("PATH_INFO", "/"))
        query = str(environ.get("QUERY_STRING", ""))
        target = f"{path}?{query}" if query else path
        headers = {
            "content-type": str(environ.get("CONTENT_TYPE", "")),
            "x-knowflow-token": str(environ.get("HTTP_X_KNOWFLOW_TOKEN", "")),
            "x-request-id": str(environ.get("HTTP_X_REQUEST_ID", "")),
        }
        client = str(environ.get("REMOTE_ADDR", "unknown"))
        limited = web.rate_limit_response(target, headers, client)
        if limited is not None:
            start_response(limited.status_line, list(limited.headers.items()))
            return [limited.body]
        length = _content_length(environ.get("CONTENT_LENGTH"))
        body = environ["wsgi.input"].read(length) if length else b""
        response = web.handle(method, target, headers=headers, body=body, client=client, rate_limited=True)
        start_response(response.status_line, list(response.headers.items()))
        return [response.body]

    return application


def _content_length(value: object) -> int:
    try:
        return max(0, int(str(value or "0")))
    except ValueError:
        return 0


application = create_application()

from __future__ import annotations

import hmac
import json
import os
from dataclasses import asdict
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from urllib.parse import parse_qs, urlparse

from .agent import RagAgent
from .chunking import SUPPORTED_EXTENSIONS, parse_document
from .evaluation import evaluate
from .models import Document, Principal
from .store import KnowledgeStore


WEB_ROOT = Path(__file__).with_name("web")
TEMPLATE_PATH = WEB_ROOT / "templates" / "index.html"
STATIC_ROOT = WEB_ROOT / "static"
EVAL_ROOT = Path("evals").resolve()
DEFAULT_EVAL_PATH = EVAL_ROOT / "rag_eval_set.jsonl"
MAX_JSON_BYTES = 32_000
MAX_UPLOAD_BYTES = 1_000_000
MAX_QUESTION_CHARS = 2_000
RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_REQUESTS = 120
MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}
CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class RateLimiter:
    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window_seconds: float = RATE_LIMIT_WINDOW_SECONDS) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = monotonic()
        recent = [stamp for stamp in self.requests.get(key, []) if now - stamp < self.window_seconds]
        if len(recent) >= self.max_requests:
            self.requests[key] = recent
            return False
        recent.append(now)
        self.requests[key] = recent
        return True


def create_handler(
    store: KnowledgeStore,
    agent: RagAgent | None = None,
    *,
    api_token: str | None = None,
    auth_tokens: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> type[BaseHTTPRequestHandler]:
    agent = agent or RagAgent(store)
    api_token = api_token if api_token is not None else os.environ.get("KNOWFLOW_API_TOKEN", "")
    authenticator = TokenAuthenticator(auth_tokens if auth_tokens is not None else os.environ.get("KNOWFLOW_AUTH_TOKENS", ""))
    rate_limiter = rate_limiter or RateLimiter()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._dispatch(self._handle_get)

        def do_POST(self) -> None:
            self._dispatch(self._handle_post)

        def do_DELETE(self) -> None:
            self._dispatch(self._handle_delete)

        def _dispatch(self, handler: object) -> None:
            try:
                client = self.client_address[0] if self.client_address else "unknown"
                if not rate_limiter.allow(client):
                    raise HttpError(429, "rate limit exceeded")
                handler()
            except HttpError as error:
                self._send_error(error.status, error.message)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error(400, "invalid json")
            except ValueError as error:
                self._send_error(400, str(error))

        def _handle_get(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/" or self.path.startswith("/?"):
                self._send_file(TEMPLATE_PATH, "text/html; charset=utf-8")
            elif parsed.path.startswith("/static/"):
                self._handle_static(parsed.path)
            elif parsed.path == "/health":
                self._send_json({"ok": True, "stats": store.stats()})
            elif parsed.path == "/documents":
                self._send_json({"documents": _document_rows(store)})
            else:
                raise HttpError(404, "not found")

        def _handle_post(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/ask":
                payload = self._read_json()
                principal = self._principal_from_request(payload)
                question = str(payload.get("question", "")).strip()
                if len(question) > MAX_QUESTION_CHARS:
                    raise HttpError(413, "question is too long")
                answer = agent.ask(
                    question,
                    principal=principal,
                    session_id=payload.get("session_id"),
                    top_k=_clamp_int(payload.get("top_k", 6), minimum=1, maximum=20),
                )
                self._send_json(asdict(answer))
            elif parsed.path == "/upload":
                self._principal_from_request({})
                self._handle_upload()
            elif parsed.path == "/eval":
                self._principal_from_request({})
                eval_path = _safe_eval_path(parse_qs(parsed.query).get("path", ["rag_eval_set.jsonl"])[0])
                self._send_json(asdict(evaluate(agent, eval_path)))
            else:
                raise HttpError(404, "not found")

        def _handle_delete(self) -> None:
            self._principal_from_request({})
            parsed = urlparse(self.path)
            if parsed.path != "/documents":
                raise HttpError(404, "not found")
            document_id = parse_qs(parsed.query).get("id", [""])[0]
            if not document_id:
                raise HttpError(400, "missing document id")
            deleted = store.delete_document(document_id)
            self._send_json({"deleted": deleted, "stats": store.stats(), "documents": _document_rows(store)})

        def _handle_static(self, path: str) -> None:
            relative = path.removeprefix("/static/")
            if "/" in relative or "\\" in relative:
                raise HttpError(404, "not found")
            file_path = (STATIC_ROOT / relative).resolve()
            if file_path.parent != STATIC_ROOT.resolve() or file_path.suffix not in MIME_TYPES:
                raise HttpError(404, "not found")
            self._send_file(file_path, MIME_TYPES[file_path.suffix])

        def _handle_upload(self) -> None:
            form = self._read_multipart()
            file_item = form.get("file")
            if not file_item or not file_item.get("filename"):
                raise HttpError(400, "missing file")
            filename = _safe_upload_filename(str(file_item["filename"]))
            content_bytes = bytes(file_item["content"])
            if len(content_bytes) > MAX_UPLOAD_BYTES:
                raise HttpError(413, "uploaded file is too large")
            content = content_bytes.decode("utf-8")
            roles = str(form.get("roles", {}).get("text", "")).strip()
            users = str(form.get("users", {}).get("text", "")).strip()
            metadata = []
            if roles:
                metadata.append(f"allowed_roles: {roles}")
            if users:
                metadata.append(f"allowed_users: {users}")
            if metadata and not content.startswith("---"):
                content = "---\n" + "\n".join(metadata) + "\n---\n" + content
            document = parse_document(Path(filename), text=content)
            added = store.add_documents([document])
            self._send_json(
                {"added": added, "document_id": document.id, "stats": store.stats(), "documents": _document_rows(store)}
            )

        def _read_json(self) -> dict:
            length = _content_length(self.headers.get("content-length"), max_bytes=MAX_JSON_BYTES)
            content_type = self.headers.get("content-type", "")
            if length and "application/json" not in content_type:
                raise HttpError(415, "content-type must be application/json")
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            if not isinstance(payload, dict):
                raise HttpError(400, "json body must be an object")
            return payload

        def _read_multipart(self) -> dict[str, dict[str, object]]:
            length = _content_length(self.headers.get("content-length"), max_bytes=MAX_UPLOAD_BYTES)
            content_type = self.headers.get("content-type", "")
            if "multipart/form-data" not in content_type:
                raise HttpError(415, "content-type must be multipart/form-data")
            body = self.rfile.read(length)
            raw = f"Content-Type: {content_type}\nMIME-Version: 1.0\n\n".encode("utf-8") + body
            message = BytesParser(policy=policy.default).parsebytes(raw)
            fields: dict[str, dict[str, object]] = {}
            for part in message.iter_parts():
                disposition = part.get("content-disposition", "")
                if "form-data" not in disposition:
                    continue
                name = part.get_param("name", header="content-disposition")
                if not name:
                    continue
                content = part.get_payload(decode=True) or b""
                fields[name] = {
                    "filename": part.get_filename(),
                    "content": content,
                    "text": content.decode("utf-8", errors="strict"),
                }
            return fields

        def _principal_from_request(self, payload: dict[str, object]) -> Principal:
            provided = self.headers.get("x-knowflow-token", "")
            if authenticator.enabled:
                principal = authenticator.principal_for(provided)
                if principal is None:
                    raise HttpError(401, "missing or invalid api token")
                return principal
            if api_token:
                if not hmac.compare_digest(provided, api_token):
                    raise HttpError(401, "missing or invalid api token")
                return Principal(user=str(payload.get("user", "anonymous")), roles=set(payload.get("roles", [])))
            return Principal(user=str(payload.get("user", "anonymous")), roles=set(payload.get("roles", [])))

        def _send_json(self, payload: object, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self._send_bytes(status, data, "application/json; charset=utf-8")

        def _send_error(self, status: int, message: str) -> None:
            self._send_json({"ok": False, "error": {"status": status, "message": message}}, status=status)

        def _send_file(self, path: Path, content_type: str) -> None:
            if not path.exists() or not path.is_file():
                raise HttpError(404, "not found")
            self._send_bytes(200, path.read_bytes(), content_type)

        def _send_bytes(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.send_header("x-content-type-options", "nosniff")
            self.send_header("referrer-policy", "same-origin")
            self.send_header("x-frame-options", "DENY")
            self.send_header("content-security-policy", CSP)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


class TokenAuthenticator:
    def __init__(self, raw_tokens: str = "") -> None:
        self.tokens: dict[str, Principal] = {}
        for item in raw_tokens.split(";"):
            item = item.strip()
            if not item:
                continue
            token, user, roles = _parse_auth_token(item)
            self.tokens[token] = Principal(user=user, roles=roles)

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)

    def principal_for(self, token: str) -> Principal | None:
        for expected, principal in self.tokens.items():
            if hmac.compare_digest(token, expected):
                return principal
        return None


def run_server(store: KnowledgeStore, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = ThreadingHTTPServer((host, port), create_handler(store))
    print(f"KnowFlow RAG Agent running at http://{host}:{port}")
    server.serve_forever()


def _content_length(value: str | None, *, max_bytes: int) -> int:
    try:
        length = int(value or "0")
    except ValueError as error:
        raise HttpError(400, "invalid content-length") from error
    if length > max_bytes:
        raise HttpError(413, "request body is too large")
    return length


def _parse_auth_token(raw: str) -> tuple[str, str, set[str]]:
    parts = raw.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError("KNOWFLOW_AUTH_TOKENS entries must be token:user:role1,role2")
    roles = {role.strip() for role in parts[2].split(",") if role.strip()}
    return parts[0], parts[1], roles


def _safe_upload_filename(filename: str) -> str:
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if not safe_name or safe_name in {".", ".."}:
        raise HttpError(400, "invalid filename")
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HttpError(415, f"unsupported file type; allowed: {allowed}")
    return safe_name


def _safe_eval_path(requested: str) -> Path:
    requested_path = Path(requested)
    if requested_path.is_absolute() or any(part == ".." for part in requested_path.parts):
        raise HttpError(400, "eval path must stay inside evals")
    if requested_path.suffix != ".jsonl":
        raise HttpError(415, "eval path must be a .jsonl file")
    candidate = (EVAL_ROOT / requested_path).resolve()
    if candidate.parent != EVAL_ROOT or not candidate.exists():
        raise HttpError(404, "eval file not found")
    return candidate


def _clamp_int(value: object, *, minimum: int, maximum: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return minimum
    return max(minimum, min(maximum, numeric))


def _document_rows(store: KnowledgeStore) -> list[dict[str, object]]:
    chunk_counts: dict[str, int] = {}
    for chunk in store.chunks():
        chunk_counts[chunk.document_id] = chunk_counts.get(chunk.document_id, 0) + 1
    return [_document_row(document, chunk_counts.get(document.id, 0)) for document in store.documents()]


def _document_row(document: Document, chunk_count: int) -> dict[str, object]:
    return {
        "id": document.id,
        "title": document.title,
        "source": document.source,
        "allowed_roles": sorted(document.allowed_roles),
        "allowed_users": sorted(document.allowed_users),
        "chunk_count": chunk_count,
        "created_at": document.created_at,
    }

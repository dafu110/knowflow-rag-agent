from __future__ import annotations

import hmac
import json
import os
import uuid
from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from urllib.parse import parse_qs, urlparse

from .agent import RagAgent
from .audit import AuditLogger, audit_logger_from_env
from .chunking import SUPPORTED_EXTENSIONS, parse_document
from .evaluation import evaluate
from .models import Document, Principal
from .store_factory import Store


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


@dataclass(slots=True)
class WebResponse:
    status: int
    body: bytes
    content_type: str
    request_id: str

    @property
    def status_line(self) -> str:
        reason = "OK" if self.status < 400 else "Error"
        return f"{self.status} {reason}"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "content-type": self.content_type,
            "content-length": str(len(self.body)),
            "x-content-type-options": "nosniff",
            "referrer-policy": "same-origin",
            "x-frame-options": "DENY",
            "content-security-policy": CSP,
            "x-request-id": self.request_id,
        }


class WebApplication:
    """Shared, framework-neutral HTTP behavior for local and WSGI adapters."""

    def __init__(
        self,
        store: Store,
        agent: RagAgent | None = None,
        *,
        api_token: str | None = None,
        auth_tokens: str | None = None,
        rate_limiter: RateLimiter | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.store = store
        self.agent = agent or RagAgent(store)
        self.api_token = api_token if api_token is not None else os.environ.get("KNOWFLOW_API_TOKEN", "")
        self.authenticator = TokenAuthenticator(auth_tokens if auth_tokens is not None else os.environ.get("KNOWFLOW_AUTH_TOKENS", ""))
        self.rate_limiter = rate_limiter or RateLimiter()
        self.audit_logger = audit_logger if audit_logger is not None else audit_logger_from_env()

    def handle(
        self,
        method: str,
        target: str,
        *,
        headers: dict[str, str],
        body: bytes,
        client: str = "unknown",
    ) -> WebResponse:
        request_id = headers.get("x-request-id") or uuid.uuid4().hex
        try:
            if not self.rate_limiter.allow(client):
                raise HttpError(429, "rate limit exceeded")
            parsed = urlparse(target)
            if method == "GET":
                return self._get(parsed.path, headers, request_id)
            if method == "POST":
                return self._post(parsed, headers, body, request_id)
            if method == "DELETE":
                return self._delete(parsed, headers, request_id)
            raise HttpError(404, "not found")
        except HttpError as error:
            self._audit("request_error", {"method": method, "path": urlparse(target).path, "status": error.status, "error": error.message}, request_id, client)
            return self._json({"ok": False, "error": {"status": error.status, "message": error.message}}, error.status, request_id)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._audit("request_error", {"method": method, "path": urlparse(target).path, "status": 400, "error": "invalid json"}, request_id, client)
            return self._json({"ok": False, "error": {"status": 400, "message": "invalid json"}}, 400, request_id)
        except ValueError as error:
            self._audit("request_error", {"method": method, "path": urlparse(target).path, "status": 400, "error": str(error)}, request_id, client)
            return self._json({"ok": False, "error": {"status": 400, "message": str(error)}}, 400, request_id)

    def _get(self, path: str, headers: dict[str, str], request_id: str) -> WebResponse:
        if path == "/":
            return self._file(TEMPLATE_PATH, "text/html; charset=utf-8", request_id)
        if path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            if "/" in relative or "\\" in relative:
                raise HttpError(404, "not found")
            file_path = (STATIC_ROOT / relative).resolve()
            if file_path.parent != STATIC_ROOT.resolve() or file_path.suffix not in MIME_TYPES:
                raise HttpError(404, "not found")
            return self._file(file_path, MIME_TYPES[file_path.suffix], request_id)
        if path == "/health":
            return self._json({"ok": True, "stats": self.store.stats(), "providers": self.agent.provider_status()}, 200, request_id)
        if path == "/documents":
            return self._json({"documents": _document_rows(self.store)}, 200, request_id)
        raise HttpError(404, "not found")

    def _post(self, parsed: object, headers: dict[str, str], body: bytes, request_id: str) -> WebResponse:
        path = parsed.path
        if path == "/ask":
            payload = self._read_json(headers, body)
            principal = self._principal(payload, headers)
            question = str(payload.get("question", "")).strip()
            if len(question) > MAX_QUESTION_CHARS:
                raise HttpError(413, "question is too long")
            answer = self.agent.ask(question, principal=principal, session_id=payload.get("session_id"), top_k=_clamp_int(payload.get("top_k", 6), minimum=1, maximum=20))
            self._audit("ask", {"principal": _principal_summary(principal), "question_chars": len(question), "answer_type": answer.answer_type, "citations": len(answer.citations), "hallucination_risk": answer.hallucination_risk, "status": 200}, request_id)
            return self._json(asdict(answer), 200, request_id)
        if path == "/upload":
            principal = self._principal({}, headers)
            form = self._read_multipart(headers, body)
            file_item = form.get("file")
            if not file_item or not file_item.get("filename"):
                raise HttpError(400, "missing file")
            filename = _safe_upload_filename(str(file_item["filename"]))
            content_bytes = bytes(file_item["content"])
            if len(content_bytes) > MAX_UPLOAD_BYTES:
                raise HttpError(413, "uploaded file is too large")
            content = content_bytes.decode("utf-8")
            metadata = _upload_metadata(form)
            if metadata and not content.startswith("---"):
                content = "---\n" + "\n".join(metadata) + "\n---\n" + content
            document = parse_document(Path(filename), text=content)
            added = self.store.add_documents([document])
            self.agent.invalidate_retriever()
            self._audit("upload", {"principal": _principal_summary(principal), "filename": filename, "document_id": document.id, "chunks_added": added, "allowed_roles_count": len(document.allowed_roles), "allowed_users_count": len(document.allowed_users), "status": 200}, request_id)
            return self._json({"added": added, "document_id": document.id, "stats": self.store.stats(), "documents": _document_rows(self.store)}, 200, request_id)
        if path == "/eval":
            principal = self._principal({}, headers)
            eval_path = _safe_eval_path(parse_qs(parsed.query).get("path", ["rag_eval_set.jsonl"])[0])
            result = evaluate(self.agent, eval_path)
            self._audit("eval", {"principal": _principal_summary(principal), "eval_path": eval_path.name, "total": result.total, "recall_at_k": result.recall_at_k, "permission_leaks": result.permission_leaks, "status": 200}, request_id)
            return self._json(asdict(result), 200, request_id)
        raise HttpError(404, "not found")

    def _delete(self, parsed: object, headers: dict[str, str], request_id: str) -> WebResponse:
        if parsed.path != "/documents":
            raise HttpError(404, "not found")
        principal = self._principal({}, headers)
        document_id = parse_qs(parsed.query).get("id", [""])[0]
        if not document_id:
            raise HttpError(400, "missing document id")
        deleted = self.store.delete_document(document_id)
        self.agent.invalidate_retriever()
        self._audit("delete_document", {"principal": _principal_summary(principal), "document_id": document_id, "deleted": deleted, "status": 200}, request_id)
        return self._json({"deleted": deleted, "stats": self.store.stats(), "documents": _document_rows(self.store)}, 200, request_id)

    def _principal(self, payload: dict[str, object], headers: dict[str, str]) -> Principal:
        provided = headers.get("x-knowflow-token", "")
        if self.authenticator.enabled:
            principal = self.authenticator.principal_for(provided)
            if principal is None:
                raise HttpError(401, "missing or invalid api token")
            return principal
        if self.api_token and not hmac.compare_digest(provided, self.api_token):
            raise HttpError(401, "missing or invalid api token")
        return Principal(user=str(payload.get("user", "anonymous")), roles=set(payload.get("roles", [])))

    def _read_json(self, headers: dict[str, str], body: bytes) -> dict[str, object]:
        if len(body) > MAX_JSON_BYTES:
            raise HttpError(413, "request body is too large")
        if body and "application/json" not in headers.get("content-type", ""):
            raise HttpError(415, "content-type must be application/json")
        payload = json.loads(body.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise HttpError(400, "json body must be an object")
        return payload

    def _read_multipart(self, headers: dict[str, str], body: bytes) -> dict[str, dict[str, object]]:
        if len(body) > MAX_UPLOAD_BYTES:
            raise HttpError(413, "request body is too large")
        content_type = headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            raise HttpError(415, "content-type must be multipart/form-data")
        raw = f"Content-Type: {content_type}\nMIME-Version: 1.0\n\n".encode("utf-8") + body
        message = BytesParser(policy=policy.default).parsebytes(raw)
        fields: dict[str, dict[str, object]] = {}
        for part in message.iter_parts():
            if "form-data" not in part.get("content-disposition", ""):
                continue
            name = part.get_param("name", header="content-disposition")
            if name:
                content = part.get_payload(decode=True) or b""
                fields[name] = {"filename": part.get_filename(), "content": content, "text": content.decode("utf-8", errors="strict")}
        return fields

    def _json(self, payload: object, status: int, request_id: str) -> WebResponse:
        return WebResponse(status, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), "application/json; charset=utf-8", request_id)

    def _file(self, path: Path, content_type: str, request_id: str) -> WebResponse:
        if not path.exists() or not path.is_file():
            raise HttpError(404, "not found")
        return WebResponse(200, path.read_bytes(), content_type, request_id)

    def _audit(self, action: str, fields: dict[str, object], request_id: str, client: str = "unknown") -> None:
        self.audit_logger.log({"request_id": request_id, "action": action, "client": client, **fields})


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


def run_server(store: Store, host: str = "127.0.0.1", port: int = 8765) -> None:
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


def _upload_metadata(form: dict[str, dict[str, object]]) -> list[str]:
    metadata = []
    roles = str(form.get("roles", {}).get("text", "")).strip()
    users = str(form.get("users", {}).get("text", "")).strip()
    if roles:
        metadata.append(f"allowed_roles: {roles}")
    if users:
        metadata.append(f"allowed_users: {users}")
    return metadata


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


def _principal_summary(principal: Principal) -> dict[str, object]:
    return {"user": principal.user, "roles": sorted(principal.roles)}


# The local stdlib HTTP server is deliberately only an adapter. WebApplication
# owns all routes so Docker/WSGI and local development cannot drift apart.
def create_handler(
    store: Store,
    agent: RagAgent | None = None,
    *,
    api_token: str | None = None,
    auth_tokens: str | None = None,
    rate_limiter: RateLimiter | None = None,
    audit_logger: AuditLogger | None = None,
) -> type[BaseHTTPRequestHandler]:
    web = WebApplication(
        store,
        agent=agent,
        api_token=api_token,
        auth_tokens=auth_tokens,
        rate_limiter=rate_limiter,
        audit_logger=audit_logger,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._respond()

        def do_POST(self) -> None:
            self._respond()

        def do_DELETE(self) -> None:
            self._respond()

        def _respond(self) -> None:
            length = _content_length(self.headers.get("content-length"), max_bytes=MAX_UPLOAD_BYTES)
            response = web.handle(
                self.command,
                self.path,
                headers={
                    "content-type": self.headers.get("content-type", ""),
                    "x-knowflow-token": self.headers.get("x-knowflow-token", ""),
                    "x-request-id": self.headers.get("x-request-id", ""),
                },
                body=self.rfile.read(length),
                client=self.client_address[0] if self.client_address else "unknown",
            )
            self.send_response(response.status)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler

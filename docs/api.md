# KnowFlow API

KnowFlow exposes a small JSON and multipart API for local demos, internal tools, and WSGI deployments. If `KNOWFLOW_AUTH_TOKENS` is configured, requests must include `x-knowflow-token`; the server derives `user` and `roles` from that token and ignores spoofed request-body identity fields.

All responses include `x-request-id`. Clients may send their own `x-request-id`; otherwise the server generates one.

## Authentication

```http
x-knowflow-token: sales-token
x-request-id: demo-001
```

`KNOWFLOW_AUTH_TOKENS` format:

```text
token:user:role1,role2;another-token:ciso:security
```

## `GET /health`

Returns service status and index statistics.

```json
{
  "ok": true,
  "stats": {
    "documents": 4,
    "chunks": 8
  },
  "index_version": "a1b2c3d4e5f67890"
}
```

When external embedding, reranking, or LLM providers are enabled, `providers` also reports local call count, average latency, retry count, degradation count, last error type, and circuit state. Prompts, tokens, and secrets are never included.

## `GET /documents`

Returns indexed document summaries.

```json
{
  "documents": [
    {
      "id": "1700b40ae1d834c2",
      "title": "销售合同审批指南",
      "source": "sample_docs/sales_contract.md",
      "allowed_roles": ["legal", "sales"],
      "allowed_users": [],
      "chunk_count": 2,
      "created_at": "2026-07-09T00:00:00+00:00"
    }
  ]
}
```

## `POST /ask`

Runs permission-filtered retrieval and grounded answer generation.

Request:

```json
{
  "question": "销售合同审批需要哪些材料？",
  "user": "alice",
  "roles": ["sales"],
  "session_id": "demo",
  "top_k": 6
}
```

Response fields:

- `answer`: grounded answer or refusal text.
- `citations`: cited chunk metadata and quotes.
- `confidence`: local confidence score.
- `hallucination_risk`: `low`, `medium`, or `high`.
- `retrieval_debug`: score breakdown for retrieved chunks.
- `answer_type`: `grounded`, `clarify`, or `refusal`.

## `POST /upload`

Uploads and indexes one `.txt`, `.md`, `.csv`, or `.json` file.

```powershell
curl.exe -X POST http://127.0.0.1:8765/upload `
  -H "x-knowflow-token: sales-token" `
  -F "file=@sample_docs/sales_contract.md" `
  -F "roles=sales,legal"
```

Limits:

- Maximum upload size: 1 MB.
- Filename is normalized to a basename.
- UTF-8 `.txt`, `.md`, `.csv`, and `.json` are supported.
- `.pdf` and `.docx` are explicitly rejected in the dependency-free build with conversion guidance; unsupported extensions are rejected.

## `DELETE /documents?id=...`

Deletes a document and all related chunks.

```powershell
curl.exe -X DELETE "http://127.0.0.1:8765/documents?id=1700b40ae1d834c2" `
  -H "x-knowflow-token: sales-token"
```

## `POST /eval`

Runs the default offline evaluation set from `evals/rag_eval_set.jsonl`.

```json
{
  "total": 32,
  "recall_at_k": 1.0,
  "mrr": 0.981,
  "citation_accuracy": 1.0,
  "faithfulness": 1.0,
  "permission_leaks": 0
}
```

## Audit Log

Set `KNOWFLOW_AUDIT_LOG=data/audit.jsonl` to enable JSONL audit events. Events contain request/action summaries such as actor, action, request ID, answer type, citation count, eval metrics, and delete/upload targets. Full answers, uploaded file content, and tokens are not written to the audit log.

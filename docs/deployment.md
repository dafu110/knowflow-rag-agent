# Deployment And Operations

This guide covers the smallest production-like KnowFlow deployment. The app can run fully local for demos, but shared environments should enable token auth, SQLite persistence, audit logging, TLS termination, and CI release gates.

## Environment Checklist

Required for shared deployments:

```text
KNOWFLOW_STORE_BACKEND=sqlite
KNOWFLOW_STORE=data/knowflow.db
KNOWFLOW_AUTH_TOKENS=<sales-token>:alice:sales;<security-token>:ciso:security;<admin-token>:admin:admin
KNOWFLOW_AUDIT_LOG=data/audit.jsonl
```

Optional model backends:

```text
OPENAI_API_KEY=...
KNOWFLOW_EMBEDDING_PROVIDER=openai
KNOWFLOW_EMBEDDING_MODEL=text-embedding-3-small
KNOWFLOW_LLM_PROVIDER=openai
KNOWFLOW_LLM_MODEL=gpt-4.1-mini
KNOWFLOW_RERANK_URL=https://rerank.example.com/rerank
```

Do not commit real `.env` files. Use a secret manager or runtime environment variables for production tokens and model keys.

## Docker Compose

```powershell
copy .env.example .env
# Edit .env and set unique, secret token values before continuing.
docker compose up --build
```

Compose binds the service to `http://127.0.0.1:8765` and stores knowledge, session, and audit data in the `knowflow-data` volume mounted at `/app/data`. It refuses to start until `KNOWFLOW_AUTH_TOKENS` is set in `.env`.

## WSGI

For non-container deployments:

```bash
KNOWFLOW_STORE_BACKEND=sqlite \
KNOWFLOW_STORE=data/knowflow.db \
KNOWFLOW_AUDIT_LOG=data/audit.jsonl \
KNOWFLOW_AUTH_TOKENS="<sales-token>:alice:sales;<security-token>:ciso:security;<admin-token>:admin:admin" \
gunicorn knowflow.wsgi:application --bind 0.0.0.0:8765 --workers 2
```

Put a reverse proxy in front of Gunicorn for TLS, request size limits, access logs, and compression.

## Release Gates

Run these before publishing a new release:

```powershell
python -m compileall -q knowflow tests scripts
python -m unittest discover -s tests -v
python scripts\check_eval.py
python scripts\demo_flow.py
```

Required eval minimums:

- recall@k >= 0.95
- MRR >= 0.90
- citation accuracy >= 0.95
- faithfulness >= 0.95
- permission leaks = 0

## Observability

Enable `KNOWFLOW_AUDIT_LOG` to capture JSONL events. Each event includes:

- `timestamp`
- `request_id`
- `action`
- `principal`
- request/result summary fields
- status or error category

Forward the audit file to your log pipeline if running beyond a single machine. Alert on:

- `permission_leaks > 0` in eval output.
- repeated `request_error` with status 401, 413, 415, or 429.
- spikes in `answer_type=refusal` for common business questions.
- long-running or failed model provider calls if external LLM/embedding/reranker is enabled.

`GET /health` includes provider retry, degradation, latency, and circuit-breaker summaries when a provider is configured. The Docker/Gunicorn deployment serves the same dashboard, static files, upload, document deletion, API, audit, and rate-limit behavior as the local server.

## Backup And Rollback

- Back up `data/knowflow.db` and `data/audit.jsonl`.
- Keep `.env` and token rotation records outside Git.
- Roll back by redeploying the previous image or commit and reusing the same mounted data volume.
- If a document caused bad retrieval behavior, delete it from `/documents` and add a regression case to `evals/rag_eval_set.jsonl`.

## Known Limits

- The built-in HTTP server is intended for local development; use WSGI/Gunicorn for shared environments.
- Local TF-IDF retrieval is strong for demos and small corpora, but larger corpora should use an external vector store or search service.
- External embeddings are cached per chunk for the process lifetime and only the query is embedded on each request. Use a persistent vector store when replicas or corpus sizes outgrow this cache.
- Audit logging is append-only JSONL and should be rotated by the hosting environment.

# Launch Hardening Checklist

## Blocking Gates

- Configure `KNOWFLOW_AUTH_TOKENS` before exposing the API to multiple users.
- Keep eval paths restricted to the `evals/` directory.
- Keep upload extension, filename, and size restrictions enabled.
- Verify permission leaks remain zero in offline evals.
- Use WSGI/Gunicorn and a reverse proxy with TLS for non-local deployment.

## Recommended Pre-Launch Commands

```powershell
python -m unittest discover -s tests
python scripts\check_eval.py
python scripts\demo_flow.py
```

Expected minimums:

- recall@k >= 0.95
- MRR >= 0.90
- citation accuracy >= 0.95
- faithfulness >= 0.95
- permission leaks = 0

## Portfolio Polish

- Keep `assets/knowflow-dashboard.png` and `assets/knowflow-mobile-check.png` current.
- Run `scripts\demo_flow.py` before recording a demo to show allowed and denied role-based questions.
- Document whether the demo used local TF-IDF or external embeddings/reranking.

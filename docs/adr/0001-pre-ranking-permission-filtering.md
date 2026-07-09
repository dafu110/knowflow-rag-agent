# ADR 0001: Pre-Ranking Permission Filtering

## Status

Accepted.

## Context

KnowFlow retrieves knowledge-base chunks that can include role- or user-restricted content. If restricted chunks are ranked before permission checks, debug output, citations, or score artifacts could reveal private document existence or content.

## Decision

Permission filtering must happen before BM25, TF-IDF or embedding scoring, reranking, citation selection, and answer synthesis. Request-body roles are ignored when token-to-principal mapping is enabled through `KNOWFLOW_AUTH_TOKENS`.

## Consequences

- Retrieval quality is evaluated only over chunks visible to the current principal.
- Permission leak tests are part of the release gate.
- Sensitive-intent answers must refuse when visible evidence does not support the answer.
- Any future vector-store backend must preserve the same filtering order.

## Verification

```powershell
python -m unittest discover -s tests
python scripts\check_eval.py
```

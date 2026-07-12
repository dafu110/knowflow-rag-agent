# KnowFlow RAG Design

## Goal and boundaries

KnowFlow answers internal policy and process questions from visible evidence only. It cites the supporting chunks and refuses unsupported or inaccessible requests. The local-first build is designed for small corpora and does not claim full PDF/DOCX parsing or a distributed vector database.

## Ingestion and indexing

The dependency-free parser accepts UTF-8 `.txt`, `.md`, `.csv`, and `.json` files. It extracts YAML-like metadata, section-aware chunks, source paths, timestamps, users, and roles. `.pdf` and `.docx` are rejected with a conversion instruction rather than silently extracting incomplete text.

Each store exposes an `index_version` derived from chunk text and access-control metadata. Document ingestion, deletion, and permission changes produce a new version; the agent uses it to rebuild its in-memory retriever before the next request.

## Retrieval and reranking

Permission filtering occurs before any score is calculated. The experiment runner compares four strategies against the same corpus and evaluation set:

| Strategy | Score |
|---|---|
| `bm25` | Exact keyword ranking |
| `vector` | TF-IDF or configured embedding similarity |
| `hybrid` | 55% BM25 + 45% vector |
| `rerank` | 45% BM25 + 35% vector + 20% local/external rerank |

Use `python scripts\retrieval_experiment.py` to produce Recall@K, MRR, citation accuracy, faithfulness, permission leaks, and mean request latency for all strategies. The current sample corpus is intentionally small, so identical scores are evidence that it is not sufficient to claim a universal ranking winner. Production decisions should use a representative held-out corpus and its measured latency/cost budget.

## Generation and grounding

The agent selects only strong evidence, composes an extractive answer by default, and validates answer sentences against retrieved text. It returns a clarification or refusal when evidence is weak. Conversation memory is scoped by user, role set, and session ID, preventing one principal from reusing another principal's context.

## Evaluation and failures

`scripts/check_eval.py` runs the primary set, the independent holdout set, and the four-strategy comparison. The holdout covers paraphrases, Chinese typos, mixed English/Chinese queries, no-answer cases, cross-document queries, and prompt-injection attempts. Security regression tests cover cross-user, cross-role, cross-session, permission-update, document-deletion, and token-spoofing paths.

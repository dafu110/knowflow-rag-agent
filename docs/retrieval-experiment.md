# Retrieval Experiment

Run the comparison with:

```powershell
python scripts\retrieval_experiment.py
```

The runner ingests `sample_docs/` into a temporary store and evaluates the same 32-case set with four strategies. It reports Recall@K, MRR, citation accuracy, faithfulness, permission leaks, and mean end-to-end latency.

## Baseline Result

Current local run on the included four-document corpus:

| Strategy | Recall@K | MRR | Citation accuracy | Permission leaks | Mean latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 1.000 | 0.981 | 1.000 | 0 | 2.31 ms |
| Vector (local TF-IDF) | 1.000 | 0.981 | 1.000 | 0 | 1.71 ms |
| Hybrid | 1.000 | 0.981 | 1.000 | 0 | 1.83 ms |
| Rerank | 1.000 | 0.981 | 1.000 | 0 | 2.09 ms |

This is deliberately not presented as evidence that any strategy is universally best: the bundled corpus is too small and lexical to separate them. Its purpose is to make the comparison repeatable and to prevent an unsupported performance claim.

For a production decision, run the same script against a representative held-out corpus with exact IDs, paraphrases, acronyms, mixed-language queries, stale policies, and multi-document conflicts. Choose hybrid retrieval only if it improves Recall@K or MRR within the agreed latency and cost budget; use reranking only if its incremental quality gain justifies its added latency.

## Why Permissions Filter Before Ranking

The permission regression set includes cross-user, cross-role, cross-session, and injection attempts. In every denied case the expected result is not merely an empty citation: the restricted source is absent from `retrieval_debug` as well. This is the measurable contract for pre-ranking filtering. A post-ranking design could still expose a restricted document through rank, score, source name, or tracing output even if the final answer were suppressed.

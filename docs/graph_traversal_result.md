# Graph-traversal retrieval result

Tests whether a real multi-hop graph traversal (LocAgent-style: build a file-level import
graph, BFS outward from BM25-seeded anchors, rank by traversal distance) beats plain BM25.
This is a more thorough version of a signal already tried once before — a shallow 1-hop
"import-adjacency boost" in `research/hybrid-fusion-signals`, found weak there too — done
this time as genuine bounded BFS rather than a single adjacency-count bump. **Negative
result** — confirms rather than overturns the earlier finding.

## What was built

- `method/graph_retriever.py`: `build_import_graph()` (file-level, undirected, import edges
  via a dot-to-slash resolution heuristic — same tractable approximation used elsewhere in
  this project) and `rank_files_graph_traversal()` (BM25-seeded, bounded BFS, ranks by
  minimum hop-distance to any seed; unreached files fall back to the seed ranking's own
  relative order rather than being scored arbitrarily).
- Scope decisions: BM25-seeded anchors (free, offline, reuses existing infra — not LLM
  entity extraction), 2-hop default expansion, file-level graph only (class/function
  containment edges considered and dropped — seeding is file-level, so containment edges
  wouldn't add traversal power without also doing symbol-level seeding).
- `scripts/run_graph_traversal_test.py`: retrieval-only comparison harness (free, offline,
  no LLM/embedding calls), reusing the existing `evaluation/screening.py`
  `screen_manifest`/`summarize_screening` machinery.
- 11 new tests in `tests/test_graph_retriever.py`.

## Result

Same manifest as the RRF weight-sweep work (`swebench-multi-n30-s42-1fb8f4b8d82f`, n=30),
BM25-symbols seeding both configs, retrieval-only (no LLM cost).

| Config | MRR | MAP | Hit@1 | Hit@10 | Hit@100 |
|---|---:|---:|---:|---:|---:|
| bm25_symbols (baseline) | 0.163 | 0.164 | 0.100 | 0.400 | **0.867** |
| graph_traversal | 0.158 | 0.159 | 0.100 | 0.400 | **0.733** |

MRR/MAP are roughly a wash, but **Hit@100 drops meaningfully** (0.867 → 0.733) — traversal
is actively pushing some ground-truth files further down the ranking, not just failing to
help. Per-instance: **7 regressions, 2 improvements, 21 unchanged** (of 30). The regressions
are large (+444, +442, +118 in rank) while the improvements are smaller (-21, -239 on one
outlier) — a real directional effect, not noise.

## Diagnosed cause

When the ground-truth file isn't import-connected to the BM25 seed set within 2 hops (i.e.
graph traversal has no opinion about it), the ranking doesn't leave it alone — every file
that *is* graph-reachable from a seed gets promoted ahead of it, regardless of whether
those files are actually more relevant to the bug report. `scikit-learn__scikit-learn-14629`
is the clearest example: BM25 alone ranked the ground truth a respectable #42; graph
traversal buried it at #486 behind a wave of import-adjacent-to-something files that BM25
itself had ranked far lower.

**Reading**: file-level import connectivity is a weak proxy for bug relevance in this
codebase style. A file being one or two imports away from a BM25-favored file says little
about whether it actually contains the bug — consistent with the earlier, shallower 1-hop
attempt also finding this signal weak (`research/hybrid-fusion-signals`, AST-
similarity/dependency-graph/commit-recency signals all individually weak, ~0.02-0.06 MRR
alone). This more thorough 2-hop BFS version confirms that reading rather than overturning
it: the problem isn't traversal depth, it's that import adjacency itself doesn't track
relevance well enough to be a strong standalone or fusion signal here.

## What would be worth trying if this direction is resumed

- **Symbol-level seeding + containment edges**: the current graph is file-level because
  seeding is file-level (BM25). A version that matches bug-report entities to specific
  symbols (classes/functions) and seeds from those, using containment edges to roll back up
  to file-level, is closer to LocAgent's actual design and wasn't attempted here.
- **Call-graph edges**, not just imports: two files can be tightly coupled by function calls
  without importing each other directly through a resolvable dot-path (the heuristic also
  misses relative imports, re-exports, and star imports). Explicitly out of scope for this
  pass as the highest-effort, highest-risk part to get right.
- **Only promote, never penalize**: the regressions come from graph-reachable files
  outranking BM25's own better guesses. A version that uses traversal distance as a
  *tiebreaker* among otherwise-similar BM25 scores, rather than as a primary ranking
  signal, might avoid the Hit@100 regression while keeping whatever small signal exists.

## Reproducing

```bash
python scripts/run_graph_traversal_test.py \
  --manifest results/manifests/swebench-multi-n30-s42-1fb8f4b8d82f.json \
  --output results/graph_traversal_swebench_30.json
```

# Project structure

```
dataset/
  base.py            # BugLocalizationDataset abstract base
  swebench.py        # SWE-bench Verified loader (supports SWEBENCH_LOCAL_PATH for offline loading)
  beetlebox.py        # BeetleBox dataset loader (multi-language: Python, Java, C++, JS, Go)
  models.py          # BugInstance dataclass
  repo_cache.py      # Bare-clone git cache for offline repo file access (used on networks with no GitHub API access)
  utils.py           # Token counting, GitHub file fetching, chunking helpers
  localizability.py  # Ground-truth localizability diagnostics: classifies each GT file as exists-before-fix / deleted-by-fix / added-by-fix / unresolved / api-error, with disk caching

method/
  base.py                    # BugLocalizationMethod abstract base
  openrouter_localizer.py    # Main localizer, routes through OpenRouter (multiple free/paid models)
  openai_localizer.py        # Direct OpenAI localizer, with chunking + aggregation for large repos
  openai_free_localizer.py   # Alternate direct-OpenAI localizer
  opensource_localizer.py    # Local/unsloth inference (currently disabled, not imported by main.py)
  rag_localizer.py           # RAG pre-filter (Qdrant + embeddings) -- dead code, lazy-imported, never invoked by default
  bm25_retriever.py          # BM25 candidate pre-filter (--bm25-top-k in main.py): path-only, skeleton (docstring+symbol names), and symbols/symbols+imports document representations
  prompt.py                  # Prompt templates (localization, chunk aggregation, report summarization)
  evaluate.py                # Top-k accuracy / precision / recall / F1 evaluation
  llm.py                     # Low-level LLM client wrapper
  utils.py                   # JSON schema generation, empty-response fallback, GitHub file fetch for RAG
  models.py                  # Response dataclasses

evaluation/
  manifest.py             # Deterministic, diversity-constrained (max-N-per-repo) evaluation manifests -- a stable seeded sample so retrieval/reranking comparisons are apples-to-apples across runs
  screening.py            # Path-only BM25 screening over a manifest: best localizable-GT rank, Hit@k, recall@100/200, MAP, difficulty band per instance
  failure_attribution.py  # Splits misses into retrieval failures (GT never reached the candidate set) vs reranking failures (GT was reachable); oracle diagnostic force-injects GT to isolate pure reranking ability (costs LLM calls -- opt-in)

scripts/
  mirror_repos.py                    # Bare-clone every repo referenced by a dataset sample into repo_cache/
  localizability_report.py           # Run ground-truth localizability diagnostics over a dataset sample; prints classification counts + coverage, optional JSON report
  generate_evaluation_manifest.py    # Build and save a manifest (evaluation/manifest.py) from a seeded dataset sample
  run_bm25_screening.py              # Run BM25 screening (evaluation/screening.py) over a saved manifest; prints difficulty distribution, optional JSON report
  run_failure_attribution.py         # Free offline retrieval-vs-reranking split by default; --run-oracle actually calls the LLM reranker on oracle candidate sets (costs API calls)
  compare_bm25_representations.py    # Compare path-only / skeleton / symbols+imports / symbols-no-imports BM25 document representations on the same manifest -- free, offline

tests/
  openrouter_key_test.py       # Standalone check that OPENROUTER_API_KEY is valid (not pytest)
  github_token_test.py         # Standalone check that GITHUB_TOKEN is valid (not pytest)
  test_localizability.py       # dataset/localizability.py: classification, coverage, caching
  test_bm25_retriever.py       # method/bm25_retriever.py: tokenization, skeleton/symbol extraction, ranking
  test_evaluation_manifest.py  # evaluation/manifest.py: diversity selection, determinism, save/load
  test_evaluation_screening.py # evaluation/screening.py: difficulty bands, hit/recall, pluggable rank_fn
  test_failure_attribution.py  # evaluation/failure_attribution.py: retrieval-reach split, oracle candidate-set prep
  test_evaluate.py             # method/evaluate.py: accuracy/precision/recall/F1
  test_dataset_utils.py        # dataset/utils.py: extension filtering, token counting, chunking
  test_dataset_models.py       # dataset/models.py: BugInstance token counting
  test_dataset_beetlebox.py    # dataset/beetlebox.py: BEETLEBOX_LOCAL_PATH offline loading
  test_repo_cache.py           # dataset/repo_cache.py: offline git-cache reads (integration tests against a locally mirrored repo, skipped if none is mirrored)

docs/
  project_structure.md   # This file
  architecture.md         # Package-dependency and runtime data-flow diagrams (Mermaid)

main.py                          # CLI entry point; --bm25-top-k/--bm25-skeleton/--bm25-symbols[-imports] wire up the pre-filter, --output writes a full JSON report (config + per-bug + overall)
results/                         # Saved run outputs (tracked in git -- see results/README.md); manifests/ holds evaluation/manifest.py outputs
pytest.ini                       # testpaths = tests
conftest.py                      # Adds repo root to sys.path so `pytest` works regardless of invocation directory
requirements.txt                 # Full dependency set (includes unused local-inference/RAG deps)
requirements-mn5.txt              # Trimmed, pinned dependency set for offline MN5 wheelhouse transfer
requirements-mn5-unpinned.txt     # Same as above, without exact version pins (used when pins go stale)
ORIGIN.md                        # Fork provenance and what's changed since forking
README.md                        # Setup, usage, and architecture overview
.env.example                     # Template for required API keys
```

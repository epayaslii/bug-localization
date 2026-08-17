# Project structure

```
dataset/
  base.py            # BugLocalizationDataset abstract base
  swebench.py        # SWE-bench Verified loader (supports SWEBENCH_LOCAL_PATH for offline loading)
  beetlebox.py        # BeetleBox dataset loader (multi-language: Python, Java, C++, JS, Go; BEETLEBOX_LOCAL_PATH for offline loading)
  bench4bl.py         # Bench4BL loader (primary dataset) -- XML bug-repository parser + git ls-tree/show reads, no live network calls
  models.py          # BugInstance dataclass
  repo_cache.py      # Offline git content cache: bare-clone mirror (SWE-bench/BeetleBox) + Bench4BL's own gitrepo/ as a fallback path
  utils.py           # Token counting, GitHub file fetching, chunking helpers
  localizability.py  # Ground-truth localizability diagnostics: classifies each GT file as exists-before-fix / deleted-by-fix / added-by-fix / unresolved / api-error, with disk caching

method/
  base.py                    # BugLocalizationMethod abstract base
  openrouter_localizer.py    # Main localizer, routes through OpenRouter (multiple free/paid models)
  openai_localizer.py        # Direct OpenAI localizer, with chunking + aggregation for large repos
  openai_free_localizer.py   # Alternate direct-OpenAI localizer
  opensource_localizer.py    # Local/unsloth inference (currently disabled, not imported by main.py)
  rag_localizer.py           # RAG pre-filter (Qdrant + embeddings) -- dead code, lazy-imported, never invoked by default
  bm25_retriever.py          # BM25 candidate pre-filter (--bm25-top-k in main.py): path_only, skeleton, symbols(+imports) representations. Dispatches to java_parsing.py for .java files.
  java_parsing.py            # Java lexical scanner (regex/brace-depth, not a real parser) -- Java counterpart to bm25_retriever/embedding_retriever's Python ast.parse() paths
  embedding_retriever.py     # Chunked-embedding retrieval (rank_files_embedding_chunked); 6 supported models incl. Qwen3-Embedding-0.6B. Dispatches to java_parsing.py for .java files.
  hybrid_retriever.py        # BM25 candidate pool -> chunked-embedding rerank -> weighted Reciprocal Rank Fusion (reciprocal_rank_fusion, rank_files_hybrid)
  fusion_signals.py          # AST-similarity, 1-hop dependency-graph, commit-recency signals -- evaluated, informative negative result, not in the recommended path
  repository_index.py        # Persistent FAISS index + metadata sidecar, content-addressed chunk-embedding cache for incremental reuse/dedup
  prompt.py                  # Prompt templates (localization, chunk aggregation, report summarization)
  evaluate.py                # Top-k accuracy / precision / recall / F1 evaluation
  llm.py                     # Low-level LLM client wrapper
  utils.py                   # JSON schema generation, empty-response fallback, GitHub file fetch for RAG
  models.py                  # Response dataclasses

evaluation/
  manifest.py             # Deterministic, diversity-constrained (max-N-per-repo) evaluation manifests -- a stable seeded sample so retrieval/reranking comparisons are apples-to-apples across runs
  screening.py            # Pluggable rank_fn screening over a manifest: best localizable-GT rank, Hit@k, Recall@k, MRR, MAP, difficulty band per instance
  failure_attribution.py  # Splits misses into retrieval failures (GT never reached the candidate set) vs reranking failures (GT was reachable); oracle diagnostic force-injects GT to isolate pure reranking ability (costs LLM calls -- opt-in)

scripts/
  mirror_repos.py                          # Bare-clone every repo referenced by a SWE-bench/BeetleBox sample into repo_cache/
  mirror_bench4bl.py                       # Download + extract Bench4BL per-project SourceForge archives into bench4bl_cache/
  localizability_report.py                 # Run ground-truth localizability diagnostics over a dataset sample; prints classification counts + coverage, optional JSON report
  generate_evaluation_manifest.py          # Build and save a manifest (evaluation/manifest.py) from a seeded dataset sample -- any of swebench/beetlebox/bench4bl
  run_bm25_screening.py                    # Run BM25 screening (evaluation/screening.py) over a saved manifest; prints difficulty distribution, optional JSON report
  run_failure_attribution.py               # Free offline retrieval-vs-reranking split by default; --run-oracle actually calls the LLM reranker on oracle candidate sets (costs API calls)
  compare_bm25_representations.py          # Compare path-only / skeleton / symbols+imports / symbols-no-imports BM25 document representations on the same manifest -- free, offline
  compare_embedding_models.py              # Bake-off across supported embedding models on the same manifest
  compare_bm25_beetlebox_mirrored.py       # BeetleBox-specific BM25 comparison, tested replacement for an earlier ad hoc script
  index_repository.py                      # CLI for building a persistent FAISS index (method/repository_index.py) over a real repo
  run_hybrid_retrieval_test.py             # Compare BM25 vs chunked-embedding-reranked vs BM25+embedding hybrid (RRF fusion) on the same manifest
  run_hybrid_rrf_weighting_test.py         # Sweep RRF weight ratios (bm25:embedding) on one manifest, serial -- fine for small n
  run_hybrid_rrf_weighting_shard.py        # Same weight sweep, but one shard of a manifest per invocation -- for Slurm array jobs when serial is too slow (e.g. real Java-aware chunking)
  run_hybrid_retrieval_candidates_shard.py # Computes hybrid-RRF retrieval candidates only (no LLM call) per shard -- Phase 1 of a two-phase end-to-end eval when retrieval needs to run somewhere without live internet (MN5) ahead of the LLM call
  aggregate_rrf_shards.py                  # Merges per-shard weight-sweep JSONs (from run_hybrid_rrf_weighting_shard.py) into one report
  run_fusion_signals_test.py               # Evaluate method/fusion_signals.py's signals, individually and fused
  run_relevance_feedback_test.py           # Prototype of the BRaIn/IQLoc-style relevance-feedback + query-reformulation pipeline (see docs/relevance_feedback_scoping.md) -- not a finished pipeline yet
  mn5/                                     # sbatch scripts for MN5 Slurm submission (see docs/mn5_execution_handbook.md)

tests/
  openrouter_key_test.py       # Standalone check that OPENROUTER_API_KEY is valid (not pytest)
  github_token_test.py         # Standalone check that GITHUB_TOKEN is valid (not pytest)
  test_localizability.py       # dataset/localizability.py: classification, coverage, caching
  test_bm25_retriever.py       # method/bm25_retriever.py: tokenization, skeleton/symbol extraction, ranking
  test_java_parsing.py         # method/java_parsing.py: symbol/skeleton extraction, comment/string-literal noise-stripping, method-body chunking
  test_embedding_retriever_chunking.py  # method/embedding_retriever.py: AST-based chunking + fallback
  test_hybrid_retriever.py     # method/hybrid_retriever.py: Reciprocal Rank Fusion
  test_fusion_signals.py       # method/fusion_signals.py: individual signal scoring
  test_repository_index.py     # method/repository_index.py: FAISS index build/search, caching/dedup
  test_evaluation_manifest.py  # evaluation/manifest.py: diversity selection, determinism, save/load
  test_evaluation_screening.py # evaluation/screening.py: difficulty bands, hit/recall/MRR/MAP, pluggable rank_fn
  test_failure_attribution.py  # evaluation/failure_attribution.py: retrieval-reach split, oracle candidate-set prep
  test_evaluate.py             # method/evaluate.py: accuracy/precision/recall/F1
  test_dataset_utils.py        # dataset/utils.py: extension filtering, token counting, chunking
  test_dataset_models.py       # dataset/models.py: BugInstance token counting
  test_dataset_beetlebox.py    # dataset/beetlebox.py: BEETLEBOX_LOCAL_PATH offline loading
  test_dataset_bench4bl.py     # dataset/bench4bl.py: real git/XML fixtures -- happy path, skip conditions, dotted-path resolution, seeded sampling
  test_repo_cache.py           # dataset/repo_cache.py: offline git-cache reads (integration tests against a locally mirrored repo, skipped if none is mirrored)

docs/
  PROGRESS_REPORT.md      # Chronological status report -- the canonical "what's done, what's next" doc; other docs point here rather than duplicating
  project_structure.md    # This file
  architecture.md         # Package-dependency and runtime data-flow diagrams (Mermaid)
  bench4bl_result.md      # Bench4BL results (BM25, hybrid RRF, end-to-end) and the content-fetch bug found/fixed underneath them
  qwen3_rrf_result.md     # Qwen3-Embedding-0.6B through the weighted-RRF pipeline, SWE-bench
  sota_comparison.md      # This project's numbers vs. published SOTA baselines, both benchmarks
  failure_case_analysis.md  # Retrieval-vs-reranking failure split, real n=30 data
  mn5_execution_handbook.md # MareNostrum 5 access, setup, blockers found/resolved
  relevance_feedback_scoping.md  # Scoping doc for the BRaIn/IQLoc-style architecture direction, not yet a finished pipeline
  next_steps.md            # Supervisor direction guidance (no-cloud-transfer constraint, target architecture, benchmarks/papers) -- not a duplicate status tracker, see its own top note
  literature_review.md     # 24+ papers surveyed
  bench4bl_reference/      # Reference material for the Bench4BL benchmark itself

main.py                          # CLI entry point -- dataset selection (bench4bl default), optional BM25 or hybrid-RRF retrieval narrowing, --candidates-file for precomputed retrieval, --output writes a full JSON report (config + per-bug + overall)
results/                         # Saved run outputs (tracked in git); manifests/ holds evaluation/manifest.py outputs
pytest.ini                       # testpaths = tests
conftest.py                      # Adds repo root to sys.path so `pytest` works regardless of invocation directory
requirements.txt                 # Full dependency set (includes unused local-inference/RAG deps)
requirements-mn5.txt              # Trimmed, pinned dependency set for offline MN5 wheelhouse transfer
requirements-mn5-unpinned.txt     # Same as above, without exact version pins (used when pins go stale)
ORIGIN.md                        # Fork provenance and what's changed since forking
README.md                        # Setup, usage, and architecture overview
.env.example                     # Template for required API keys
```

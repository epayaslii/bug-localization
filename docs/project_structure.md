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
  prompt.py                  # Prompt templates (localization, chunk aggregation, report summarization)
  evaluate.py                # Top-k accuracy / precision / recall / F1 evaluation
  llm.py                     # Low-level LLM client wrapper
  utils.py                   # JSON schema generation, empty-response fallback, GitHub file fetch for RAG
  models.py                  # Response dataclasses

scripts/
  mirror_repos.py            # Bare-clone every repo referenced by a dataset sample into repo_cache/
  localizability_report.py   # Run ground-truth localizability diagnostics over a dataset sample; prints classification counts + coverage, optional JSON report

tests/
  openrouter_key_test.py   # Standalone check that OPENROUTER_API_KEY is valid (not pytest)
  github_token_test.py     # Standalone check that GITHUB_TOKEN is valid (not pytest)

docs/
  project_structure.md   # This file

main.py                          # CLI entry point
requirements.txt                 # Full dependency set (includes unused local-inference/RAG deps)
requirements-mn5.txt              # Trimmed, pinned dependency set for offline MN5 wheelhouse transfer
requirements-mn5-unpinned.txt     # Same as above, without exact version pins (used when pins go stale)
ORIGIN.md                        # Fork provenance and what's changed since forking
README.md                        # Setup, usage, and architecture overview
.env.example                     # Template for required API keys
```

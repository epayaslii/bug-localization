# Fork origin

This repo was forked from [ekaramustafa/bug-localization](https://github.com/ekaramustafa/bug-localization).

Shared history ends at `aa4a800` ("Merge pull request #10 from ekaramustafa/feature-refactoring"). Everything after that commit is independent work in this fork:

- Offline repo-file access (`dataset/repo_cache.py`, `scripts/mirror_repos.py`) — bare-clone repos locally instead of hitting the live GitHub API, needed for environments with no outbound internet (e.g. HPC clusters).
- `SWEBENCH_LOCAL_PATH` support in `dataset/swebench.py` so SWE-bench Verified can load from a local disk copy offline.
- Switched target benchmark from the full original SWE-bench to SWE-bench Verified.
- OpenRouter default model switched from `qwen/qwen3-coder:free` to `gpt-oss-20b` after OpenRouter retired the free Qwen3-Coder tier.
- Fixed the chunk-aggregation prompt corrupting file paths with confidence text.
- Lazy/optional RAG import in `method/openrouter_localizer.py` (RAG is dead code, never invoked by default).
- `requirements-mn5.txt` / `requirements-mn5-unpinned.txt` — trimmed dependency set for offline transfer to MN5 (MareNostrum 5), excluding the unused local-inference/RAG stack (torch, transformers, accelerate, bitsandbytes, sentence-transformers, langchain/langgraph).

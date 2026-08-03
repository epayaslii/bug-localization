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

## Documentation, packaging, and research infrastructure

- Added `README.md`, `.env.example`, `docs/project_structure.md`, and this `ORIGIN.md`.
- Ported four improvements from a sibling fork (`adisenaa/Bug-Report-Localization`): `--max-files` flag, configurable OpenAI model, more OpenRouter model options (`gpt-4o-mini`, `gemini-flash`, `haiku`), and standalone API-key sanity-check scripts (`tests/`).
- Built `docs/literature_review.md` — 24 papers on LLM-based bug/fault localization reviewed directly from source PDFs, including all four baselines named in the original project handover doc (BLAZE, BugCerberus, FlexFL, MarsCode) plus AgentFL. Documents the SWE-bench Verified contamination caveat (OpenAI no longer evaluates on it) and a quick-reference table of the strongest reported Top-1/file-localization results in the field.
- Added a BM25 retrieval pre-filter (`method/bm25_retriever.py`) with path-only and content-skeleton (docstring + class/function names) variants, and an embedding-based retriever (`method/embedding_retriever.py`, UniXCoder) for comparison — both narrow the candidate file list before prompting, addressing the literature finding that dumping an entire repo's file list into one prompt caps accuracy. Uses `git cat-file --batch` to fetch file content for an entire candidate list in one subprocess rather than one `git show` per file.

## Beyond the repo

The offline dataset, `repo_cache`, and wheelhouse were transferred to and validated on MareNostrum 5 (MN5): a real 5-sample pipeline run confirmed the dataset-loading, repo-cache, and prompt-construction machinery works end-to-end there. Live LLM inference still requires a networked machine (MN5 has no outbound internet), by design.

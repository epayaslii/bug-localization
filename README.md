# Bug Localization

Given a bug report and a snapshot of a repository's code, predict which source files need to change to fix the bug. This repo implements an LLM-based bug localizer and evaluates it against public bug-localization benchmarks (SWE-bench Verified, BeetleBox).

> **Benchmark caveat:** OpenAI has [stopped evaluating on SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) and now recommends SWE-bench Pro instead, citing tests that reject functionally correct solutions and possible train/test contamination. This project still targets Verified for continuity with prior results — treat any accuracy numbers here with that caveat in mind, and consider SWE-bench Pro or a time-separated benchmark for future work.

Forked from [ekaramustafa/bug-localization](https://github.com/ekaramustafa/bug-localization) — see [ORIGIN.md](ORIGIN.md) for what's changed since.

See [docs/literature_review.md](docs/literature_review.md) for the survey of recent LLM-based bug localization/retrieval papers informing this project's direction.

## How it works

1. **Dataset** (`dataset/`) loads bug instances (bug report + target repo + base commit + ground-truth changed files) from SWE-bench Verified or BeetleBox.
2. **Repo file access**: for each bug instance, the list of candidate code files (and their contents) is pulled either from the GitHub API or from a local git cache (see [Offline repo access](#offline-repo-access-repo_cache) below).
3. **Method** (`method/`) sends the bug report plus the candidate file list/contents to an LLM and asks it to rank/select the files most likely to need changes.
4. **Evaluation** (`method/evaluate.py`) compares predicted files against ground truth and reports accuracy (hit@k), precision, recall, and F1.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
OPENROUTER_API_KEY=...
GITHUB_TOKEN=...
OPENAI_API_KEY=...
```

- `GITHUB_TOKEN` is used to raise GitHub API rate limits when fetching repo file listings live (optional but recommended).
- `OPENROUTER_API_KEY` is required for the default `openrouter` method.
- `OPENAI_API_KEY` is only needed for the `openai`/`openai-free` methods.

## Running the test suite

```bash
python -m pytest
```

Covers `dataset/localizability.py`, `method/bm25_retriever.py`, `evaluation/` (manifest, screening, failure attribution), `method/evaluate.py`, and the `dataset/` helper/model/repo-cache modules -- unit tests plus a few integration tests against a locally mirrored repo (see [Offline repo access](#offline-repo-access-repo_cache)), which skip gracefully if none is mirrored yet. No network access or API keys required. `tests/openrouter_key_test.py` and `tests/github_token_test.py` are separate standalone scripts (not pytest) for checking your own API keys are valid -- run them directly, not via `pytest`.

## Usage

```bash
python main.py --method openrouter --dataset swebench --model gpt-oss-20b --sample-size N
```

Arguments (`main.py`):
- `--method`: `openai`, `openai-free`, `openrouter` (default), or `unsloth` (currently disabled, see [Notes](#notes))
- `--dataset`: `swebench` or `beetlebox` (default)
- `--model`: model name, see [Models](#models) below (default `gpt-oss-20b`)
- `--sample-size`: number of bug instances to sample
- `--max-files`: cap the number of code files sent to the model per bug (cheap smoke-testing only — may truncate away the actual ground-truth file, so don't use it for real evaluation runs)
- `--device`: `cuda`/`cpu`/`auto` (local inference only)
- `--bm25-top-k`: narrow code files to the top-K most relevant via BM25 before prompting (disabled by default)
- `--bm25-skeleton` / `--bm25-symbols` [`--bm25-symbols-imports`]: BM25 document representation when `--bm25-top-k` is set (see [Symbol-aware BM25 document representations](#symbol-aware-bm25-document-representations)); `--bm25-symbols` takes precedence if both are passed, plain path-only BM25 if neither is
- `--output`: optional path to write the run config + full evaluation results (including per-bug candidate files and ground truths) as JSON

Results are printed as an overall metrics table (accuracy/precision/recall/F1) plus per-bug breakdowns; see `results/README.md` for saved runs.

## Models

Models available per `--method`, and their evaluation status. Numbers are intentionally left blank — sample size and model set are both still changing, so any fixed number here would go stale immediately. See the most recent run logs/commits for current results.

| Method | `--model` | Backend | Cost | Top-1 Accuracy |
|---|---|---|---|---|
| `openrouter` | `gpt-oss-20b` (default) | `openai/gpt-oss-20b:free` | free | ... |
| `openrouter` | `qwen-coder` | `qwen/qwen3-coder` | paid | ... |
| `openrouter` | `gpt-4o-mini` | `openai/gpt-4o-mini` | paid (cheap) | ... |
| `openrouter` | `gemini-flash` | `google/gemini-flash-1.5` | paid (cheap) | ... |
| `openrouter` | `haiku` | `anthropic/claude-haiku-3-5` | paid (cheap) | ... |
| `openai` | `gpt-5-nano` (default) or any `--model` | OpenAI direct | paid | ... |
| `openai-free` | configurable | OpenAI direct | paid | ... |

## Offline repo access (`repo_cache`)

The GitHub API path (`dataset/utils.get_code_files`) times out on networks without outbound internet access (e.g. HPC clusters like MN5). To avoid this, repos can be mirrored locally as bare git clones and read from disk instead:

```bash
python scripts/mirror_repos.py --dataset swebench --sample-size N
```

This clones every repo referenced by the sampled bug instances into `repo_cache/<owner>__<repo>.git` (bare clones, gitignored). `dataset/utils.get_code_files` automatically prefers the local cache when a repo is already mirrored there, falling back to the GitHub API otherwise. `dataset/repo_cache.py` also fetches a specific commit directly by SHA if it isn't reachable from any already-fetched ref.

Cache location can be overridden with the `REPO_CACHE_DIR` environment variable.

## Ground-truth localizability diagnostics

Not every ground-truth file in a bug instance can actually be retrieved: some are introduced by the fixing commit itself and therefore don't exist in the before-fix corpus that retrieval searches. Scoring those as "retrieval failures" understates real performance. `dataset/localizability.py` classifies each ground-truth path as `exists_before_fix`, `deleted_by_fix` (present before, removed by the fix -- still a valid retrieval target), `added_by_fix` (not localizable), `missing_unresolved`, or `api_error` (never cached, so it's retried next run instead of remembered as final):

```bash
python scripts/localizability_report.py --dataset swebench --sample-size 30 --output results/localizability_swebench_30.json
```

Prints classification counts and three coverage metrics (raw / available-corpus / localizable coverage, averaged across sampled instances), and caches classification results in `repo_cache/localizability_cache.json`.

## Evaluation manifests and BM25 screening

Comparing retrieval/reranking approaches against different ad-hoc samples makes results non-comparable. `evaluation/manifest.py` builds a deterministic, seeded sample capped at `--max-per-repo` instances per repository (so no single repo dominates), saved with a stable content-derived ID:

```bash
python scripts/generate_evaluation_manifest.py --dataset swebench --size 24 --pool-size 100 --seed 42 --max-per-repo 2
```

`evaluation/screening.py` then runs path-only BM25 over every instance in a saved manifest, reporting the best rank among each instance's *localizable* ground-truth files (via the localizability diagnostics above), Hit@1/5/10/100/200, recall@100/200, and a difficulty band (`easy` / `medium` / `hard` / `outside_top200` / `no_localizable_gt`):

```bash
python scripts/run_bm25_screening.py --manifest results/manifests/swebench-multi-n24-s42-<hash>.json --output results/screening_swebench_24.json
```

The screening script re-derives the same seeded pool the manifest was built from (rather than reprocessing the whole dataset) to stay cheap, and warns on any instance-ID/repo mismatch.

## Retrieval-vs-reranking failure attribution

A low score can mean two different things: the correct file never made it into the candidate set BM25 hands to the LLM (a **retrieval failure**, which reranking can never fix), or it was in the candidate set but the LLM didn't surface it (a **reranking failure**). Conflating the two makes it easy to blame the wrong stage.

```bash
python scripts/run_failure_attribution.py --manifest results/manifests/<manifest_id>.json --candidate-size 100
```

By default this only does the free, offline split (reusing the BM25 screening ranks -- no LLM calls). Add `--run-oracle` to additionally run the **oracle diagnostic**: force-inject every localizable ground-truth file into the candidate set (so retrieval recall is artificially 100% for that call) and measure how well the LLM reranker places them, isolating pure reranking ability from retrieval quality. `--run-oracle` calls the LLM once per manifest instance and therefore costs real API usage -- it prints an estimated prompt-token count before making any calls, and defaults to the free `gpt-oss-20b` OpenRouter model (override with `--model`).

## Symbol-aware BM25 document representations

`method/bm25_retriever.py` supports four document representations for the BM25 candidate pre-filter, moving beyond bare file paths:

| Representation | What each file's BM25 document contains |
|---|---|
| `path_only` (`rank_files_bm25`) | Path tokens only |
| `skeleton` (`rank_files_bm25_with_skeleton`) | Path tokens + module docstring + class/function names |
| `symbols_with_imports` (`rank_files_bm25_with_symbols`, default) | Path tokens + class/function/method names + imported module/name tokens |
| `symbols_no_imports` (`rank_files_bm25_with_symbols(include_imports=False)`) | Path tokens + class/function/method names only |

All are Python-AST-based (SWE-bench Verified's only language today) and fall back to path-only tokens for any file whose content can't be read or parsed -- never a live network call, only the offline `repo_cache`. `main.py` wires up `path_only` (default), `skeleton` (`--bm25-skeleton`), and `symbols_with_imports`/`symbols_no_imports` (`--bm25-symbols`, `--bm25-symbols-imports` to include imports) for end-to-end runs. Compare all four on the same manifest without spending anything on an LLM:

```bash
python scripts/compare_bm25_representations.py --manifest results/manifests/<manifest_id>.json --output results/bm25_representation_comparison.json
```

Prints macro Hit@1/5/10, MRR, and MAP per representation, plus each one's difficulty-band distribution -- free and offline, no LLM calls. Retrieval still operates at file granularity (ground truth in this dataset is file-level); symbols/imports are used to build a richer document per file, not to retrieve individual symbols.

## Loading datasets from local disk

Both dataset loaders normally pull from Hugging Face (`SWE-bench/SWE-bench_Verified`, `bug-localization/BeetleBox`). For offline environments, a pre-downloaded copy (via `datasets.save_to_disk`) can be loaded instead by setting:

```
SWEBENCH_LOCAL_PATH=/path/to/local/swebench_verified
BEETLEBOX_LOCAL_PATH=/path/to/local/beetlebox
```

## Running on an HPC cluster (MN5 / SLURM)

Because clusters like MN5 have no outbound internet access, the LLM API call itself cannot run there — only the dataset loading, repo cache, and prompt-construction machinery can be validated on-cluster. The intended workflow is:

1. Produce real results locally (Mac/laptop with internet) using `repo_cache/` and the OpenRouter/OpenAI methods.
2. Ship the source tree, a prebuilt wheelhouse (`requirements-mn5.txt`, offline pip wheels), and the populated `repo_cache/` to the cluster.
3. Use the offline SLURM scaffold to validate the pipeline end-to-end on-cluster (dataset loading → repo cache reads → prompt construction), accepting that the final inference call will fail there by design.

`requirements-mn5.txt` is a trimmed dependency list for this offline transfer — it excludes the unused local-inference/RAG stack (torch, transformers, accelerate, bitsandbytes, sentence-transformers, langchain/langgraph), which are not imported by any active code path.

## Repo layout

See [docs/project_structure.md](docs/project_structure.md) for a full file-by-file breakdown, or [docs/architecture.md](docs/architecture.md) for package-dependency and runtime data-flow diagrams.

```
dataset/          Dataset loaders (SWEBench, BeetleBox), repo cache, token/utility helpers
method/           Localizer implementations (OpenAI, OpenAI-free, OpenRouter), prompt generation, evaluation
scripts/          Standalone utilities (mirror_repos.py for offline repo caching)
tests/            Standalone .env key sanity checks (not pytest) -- run directly to confirm API keys work
main.py           CLI entry point
requirements.txt  Full dependency set (includes unused local-inference/RAG deps, kept for compatibility)
requirements-mn5.txt   Trimmed dependency set for offline HPC transfer
```

## Notes

- The `unsloth`/local-inference method (`method/opensource_localizer.py`) and the RAG-based localizer (`method/rag_localizer.py`) are present but not used in the current pipeline — imports are lazy/optional and no CLI flag enables RAG by default.
- `method/evaluate.py` reports hit@k (default k=1), precision, recall, and F1, both per-bug and aggregated.

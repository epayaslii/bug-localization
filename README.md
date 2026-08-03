# Bug Localization

Given a bug report and a snapshot of a repository's code, predict which source files need to change to fix the bug. This repo implements an LLM-based bug localizer and evaluates it against public bug-localization benchmarks (SWE-bench Verified, BeetleBox).

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

## Usage

```bash
python main.py --method openrouter --dataset swebench --model gpt-oss-20b --sample-size 15
```

Arguments (`main.py`):
- `--method`: `openai`, `openai-free`, `openrouter` (default), or `unsloth` (currently disabled, see [Notes](#notes))
- `--dataset`: `swebench` or `beetlebox` (default)
- `--model`: model name, e.g. `gpt-oss-20b` (default), `qwen-coder`
- `--sample-size`: number of bug instances to sample
- `--device`: `cuda`/`cpu`/`auto` (local inference only)

Results are printed as an overall metrics table (accuracy/precision/recall/F1) plus per-bug breakdowns.

## Offline repo access (`repo_cache`)

The GitHub API path (`dataset/utils.get_code_files`) times out on networks without outbound internet access (e.g. HPC clusters like MN5). To avoid this, repos can be mirrored locally as bare git clones and read from disk instead:

```bash
python scripts/mirror_repos.py --dataset swebench --sample-size 15
```

This clones every repo referenced by the sampled bug instances into `repo_cache/<owner>__<repo>.git` (bare clones, gitignored). `dataset/utils.get_code_files` automatically prefers the local cache when a repo is already mirrored there, falling back to the GitHub API otherwise. `dataset/repo_cache.py` also fetches a specific commit directly by SHA if it isn't reachable from any already-fetched ref.

Cache location can be overridden with the `REPO_CACHE_DIR` environment variable.

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

```
dataset/          Dataset loaders (SWEBench, BeetleBox), repo cache, token/utility helpers
method/           Localizer implementations (OpenAI, OpenAI-free, OpenRouter), prompt generation, evaluation
scripts/          Standalone utilities (mirror_repos.py for offline repo caching)
main.py           CLI entry point
requirements.txt  Full dependency set (includes unused local-inference/RAG deps, kept for compatibility)
requirements-mn5.txt   Trimmed dependency set for offline HPC transfer
```

## Notes

- The `unsloth`/local-inference method (`method/opensource_localizer.py`) and the RAG-based localizer (`method/rag_localizer.py`) are present but not used in the current pipeline — imports are lazy/optional and no CLI flag enables RAG by default.
- `method/evaluate.py` reports hit@k (default k=1), precision, recall, and F1, both per-bug and aggregated.

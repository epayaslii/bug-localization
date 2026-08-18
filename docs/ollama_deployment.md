# Local LLM deployment via Ollama

## Why

Every LLM-reranking path in this project so far (`--method openrouter`, `openai`, `openai-free`)
calls a cloud API. That's fine for local development, but it doesn't match the supervisor's
actual target architecture, which needs to run somewhere with **no outbound internet** — MN5
is the concrete example already hit this session (its GPU is fully usable now, but the login/
compute nodes still can't reach the internet, which is why the hybrid-retrieval pipeline had
to be split into a two-phase MN5-retrieval / local-LLM-call pattern in
`scripts/run_hybrid_retrieval_candidates_shard.py`).

Ollama serves an LLM locally over HTTP and needs internet only once, to pull model weights —
after that it runs fully offline, the same shape as the HF-cache-transfer pattern already used
for the embedding models (BGE-Code-v1, UniXCoder, CodeBERT) and torch/CUDA wheels this session.
This closes that gap: the LLM-rerank step can now run without a cloud API key anywhere,
including MN5.

## Architecture

`method/ollama_localizer.py`'s `OllamaLocalizer` is a near-exact copy of
`method/openrouter_localizer.py`'s `OpenRouterLocalizer` — same `BugLocalizationMethod`
interface, same `invoke_structured()` / `generate_json_schema()` call, same
`PromptGenerator`. The only real differences:

- **Endpoint**: `http://<host>:11434/v1` (Ollama's OpenAI-compatible endpoint) instead of
  OpenRouter's `https://openrouter.ai/api/v1`.
- **Auth**: none required — Ollama doesn't check the API key, but the OpenAI Python SDK
  requires a non-empty string, so a dummy `"ollama"` is passed.
- **Model names**: local Ollama tags (`qwen2.5-coder:14b`) instead of OpenRouter slugs.

This worked without any fallback/simplified JSON handling — Ollama's OpenAI-compat endpoint
accepts the exact same `response_format: {"type": "json_schema", "json_schema": {..., "strict":
true, ...}}` payload `generate_json_schema()` already produces for OpenRouter/OpenAI, unchanged.
Confirmed with a real smoke test (2026-08-18, `qwen2.5:latest`, Bench4BL instance SWARM-844):
the model returned a valid structured `candidate_files` list, parsed cleanly, and flowed
through the existing evaluator with no errors.

## Usage

```bash
# Wire-up flags added to main.py:
#   --method ollama
#   --model <friendly name or raw tag, e.g. qwen2.5-coder or qwen2.5-coder:32b-instruct-q4_K_M>
#   --ollama-host <optional override, default $OLLAMA_HOST or http://localhost:11434>

ollama serve &                          # start the local server (once per machine/session)
ollama pull qwen2.5-coder:14b           # one-time, needs internet

python main.py --method ollama --model qwen2.5-coder --dataset bench4bl \
  --sample-size 1 --bm25-top-k 50 --bm25-skeleton
```

Friendly names currently mapped in `OllamaLocalizer.model_mapping` (edit this dict, and the
model-choice section below, if the recommendation changes):

| Friendly name | Ollama tag | Notes |
|---|---|---|
| `qwen2.5-coder` (default) | `qwen2.5-coder:14b` | recommended default, see below |
| `qwen2.5-coder-32b` | `qwen2.5-coder:32b` | stronger, ~2x the VRAM/time |
| `qwen2.5-coder-7b` | `qwen2.5-coder:7b` | faster, for cheap smoke-testing |
| `llama3.1` | `llama3.1:8b` | general-purpose fallback |
| `deepseek-coder-v2` | `deepseek-coder-v2:16b` | alternative code-specialized model |

A raw `family:tag` string (e.g. `qwen2.5-coder:32b-instruct-q4_K_M`) also works directly,
bypassing the friendly-name table, for any model not listed above.

## Model choice

Not yet benchmarked against `gpt-4o-mini` (the current OpenRouter default) on this project's
own manifests — that's the natural next step once a model is actually pulled and run at n=30.
`qwen2.5-coder:14b` is the current recommended default because:

- it's code-specialized (this is a bug-localization task over Java source), matching the
  project's existing preference for code-tuned models (Qwen3-Embedding for retrieval);
- 14B fits comfortably on MN5's H100 with headroom, and is a reasonable size for local Mac
  testing too, unlike `qwen2.5-coder:32b`;
- Qwen model family already has a track record in this project (Qwen3-Embedding-0.6B is the
  confirmed-best retrieval embedding model on both SWE-bench and Bench4BL).

The smoke test above used `qwen2.5:latest` (general-purpose, already pulled locally) only to
verify the plumbing — it was not evaluated for quality, and its one-instance result (a miss)
says nothing about real performance. `qwen2.5-coder:14b` has not been pulled or tested yet.

## MN5 deployment (planned, not yet executed)

Ollama is already available on MN5 as an environment module — no manual binary install
needed:

```bash
ssh mn5 "module load ollama/0.11.8"   # confirmed present 2026-08-18; also 0.1.38/0.3.4/0.3.12/0.6.2
```

Since MN5 has no outbound internet, models must be pulled locally and transferred, exactly
like the embedding-model HF caches this session (`hf_cache_mn5/`, see
`docs/mn5_execution_handbook.md`):

1. `ollama pull qwen2.5-coder:14b` locally — stored under `~/.ollama/models/{blobs,manifests}`
   (currently 39GB total across all locally-pulled models; a single 14B model is roughly
   9-10GB as q4 quantized weights).
2. `rsync -avz --partial ~/.ollama/models/ mn5:/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models/`
   — same resumable-transfer pattern used for the embedding weights this session (plain `scp`
   silently dropped a connection mid-transfer earlier today; `rsync --partial` is the safer
   choice for anything this size).
3. On MN5, before starting the server: `export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models`
   (Ollama's own env var for a non-default model directory — no code change needed).
4. `ollama serve &` inside the same sbatch job/allocation that will run `main.py --method
   ollama` against it (both need to run on the same node, same as any localhost service) —
   likely alongside the `--gres=gpu:1` GPU jobs already working this session, so Ollama's
   llama.cpp backend can use the H100.

**Not yet done**: the actual model pull+transfer, an MN5 smoke test, and a real quality
comparison against `gpt-4o-mini`. This section is a verified-feasible plan (module exists,
env var mechanism is documented Ollama behavior, transfer pattern is proven for
similarly-sized artifacts this session) but not yet executed end-to-end on MN5 itself.

# MareNostrum 5 execution handbook

Practical reference for running this repo's offline retrieval pipeline on MareNostrum 5
(MN5, BSC). Consolidates everything established across real access sessions to date —
account details, environment setup, five blockers found and diagnosed so far (**all five
now resolved**, as of 2026-08-17's GPU fix), and what's already confirmed working, including
a completed real `sbatch` submission. This is a **final deliverable named explicitly** in
the official study plan.

## What this handbook is for, and what it isn't

MN5 has **no outbound internet by design** — this is structural, not a bug to fix. That
means the LLM-calling parts of this pipeline (`main.py`'s OpenRouter/OpenAI localizers)
**cannot run on MN5 at all**, regardless of any setup work. MN5's actual value to this
project is running the **retrieval-only** stack (BM25 + local embeddings + RRF fusion) at
scale far beyond what's practical on a laptop — not hosting the full end-to-end pipeline.
Scope this handbook, and any MN5 work, accordingly.

## Account and access

- Account: `comm842299`, project: `ehpc680`
- Login nodes: `glogin1.bsc.es` / `glogin2.bsc.es` (GPP), `alogin1.bsc.es` / `alogin2.bsc.es` (ACC)
- Transfer nodes: `transfer1-4.bsc.es`
- **No GPP QoS on this account — ACC only** (`acc_debug`, `acc_ehpc`, +1 more). All Slurm
  submissions must target `--partition=acc`.
- Paths:
  - `$HOME = /home/comm/comm842299`
  - Project dir: `/gpfs/projects/ehpc680/comm842299/repos/bug-localization/`
  - Scratch: `/gpfs/scratch/ehpc680/comm842299/`

**SSH access — RESOLVED 2026-08-11.** No key existed on the primary working machine as of
earlier sessions; password auth doesn't work through any terminal relay (Claude Code's `!`
prefix included — confirmed no real TTY is available for the interactive password prompt).
Fix: generated a dedicated key (`~/.ssh/mn5_ed25519`, no passphrase — scoped only to MN5
automation, not a general-purpose key) and copied it to MN5 once via `ssh-copy-id` from a
real terminal app (not through Claude Code). An `~/.ssh/config` `Host mn5` entry (pointing
at `alogin1.bsc.es`, this key, `ControlMaster auto` for connection reuse) makes subsequent
access just `ssh mn5`. Verify with `ssh -o BatchMode=yes mn5 echo ok` — if that fails,
password auth is still the only path and needs the same real-terminal workaround.

### Slurm account/QoS/partition facts (discovered 2026-08-11, `sbatch` submission)

- `sbatch` needs an explicit account: `--account=ehpc680` (job submission fails with "No
  account specified" otherwise, even though `sacctmgr show associations` only lists one).
- QoS wall-clock limits (from `sacctmgr show qos`): `acc_debug` = 2 hours, high priority;
  `acc_ehpc` = 3 days, lower priority. Pick based on expected runtime, not just habit —
  CPU-bound embedding jobs (see Blocker 5 below) can easily exceed 2 hours.
- **`acc_debug` has `MaxSubmitPU=1`** (`sacctmgr show qos format=Name,MaxSubmitPU,MaxSubmitPA -p`)
  — only one job/task can be submitted at a time under this QoS. **Any array job
  (`--array=0-N` with N>0) must use `acc_ehpc` instead** — confirmed 2026-08-20, a real
  8-task array job failed outright with `QOSMaxSubmitJobPerUserLimit` under `acc_debug`.
  `acc_ehpc`'s limit is 366, comfortably covers any array size used so far.
- GPU requests have a fixed CPU ratio: Slurm rejects anything below `--cpus-per-task=20`
  per `--gres=gpu:1` ("Minimum cpus requested should be (nodes * gpus/node * 20)").
- `sinfo` is permission-denied for this account (not just unconfigured — a real, different
  restriction than node listing generally). `scontrol show partition acc` works instead and
  is enough to confirm the GPU resource name (`gres/gpu`) and node count.

## Environment setup — the module-load footgun

### The `python/3.11.5-gcc` module has a 4-deep undeclared prerequisite chain

Just running `module load python/3.11.5-gcc` fails outright — Lmod reports each missing
prerequisite **one at a time**, not all at once, so naively following the errors means
several rounds of trial and error. **The full chain, discovered 2026-08-10, load in this
order:**

```bash
module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
```

Verify all 6 (including `bsc/1.0`, loaded automatically) show up with `module list` before
proceeding — if any single one of these is missing, the next one in the chain fails with an
Lmod "Cannot load module ... without these module(s) loaded: X" error naming exactly what's
missing. Follow that error's own naming if this exact chain ever changes (BSC updates its
module tree periodically) — but this chain worked as of the date above. **`module spider
python/3.11.5-gcc` is worth trying first on a future session** — Lmod's `spider` subcommand
is designed to show a module's complete dependency tree in one shot, which would have saved
several of the round trips that were needed to discover this chain manually.

### PYTHONHOME/PYTHONPATH shadowing, on top of the chain above

Once `python/3.11.5-gcc` actually loads, it sets `PYTHONHOME` / `PYTHONPATH` pointing at the
module's own bundled system site-packages, which **silently shadow the project's own venv
(`.venv_mn5`)** — even `pip show` inside the venv reports the wrong package versions. This
was diagnosed once already (a stale bundled `pyarrow 16.1.0` was masking the venv's real
`pyarrow 25.0.0`).

**Fix — must be repeated on every fresh login**, since the module load resets these every time:

```bash
module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH
source /gpfs/projects/ehpc680/comm842299/repos/bug-localization/.venv_mn5/bin/activate
# or invoke .venv_mn5/bin/python directly without activating, matching this repo's own
# local convention (this repo's .venv/bin/activate has a stale hardcoded path issue too)
```

Verify the fix worked: `python -c "import pyarrow; print(pyarrow.__version__)"` should
print the venv's version (`25.0.0` as of the last check), not the module's bundled one.

Note: skipping the module-load chain entirely (going straight to `.venv_mn5/bin/pip` or
`.venv_mn5/bin/python`) fails a different way — `error while loading shared libraries:
libpython3.11.so.1.0: cannot open shared object file` — because the venv's own Python binary
dynamically links against the module's `libpython3.11.so.1.0` (`/apps/ACC/PYTHON/3.11.5/GCC/lib`,
per the module's own `LD_LIBRARY_PATH` entry). The venv literally cannot run at all without
the full chain above loaded first, not just for package resolution.

## Confirmed working today

- `SWEBENCH_LOCAL_PATH` offline dataset loading.
- `repo_cache`'s 8 mirrored repos (astropy, django, matplotlib, pydata-xarray, pylint,
  pytest, scikit-learn, sympy) — real `git clone --bare` mirrors, so file content at any
  commit is read via `git show`/`git cat-file`, zero live GitHub API calls.
- Prompt construction and token counting.
- `mn5_smoke_test.py` (1 instance) and `mn5_pipeline_run.py` (5 instances, one per cached
  repo) both re-verified end-to-end after the environment fix above: real `code_files`
  counts and `ground_truths` returned, `cached=True` throughout, no live network calls.
- `rank_bm25==0.2.2` (a real pinned dependency of `method/bm25_retriever.py`) was missing
  from the wheelhouse entirely — downloaded locally (pure-Python, 8.6KB) and added.
- Offline Qwen3-Embedding-0.6B loading (tokenizer + model, see Blocker 4).
- A real `sbatch` submission (see "First real Slurm submission" below) — queued, ran, and
  produced live progress log output on a GPU-partition node.

## Blocker 1 — huggingface-hub version conflict — RESOLVED 2026-08-10

`transformers` needs `huggingface-hub<1.0,>=0.23.2`, but the wheelhouse had
`huggingface_hub==1.26.0` (installed earlier for an unrelated `datasets`/`pyarrow` fix).
With `--no-index` there's no PyPI fallback, so `pip install` hard-failed on this. **This
project's own local `transformers` pin was bumped from 4.46.0 to 4.51.0 this session (for
Qwen3-Embedding support) — the constraint range is identical (`<1.0,>=0.23.2`), so the
transformers-version bump did NOT resolve this blocker.** The fix had to happen on the
`huggingface-hub` side, not by picking a different transformers version.

**Fix, confirmed working on the real cluster**: downloaded `huggingface_hub==0.34.3`
(pure-Python wheel, `py3-none-any` — no platform-specific build needed) locally into
`wheelhouse_mn5/`, matching the version already pinned in `requirements-mn5.txt`, removed
the stale `1.26.0` wheel, transferred it over, and installed it — after working through the
full module-load chain above (this is what actually took the many round trips, not the
huggingface-hub install itself, which worked on the first real attempt once the modules
were actually loaded):

```bash
# from a machine with real MN5 access, this repo's own directory:
scp wheelhouse_mn5/huggingface_hub-0.34.3-py3-none-any.whl \
  comm842299@alogin1.bsc.es:/gpfs/projects/ehpc680/comm842299/repos/bug-localization/wheelhouse_mn5/

# on MN5, after the full module-load chain + unset fix above:
.venv_mn5/bin/pip install --no-index --find-links=wheelhouse_mn5 huggingface-hub==0.34.3
.venv_mn5/bin/python -c "import huggingface_hub; print(huggingface_hub.__version__)"
# -> 0.34.3, confirmed
```

## Blocker 2 — torch missing from the venv — RESOLVED 2026-08-10

The original hypothesis (system PyTorch shadowing the venv via a `.pth` file or
`PYTHONPATH` injection) was wrong. The real cause was simpler: **torch was never installed
in `.venv_mn5` at all.** Confirmed by checking `sys.path` and `.venv_mn5/site-packages`
directly rather than reasoning about env vars.

**Fix**: install `torch==2.6.0+cpu` from the wheelhouse —

```bash
.venv_mn5/bin/pip install --no-index --find-links=wheelhouse_mn5 torch==2.6.0+cpu
.venv_mn5/bin/python -c "import torch; print(torch.__file__, torch.__version__)"
# -> points into .venv_mn5/, 2.6.0+cpu, confirmed
```

`transformers`/`tokenizers` needed a matching pass too — the wheelhouse only had
`tokenizers==0.23.1`, but `transformers` (bumped to 4.51.0 locally, see blocker 1) pins
`tokenizers<0.21,>=0.20`. Fixed by downloading a platform-specific wheel locally
(`pip download tokenizers==0.20.3 --platform manylinux2014_x86_64 --python-version 311
--implementation cp --abi cp311 --only-binary=:all:`) and transferring it the same way as
the other wheels.

## Blocker 3 — this project's own code was never fully transferred to MN5

Discovered 2026-08-11, live, while trying to actually run embedding inference (not just
`import torch`). `mn5_smoke_test.py` had already passed, which only proved the **dataset/
BM25 side** of the pipeline worked — it doesn't touch `method/embedding_retriever.py` at
all. The MN5 checkout of this repo turned out to be transferred from `main` at a point in
time that **predates the hybrid-retrieval/embedding work entirely** — `main` never had
`embedding_retriever.py`, `hybrid_retriever.py`, `repository_index.py`, or
`fusion_signals.py` (those only exist on `research/embedding-model-bakeoff` and its
descendant branches), and even the pre-existing `method/bm25_retriever.py` and the
`get_file_contents_batch` function in `dataset/repo_cache.py` turned out to be missing —
an even older snapshot than `main`'s current HEAD.

**Fix, confirmed working**: pulled each missing file from the correct branch with `git
show <branch>:<path> > /tmp/<file>` on a machine with both the repo and MN5 SSH access, then
`scp`'d it directly into place on MN5 (not a heredoc paste — pasting a 300-line Python file
into an interactive SSH session corrupts it, since large multi-line pastes without
bracketed-paste support get interpreted line-by-line by bash instead of being buffered).
Files transferred this way: `method/embedding_retriever.py` (from
`research/embedding-model-bakeoff`, the most complete version — including Voyage AI and
last-token-pooling support that later branches split off before), `method/bm25_retriever.py`
and `dataset/repo_cache.py` (from `main`, both safe supersets of what was already there —
verified via `git diff` before transferring, no removed/changed functions, only additions).

**Open thread, not yet resolved**: MN5's checkout is still not a git clone at all — it's a
directory transferred by file copy, with no `.git` (confirmed via `fatal: not a git
repository`). This means there is no way to `git pull` a full sync; every gap has to be
discovered one `ModuleNotFoundError` at a time and patched file-by-file. **Before the next
MN5 session, consider transferring a full fresh tarball of whichever branch has the complete
current state** (as of 2026-08-11, `feature/incremental-indexing` has the most integrated
method/ set, though it's missing the later Voyage/last-token-pooling additions that only
landed on `research/embedding-model-bakeoff` — no single branch has everything merged) rather
than continuing to patch individual files as they're discovered missing.

## Blocker 4 — no outbound internet blocks HuggingFace model downloads — RESOLVED 2026-08-11

Once blocker 3's files were all in place, running real embedding inference failed exactly
as expected given MN5's no-internet constraint:

```
requests.exceptions.ConnectTimeout: ... Connection to huggingface.co timed out. (connect timeout=10)
```

This turned out to be the first of **five separate issues** surfaced in sequence while
getting a real `sbatch` job (Qwen3-Embedding-0.6B through `run_hybrid_rrf_weighting_test.py`)
to actually run — each fixed with the same pattern (download/resolve locally, ship the
artifact to MN5, run offline), documented here in the order they were hit:

**4a. Model weights not present offline.** Fixed: `huggingface_hub.snapshot_download(...)`
locally, tarred the resulting `hub/` cache dir (`tar -czf ... -C hf_cache_mn5 hub`,
dereferencing symlinks with `cp -RL` first so the archive is self-contained), `scp`'d it
over (1.8GB), extracted into `hf_cache_mn5/` on MN5, set `HF_HOME` to that path plus
`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`. Confirmed loading fully offline afterward.

**4b. `HF_HUB_OFFLINE=1` also silently blocks `datasets.load_dataset()`.** Not just model
weights — the dataset loader tries to reach `SWE-bench/SWE-bench_Verified` on the Hub too,
and offline mode blocks that identically (`ConnectionError: ... OfflineModeIsEnabled`). Fix:
set `SWEBENCH_LOCAL_PATH` to the already-mirrored local copy
(`hf_datasets/swebench_verified_test`) — the interactive smoke test scripts already did this
via `os.environ.setdefault(...)`, but a batch script needs it set explicitly, it isn't
implied by having the weights cached.

**4c. `run_hybrid_rrf_weighting_test.py` needs every repo in the full manifest pool, not
just the final sampled instances.** The script's `pool_size` (500 for this project's
manifests — the entire SWE-bench Verified split) determines how many raw instances
`get_bug_instances()` processes before filtering down to the manifest's actual 30 wanted
instances — and every one of those 500 triggers a `get_code_files()` call. **All 12 repos
in the full dataset need to be mirrored, not just the ~11 repos that happen to appear in
the final 30-instance sample** (checking only the manifest's own `instances` list
undercounts this). Missing `mwaskom/seaborn` surfaced this — confirmed the full dataset's
repo set with a local `SWEBench().get_bug_instances(sample_size=500)` + `set(b.repo ...)`,
mirrored the gap, transferred it (55MB, small).

**4d. `transformers==4.46.0` on MN5 doesn't recognize the `qwen3` architecture** (added in
4.51.0+, which this project's local venv already had). Fixed by downloading the
`transformers==4.51.0` wheel (`py3-none-any`, pure Python) and transferring it — but that
version also needs `tokenizers>=0.21,<0.22`, and the wheelhouse only had 0.20.3/0.23.1
(neither fits) — needed a fresh platform-specific `tokenizers==0.21.4` wheel too
(`--platform manylinux2014_x86_64 --python-version 311 --implementation cp --abi cp311`).

**4e. Loading `transformers` then tried to import a broken system-level `tensorflow`.**
`transformers` eagerly imports TF as an optional backend even when unused; MN5's
module-provided `tensorflow` has `undefined symbol: ncclMemFree` in its shared library (a
CUDA/NCCL mismatch on the node, not something to fix at the Python level). Fixed by setting
`USE_TF=0` / `USE_TORCH=1` before importing — this is a real recurring env var to set on
every future MN5 job that touches `transformers`, not a one-off.

**4f. `accelerate` (a real pinned dependency, `accelerate==1.2.1` locally) was never
installed on MN5's venv at all** — surfaced as `NameError: name 'init_empty_weights' is not
defined` deep in `transformers`' model-loading path. Fixed by downloading + transferring the
wheel; its own dependency `psutil` was *also* missing (a second, nested gap) and needed a
separate platform-specific wheel download the same way as `tokenizers`.

**Net fix, all together, confirmed working**: model loads fully offline once `HF_HOME`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `USE_TF=0`, `USE_TORCH=1`, and
`SWEBENCH_LOCAL_PATH` are all set, and `transformers==4.51.0`/`tokenizers==0.21.4`/
`accelerate==1.2.1`/`psutil` are all installed. See the `sbatch` template below for the full
working environment block.

## Blocker 5 — GPU allocated but not actually used — RESOLVED 2026-08-17

`torch==2.6.0+cpu` was installed on MN5 (a CPU-only build, installed back when first fixing
Blocker 2, before GPU use was ever the intent) — so `torch.cuda.is_available()` was always
`False` regardless of `--gres=gpu:1`, and `method/embedding_retriever.py`'s device
auto-detection silently fell back to CPU. First real symptom: a job requesting a GPU node
still took ~300-500s/instance for Qwen3-Embedding chunked embedding — the same order of
magnitude as local CPU timing, not GPU-accelerated. This nearly caused a real job to be
killed by `acc_debug`'s 2-hour wall-clock limit partway through a 30-instance sweep (~3hr
actual runtime) — resubmitted under `acc_ehpc` (3-day limit) instead as the immediate fix.

**Fix**: checked the actual hardware first (`nvidia-smi` on an interactive `salloc` GPU
session — `NVIDIA H100`, driver `595.71.05`, CUDA `13.2`), then downloaded
`torch==2.6.0+cu124` (768MB) plus its 13 `nvidia-*-cu12` runtime-library dependencies and
`triton` (another ~2.1GB total) locally via `pip download --platform ... --python-version
... --abi ...` (cross-platform download from macOS, no matching local install needed) since
MN5 has no internet. Two real tag gotchas along the way: the `nvidia-*-cu12` packages are
hosted on PyPI proper, not the `download.pytorch.org` index torch itself uses, and they're
tagged `py3-none-manylinux2014_x86_64` (not `cp311`) — different filter flags than torch's
own `cp311-linux_x86_64` tag. All 15 wheels transferred via `scp` into `wheelhouse_mn5/`
(verified byte-for-byte both ends), then `pip install --no-index --find-links=wheelhouse_mn5
torch==2.6.0+cu124` — cleanly uninstalled the old `+cpu` build and installed the CUDA one.

**Verified for real, not just detected**: `srun -A ehpc680 -q acc_debug -p acc --gres=gpu:1
--cpus-per-task=20` to an actual GPU-allocated node confirms `torch.cuda.is_available() ==
True`, `torch.cuda.get_device_name(0) == "NVIDIA H100"`, and a real 4096×4096 matmul
executes on `cuda:0` and returns a correct result — not just device detection, actual GPU
compute. Every future MN5 GPU job should now get real acceleration; the array-job-sharding
workaround built for CPU-only jobs is no longer strictly necessary, though still valid
infrastructure for wall-clock/resilience reasons independent of GPU speed.

## First real Slurm submission — 2026-08-11

Everything before this session was interactive (`module load` + direct `python` in a login
shell). The `qwen3_rrf_sweep.sbatch` template below is the first actual `sbatch` job
submitted for this project — closes the WP 6.2 gap of never having run a real batch job.

```bash
#!/bin/bash
#SBATCH --job-name=qwen3-rrf-sweep
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/qwen3_rrf_sweep_%j.out
#SBATCH --error=logs/qwen3_rrf_sweep_%j.err

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SWEBENCH_LOCAL_PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_datasets/swebench_verified_test
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_hybrid_rrf_weighting_test.py \
  --manifest results/manifests/swebench-multi-n30-s42-1fb8f4b8d82f.json \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output results/hybrid_rrf_qwen3_swebench_30_mn5.json
```

Submit with `sbatch qwen3_rrf_sweep.sbatch`, monitor with `squeue -u comm842299` /
`sacct -j <id> --format=JobID,State,Elapsed,ExitCode`, results land in
`logs/qwen3_rrf_sweep_<id>.{out,err}` and the `--output` JSON path.

**Job history from getting this right** (useful for the next person hitting the same
errors): job `44503792` failed on 4b (missing `SWEBENCH_LOCAL_PATH`), `44503928` failed on
4c (missing `seaborn`), `44503960` failed on 4d (old `transformers`), then 4e/4f were caught
and fixed *before* the next submission rather than via another failed job. `44504357`
(first fully-passing submission) was still running when caught by Blocker 5's timeout risk
and cancelled/resubmitted as `44505385` under a longer QoS.

## Reproducing the smoke test

```bash
module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH
cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
.venv_mn5/bin/python mn5_smoke_test.py
.venv_mn5/bin/python mn5_pipeline_run.py
```

## Status as of 2026-08-11 end of session

The real payoff this handbook was building toward — running `run_hybrid_rrf_weighting_test.py`
at an `n` beyond what's practical locally — is in progress, not yet landed: job `44505385`
(Qwen3-Embedding-0.6B RRF sweep, n=30) was submitted and running when this session ended,
projected ~3 hours on CPU (see Blocker 5). Check `squeue -u comm842299` / `sacct -j 44505385`
next session; if it completed, the result is at
`results/hybrid_rrf_qwen3_swebench_30_mn5.json` on MN5 and needs pulling back
(`scp mn5:/gpfs/projects/ehpc680/comm842299/repos/bug-localization/results/hybrid_rrf_qwen3_swebench_30_mn5.json results/`)
and writing up in `docs/qwen3_rrf_result.md` (currently only has the n=6 local result). If
it failed or got killed, `sacct` will show why — check the wall-clock QoS limit first before
assuming a new bug.

## Update 2026-08-19/20: Ollama+GPU deployment, real array-job patterns, telemetry, a real bug found+fixed, and a reproducibility checklist

Everything below is confirmed working on real jobs this session (BM25 full-population job
`44806072`, IQLoc-approximation n=200 job `44814164`, both GPU-accelerated). Commands are
written to be copy-pasted directly — per project decision 2026-08-20, MN5 commands are run by
the user, not executed by Claude directly.

### Ollama deployment on MN5 — real, GPU-accelerated, offline

Module + model weights are already staged (`ollama_models/` in the project dir, transferred
from a local `ollama pull` in an earlier session). To verify it's still working:

```bash
ssh mn5
cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
module load ollama/0.11.8
```

Then, inside a real GPU allocation (Ollama needs the GPU node, not the login node):

```bash
srun -A ehpc680 -q acc_debug -p acc --gres=gpu:1 --cpus-per-task=20 --time=00:10:00 bash -c '
module load ollama/0.11.8
export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models
ollama serve > /tmp/ollama_check.log 2>&1 &
OLLAMA_PID=$!
sleep 6
curl -s http://localhost:11434/api/tags
echo
time curl -s http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen2.5-coder:7b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}"
kill $OLLAMA_PID
'
```

Expect a real chat completion in ~5s on the H100 (vs. up to 460s/instance observed running
the same model on a CPU-only local Mac) — confirms both the model and the GPU path.

### Array-job pattern for any Ollama-dependent job

Ollama has **no shared-server mode across nodes** — every Slurm array task must start its own
local `ollama serve`, wait for it to actually respond before sending requests (model load onto
the GPU takes a few seconds), then kill it at the end. Template (see
`scripts/mn5/bench4bl_iqloc_approximation_array.sbatch` in the repo for the full working
version):

```bash
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-N

module load intel mkl impi hdf5 python/3.11.5-gcc ollama/0.11.8
unset PYTHONHOME PYTHONPATH
export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models

ollama serve > "logs/ollama_serve_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 30); do
  curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200 && break
  sleep 2
done

# ... run the actual shard script here ...

# IMPORTANT: capture the real exit code before the kill, or a normally-exited-but-
# already-dead Ollama server's nonzero kill status silently becomes the job's FAILED status.
# Confirmed on job 44814164: 9/20 tasks (45%) wrongly marked FAILED in sacct despite writing
# valid results, before this fix.
PIPELINE_STATUS=$?
kill "$OLLAMA_PID" 2>/dev/null
exit "$PIPELINE_STATUS"
```

**`--cpus-per-task=20` is required whenever `--gres=gpu:1` is requested** — Slurm rejects
anything below `nodes * gpus/node * 20` ("Minimum cpus requested should be...").

### Real telemetry — `sacct` after any array job

```bash
sacct -j <JOBID> --format=JobID,JobName,Partition,Elapsed,State,ExitCode -X
```

Real numbers from this session: BM25 full-population job (50 tasks) — 28m46s wall-clock
(first task start → last task end), mean 9m/task. IQLoc n=200 job (20 tasks) — 5m56s
wall-clock, mean 3m33s/task. **Check `ExitCode` and `State` per task, not just whether the
job "finished"** — see the false-FAILED bug above; a task can write valid output and still
show `FAILED` if the job script's own exit-code handling is wrong.

### Local↔MN5 determinism check

```bash
# On MN5, inside a GPU allocation (same module-load chain as above):
.venv_mn5/bin/python scripts/check_local_mn5_determinism.py --n 2 --candidate-pool-size 30 --output results/determinism_mn5.json
```

Run the same command locally (`.venv/bin/python scripts/check_local_mn5_determinism.py ...`)
and diff the printed `sha256` hashes per instance — they should match exactly. Uses
Qwen3-Embedding (fully local/offline) rather than this project's actual OpenAI-based
confirmed-best config, since MN5 has no outbound internet and an OpenAI-dependent pipeline
cannot run there at all.

### Reproducibility checklist (mirrors the supervisor's expected format)

- [ ] `ssh mn5` succeeds, `pwd` after `cd` lands in `/gpfs/projects/ehpc680/comm842299/repos/bug-localization/`
- [ ] `module load intel mkl impi hdf5 python/3.11.5-gcc` succeeds with no missing-prerequisite errors
- [ ] `unset PYTHONHOME PYTHONPATH` run after the module load, every session
- [ ] `.venv_mn5/bin/python -c "import torch; print(torch.cuda.is_available())"` inside a
      `--gres=gpu:1` allocation prints `True`
- [ ] `module load ollama/0.11.8; which ollama` resolves to `/apps/ACC/OLLAMA/0.11.8/bin/ollama`
- [ ] `curl http://localhost:11434/api/tags` (after `ollama serve` + `OLLAMA_MODELS` export)
      lists `qwen2.5-coder:7b`
- [ ] Any new array sbatch script captures the pipeline's real exit code before killing
      Ollama, not just `kill $OLLAMA_PID` as the last line
- [ ] `sacct -j <id> -X` checked per-task, not just overall job state

### Minimal rerun commands — full BM25 comparison, full population

```bash
ssh mn5
cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_bm25_full/shards
sbatch scripts/mn5/bench4bl_bm25_comparison_full_array.sbatch
squeue -u comm842299
# once all 50 tasks show COMPLETED in: sacct -j <jobid> -X --format=JobID,State
module load intel mkl impi hdf5 python/3.11.5-gcc
unset PYTHONHOME PYTHONPATH
.venv_mn5/bin/python scripts/aggregate_bm25_shards.py --shard-dir results/hpc_mn5_bm25_full/shards --num-shards 50 --output results/bm25_comparison_bench4bl_full4418.json
```

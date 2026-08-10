# MareNostrum 5 execution handbook

Practical reference for running this repo's offline retrieval pipeline on MareNostrum 5
(MN5, BSC). Consolidates everything established across real access sessions to date —
account details, environment setup, four blockers found and diagnosed so far (three
resolved, one open), and what's already confirmed working. This is a **final deliverable
named explicitly** in the official study plan; it did not exist as a document before this
pass, only as scattered session notes.

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

No SSH access to any of the above exists from the machine this handbook was written on —
verify connectivity from whatever machine will actually run these steps before assuming any
of the rest of this document is directly executable there and then.

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

## Blocker 4 — no outbound internet blocks HuggingFace model downloads — CONFIRMED, OPEN

Once blocker 3's files were all in place, running real embedding inference (`embed_texts`
with `microsoft/unixcoder-base`) failed as expected given MN5's no-internet constraint
(stated at the top of this handbook, but untested against this specific code path until
now):

```
requests.exceptions.ConnectTimeout: ... Connection to huggingface.co timed out. (connect timeout=10)
...
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it
in the cached files ...
```

`AutoTokenizer.from_pretrained` / `AutoModel.from_pretrained` always try to hit the Hub first
unless the model is already in the local HF cache — there is no offline fallback by default.

**Next step, not yet tried**: pre-download the model weights locally (e.g. via
`huggingface_hub.snapshot_download("microsoft/unixcoder-base")` or
`git clone https://huggingface.co/microsoft/unixcoder-base`), transfer the resulting cache
directory to MN5 the same way as the wheels/code files, and either point
`HF_HOME`/`TRANSFORMERS_CACHE` at the transferred location or set `HF_HUB_OFFLINE=1` so
`from_pretrained` reads the local cache instead of attempting a network call. This is the
same pattern as every other MN5 blocker so far — download once with internet access, ship
the artifact over, run offline.

## Reproducing the smoke test once both blockers are cleared

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

Once blocker 4 (model weights) is cleared, the next real milestone is adapting
`scripts/compare_embedding_models.py` / `scripts/run_hybrid_rrf_weighting_test.py` (both
already dataset/manifest-driven, no code changes needed beyond pointing them at MN5's
paths and making sure their imports are fully present per blocker 3) to run at an `n` far
beyond what's practical locally — that's the actual payoff this handbook is building
toward, not just getting `import torch` to succeed.

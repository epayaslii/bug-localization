# MareNostrum 5 execution handbook

Practical reference for running this repo's offline retrieval pipeline on MareNostrum 5
(MN5, BSC). Consolidates everything established across real access sessions to date —
account details, environment setup, the two open blockers and how to diagnose them further,
and what's already confirmed working. This is a **final deliverable named explicitly** in
the official study plan; it did not exist as a document before this pass, only as scattered
session notes.

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

## Open blocker 2 — system PyTorch shadows the venv

`import torch` resolves to an NVIDIA-container-style GPU build baked into the ACC module
tree (`/gpfs/apps/MN5/ACC/PYTORCH/2.4.0`, version `2.4.0a0+gite3b9b71`) and fails with
`ImportError: libcudnn.so.9: cannot open shared object file` — **even after the same
`unset PYTHONHOME`/`unset PYTHONPATH` fix that resolved the pyarrow shadowing above.** Not
yet diagnosed why the same fix didn't work here.

**Next step, not yet tried**: check `sys.path` directly inside the venv's Python (not just
the env vars) —

```bash
python -c "import sys; print('\n'.join(sys.path))"
```

— to see whether something *beyond* `PYTHONPATH` is injecting the ACC module's torch path:
a `.pth` file inside the venv's `site-packages` (some HPC module systems auto-install one
on activation), or a separately auto-loaded module dependency the `python` module itself
pulls in regardless of `PYTHONPATH`. If a `.pth` file is the cause, removing or editing it
inside `.venv_mn5` should be a one-time fix (not needing to be repeated every login, unlike
the `unset` workaround above).

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

Once torch/transformers actually import cleanly, the next real milestone is adapting
`scripts/compare_embedding_models.py` / `scripts/run_hybrid_rrf_weighting_test.py` (both
already dataset/manifest-driven, no code changes needed beyond pointing them at MN5's
paths) to run at an `n` far beyond what's practical locally — that's the actual payoff this
handbook is building toward, not just getting `import torch` to succeed.

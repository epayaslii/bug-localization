# Bench4BL — first result

First real result on Bench4BL (`github.com/exatoa/Bench4BL`, ISSTA 2018 reproducibility
study), following up on the scoping notes in `docs/next_steps.md`. New loader
(`dataset/bench4bl.py`) and mirror script (`scripts/mirror_bench4bl.py`), no legacy Python 2
tooling required at any point — see "How this works" below.

## Result (n=30, BM25 retrieval-only, free/offline)

`results/manifests/bench4bl-multi-n30-s42-9449c3b8a675.json` (30 instances, 3 distinct
repos, drawn from a pool of 133 usable instances across 5 mirrored projects: WEAVER,
CRYPTO, CODEC, IO — WFMP contributed 0, see below), via
`scripts/compare_bm25_representations.py`:

| Representation | Hit@1 | Hit@5 | Hit@10 | MRR | MAP |
|---|---:|---:|---:|---:|---:|
| path_only | 23.3% | 50.0% | 60.0% | 0.3564 | 0.3519 |
| skeleton | 23.3% | 50.0% | 60.0% | 0.3564 | 0.3519 |
| symbols_with_imports | 23.3% | 50.0% | 60.0% | 0.3564 | 0.3519 |
| symbols_no_imports | 23.3% | 50.0% | 60.0% | 0.3564 | 0.3519 |

**All four representations are identical** — this is the already-documented Python-only AST
caveat (symbol/chunk extraction silently degrades to path-only fallback on non-Python code),
not a bug. Bench4BL is 100% Java; our AST-based skeleton/symbol extraction only parses
Python. Same pattern already confirmed on BeetleBox's Go/JS instances. Directional at n=30,
same small-sample caveat as every other n=30 result in this project.

## Known coverage gap: WFMP

The Wildfly WFMP project mirrored cleanly but contributed 0 usable instances — its bug
reports reference fine-grained pre-release tags (`1.1.0.Alpha3`, `1.1.0.Alpha5`, ...) that
`versions.txt` doesn't map to any known git tag (`versions.txt` only tracks 4 "final"
release versions). Real data-coverage limitation in that project's version file, not a
loader bug — the loader correctly skips bugs it can't map to a real commit rather than
guessing. Worth checking whether other Bench4BL projects have the same gap once more are
mirrored.

## How this works

Unlike SWE-bench/BeetleBox (clean HuggingFace datasets), Bench4BL has no single downloadable
dataset — but each per-project SourceForge archive turns out to already contain the fully
processed output of the project's own legacy Python 2 pipeline: a real git repo
(`gitrepo/`, tags matching `versions.txt`) and a pre-generated bug-repository XML
(`bugrepo/repository.xml`) with summary/description/version/fixedVersion/fixedFiles per bug.
So the legacy Python 2.7 + Java + Indri toolchain scoped as a risk in `docs/next_steps.md`
turned out to be unnecessary entirely — `scripts/mirror_bench4bl.py` just downloads and
extracts the archive, and `dataset/bench4bl.py` is a plain Python 3 XML parser + `git
ls-tree`/`git show` reader, same shape as the existing loaders.

One real wrinkle: `fixedFiles` entries are dotted Java package+class names
(`org.foo.Bar.java`), not real file paths — Maven multi-module projects nest the actual
file under a module-specific `src/main/java/` root, so the dotted form is only a *suffix*
of the true path. `Bench4BL._resolve_dotted_path()` converts dots to slashes and searches
the tree at that bug's commit for a file ending in that suffix. Verified correct against a
real example (`org.apache.commons.weaver.normalizer.Normalizer.java` ->
`modules/normalizer/src/main/java/org/apache/commons/weaver/normalizer/Normalizer.java`).

## Mirrored so far

5 of 51 projects (WEAVER, CRYPTO, WFMP, CODEC, IO — the smallest, for a fast first
validation pass), ~23MB total. Total dataset size across all 51 projects is ~5.6GB
(confirmed via the SourceForge RSS file listing) — small enough to mirror the whole thing,
not yet done.

## Reproducing

```bash
python scripts/mirror_bench4bl.py --projects WEAVER CRYPTO WFMP CODEC IO
# or: python scripts/mirror_bench4bl.py --all   (~5.6GB total)

python scripts/generate_evaluation_manifest.py --dataset bench4bl --size 30 --pool-size 133 --seed 42 --max-per-repo 15
python scripts/compare_bm25_representations.py --manifest results/manifests/bench4bl-multi-n30-s42-9449c3b8a675.json --output results/bm25_comparison_bench4bl_30.json
```

## Next steps

- Mirror the remaining 46 projects (~5.5GB more) for real dataset-wide coverage.
- No test coverage yet for `dataset/bench4bl.py` (`tests/test_dataset_beetlebox.py` is the
  pattern to follow) — not yet done.
- Not yet wired into `main.py`'s end-to-end (LLM rerank) path, only the free/offline
  retrieval-only comparison scripts so far.

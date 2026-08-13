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

## Hybrid result (n=30, BM25 + Qwen3-Embedding-0.6B + weighted RRF, MN5)

Follow-up run through the same weighted-RRF pipeline validated on SWE-bench
(`results/hybrid_rrf_qwen3_swebench_30_mn5.json`, MRR 0.4216 at 1:5) — first time Bench4BL
was run through embedding/hybrid retrieval rather than BM25-only. `scripts/run_hybrid_rrf_weighting_test.py`
gained `--dataset bench4bl` support (mirrors the existing `swebench`/`beetlebox` branches).
Job `44552836` on MN5 (`acc_debug`, ~64s total for all 30 instances — see "Why so fast"
below), full sweep (bm25-alone, embedding-alone, RRF 1:1 through 1:50):

| Config | Hit@1 | Hit@5 | Hit@10 | Hit@100 | MRR | MAP |
|---|---:|---:|---:|---:|---:|---:|
| bm25 alone | 3.3% | 23.3% | 36.7% | 86.7% | 0.1419 | 0.0920 |
| **chunked_embedding (Qwen3) alone — best** | 40.0% | 73.3% | 76.7% | 93.3% | **0.5424** | 0.4159 |
| rrf_1_1 | 33.3% | 70.0% | 70.0% | 90.0% | 0.4547 | 0.2863 |
| rrf_1_5 | 40.0% | 70.0% | 76.7% | 93.3% | 0.5176 | 0.3977 |
| rrf_1_15 | 40.0% | 70.0% | 76.7% | 93.3% | 0.5311 | 0.4138 |
| rrf_1_30 | 40.0% | 73.3% | 76.7% | 93.3% | 0.5351 | 0.4137 |
| rrf_1_50 | 40.0% | 73.3% | 76.7% | 93.3% | 0.5424 | 0.4159 |

**Verdict: no RRF weighting beats embedding-alone on Bench4BL** — the opposite of SWE-bench,
where weighted RRF (1:5) beat embedding-alone (0.4216 vs. 0.3165). Two real findings, not
one: Qwen3-Embedding is a strong reranker on both benchmarks (0.5424 MRR here vs. 0.4216 on
SWE-bench Verified — actually higher), but whether BM25+embedding fusion helps or just adds
noise is benchmark-dependent, not a universal property of the method. Don't average the two
verdicts together.

**Why the high-weight configs converge to the same numbers**: RRF's fused score per file is
`sum(weight_i / (k + rank_i))`, k=60. BM25's contribution is bounded — ranks only run 1 to
the candidate pool size (200), so its max possible term is `1/61` and its floor is `1/260`, a
narrow fixed band regardless of weight elsewhere. Once the embedding weight is scaled high
enough (here, by 1:50) that its own term spread swamps that whole BM25 band, BM25 can no
longer flip any file's relative order — the fused ranking becomes bit-identical to sorting by
embedding rank alone. Confirmed exactly, not approximately: rrf_1_50's MRR/MAP (0.5424/0.4159)
match chunked_embedding-alone to 4 decimal places, meaning the actual rankings are identical,
not just close in aggregate. rrf_1_30 is close but not there yet (0.5351/0.4137) — a few
borderline instances still get nudged by BM25 at that weight. Same convergence pattern already
seen on the SWE-bench sweep (1:50 converging back to embedding-alone there too) — expected
behavior of weighted RRF, not specific to this benchmark.

**Why the job ran so fast (~64s for all 30 instances, vs. ~5.5min/instance on SWE-bench)**:
every per-instance log line shows chunk count exactly equal to file count (e.g. `188 chunks
over 188 files`, `56 chunks over 56 files`) — a real consequence of the same Python-only AST
caveat already documented above for BM25 representations, this time hitting the embedding
chunker instead. `_chunk_file_content()` in `method/embedding_retriever.py` tries `ast.parse()`
first (splits a file into one chunk per top-level function/class, the fine-grained path);
Java throws `SyntaxError` on every file, so it falls through to the fixed-1500-character-window
fallback. Apache Commons utility-class files (IO/Codec/Weaver) are mostly well under 1500
characters, so that fallback yields exactly one chunk per file — versus SWE-bench's Python
files, where the AST path can produce 10+ chunks from one file with several methods. Fewer
files (BM25 top-200 barely narrows repos that only have 41-198 files total) times one coarse
chunk each, instead of many fine-grained ones, is most of the speed difference. **Caveat this
puts on the result above**: "chunked embedding" on Bench4BL isn't really getting the
fine-grained benefit the technique is designed for (the docstring's own cited paper shows
chunked embedding beating whole-file embedding 33-71% vs. 3-12% Acc@10) — it's closer to
whole-file embedding in disguise here, since most files never get split at all.

**Manifest note**: the original n=30 manifest (`bench4bl-multi-n30-s42-9449c3b8a675.json`,
used for the BM25-only result above) was generated when only 5 projects were mirrored. By the
time this hybrid run happened, 4 more projects had been mirrored (AMQP/ANDROID/BATCH/BATCHADM),
changing the pool the same seed samples from — a first attempt against the old manifest
silently matched only 21/30 instances on MN5 (all 9 missing ones were CODEC). Fixed by
generating a new manifest directly on MN5, against MN5's actual current pool (`--pool-size 250`,
comfortably above the 213-instance total so `random.sample` isn't triggered by a pool-size
mismatch again as more projects get mirrored later): `results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json`,
30 instances, 4 distinct repos, drawn from 213 instances across 6 contributing repos.

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

9 of 51 projects locally (AMQP, ANDROID, BATCH, BATCHADM, CODEC, CRYPTO, IO, WEAVER, WFMP),
182MB. **Local and MN5 have diverged**: MN5 only has 8 of these usable — BATCH's `gitrepo/`
(124MB, the largest of the 9) repeatedly failed to transfer over SSH (connection drops
partway through, unresolved), so only its `bugrepo/`+`versions.txt` made it across. The
loader handles this gracefully (skips BATCH entirely on MN5 with a warning, doesn't crash),
but it does mean **local's total usable-instance pool (467) and MN5's (213) are different
sizes** — see the pool-size note under "Reproducing" below, this matters for exact
reproduction. Total dataset size across all 51 projects is ~5.6GB (confirmed via the
SourceForge RSS file listing) — small enough to mirror the whole thing, not yet done.

## Reproducing

```bash
python scripts/mirror_bench4bl.py --projects WEAVER CRYPTO WFMP CODEC IO
# or: python scripts/mirror_bench4bl.py --all   (~5.6GB total)

# BM25-only (retrieval, free/offline) -- original n=30 result at the top of this doc
python scripts/generate_evaluation_manifest.py --dataset bench4bl --size 30 --pool-size 133 --seed 42 --max-per-repo 15
python scripts/compare_bm25_representations.py --manifest results/manifests/bench4bl-multi-n30-s42-9449c3b8a675.json --output results/bm25_comparison_bench4bl_30.json

# Hybrid: BM25 + Qwen3-Embedding-0.6B + weighted RRF -- the "Hybrid result" section above
python scripts/run_hybrid_rrf_weighting_test.py \
  --dataset bench4bl \
  --manifest results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json \
  --pool-size 500 \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output results/hybrid_rrf_qwen3_bench4bl_30_local.json
```

**`--pool-size 500` on the hybrid command is required, not optional, if your local mirror
includes BATCH** (i.e. any local checkout that hasn't hit the same MN5 transfer failure).
`run_hybrid_rrf_weighting_test.py` re-derives its instance pool at runtime via
`Bench4BL().get_bug_instances(sample_size=pool_size, random_sample=True, seed=42)` — if
`sample_size` is smaller than the environment's actual total instance count, it triggers
`random.sample()` over that environment's specific instance list, and a different total
(local's 467 vs. MN5's 213) means the *same seed samples a different subset*, silently
missing manifest instances rather than erroring. Confirmed by testing locally without the
override: only 18/30 matched (12 missing, all AMQP — a different project than MN5's original
21/30 mismatch, but the identical root cause). With `--pool-size 500` (comfortably above
local's 467 total), sampling is skipped entirely (deterministic full-pool return), all 30/30
match, and the result reproduces MN5's numbers exactly (chunked_embedding MRR 0.5424 both
places, verified to 4 decimal places). This is why `--pool-size` was added as a real CLI
override in this session — it previously only came from the manifest's stored value, with no
way to correct for an environment-size mismatch after the fact.

## Java support — not yet adapted

The AST-based chunker (`_chunk_file_content` in `method/embedding_retriever.py`) and the
symbol/skeleton BM25 representations both only parse **Python** (`ast.parse()`). Neither has
been adapted for Java. On Bench4BL this shows up two ways: (1) all 4 BM25 representations
tie exactly (symbol/skeleton extraction silently falls back to path-only, see "Result" above)
and (2) the embedding chunker falls back to fixed 1500-character windows instead of
per-method/class chunks, which for Bench4BL's mostly-short Java files collapses to one coarse
chunk per file (see "Hybrid result" above) — faster to run, but not exercising the
fine-grained chunking the technique is designed around. A real fix would mean adding a
Java-aware parser (e.g. `javalang` or `tree-sitter-java`) alongside the existing Python `ast`
path in both places — not started, no library chosen yet.

## Next steps

- Mirror the remaining 46 projects (~5.5GB more) for real dataset-wide coverage.
- Retry BATCH's `gitrepo/` transfer to MN5 (compress first, or try `rsync` instead of `scp`
  for resumability) so local and MN5's pools match again.
- Add Java-aware chunking/symbol-extraction (see "Java support" above) so BM25
  representations stop tying and the embedding chunker gets real per-method granularity.
- No test coverage yet for `dataset/bench4bl.py` (`tests/test_dataset_beetlebox.py` is the
  pattern to follow) — not yet done.
- Not yet wired into `main.py`'s end-to-end (LLM rerank) path, only the free/offline
  retrieval-only comparison scripts so far.

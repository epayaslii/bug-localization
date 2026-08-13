# Bench4BL — first result

First real result on Bench4BL (`github.com/exatoa/Bench4BL`, ISSTA 2018 reproducibility
study), following up on the scoping notes in `docs/next_steps.md`. New loader
(`dataset/bench4bl.py`) and mirror script (`scripts/mirror_bench4bl.py`), no legacy Python 2
tooling required at any point — see "How this works" below.

## Corrected: the original "all four tie" result was never really testing content at all

**The explanation originally written here (Python-only AST parsing) was wrong.** The real
cause, found while adding Java support (see "Java support" below): `bm25_retriever.py` and
`embedding_retriever.py` fetch file content through `dataset/repo_cache.py`'s
`is_repo_cached()`/`get_file_contents_batch()`, which only know how to look in `repo_cache/`
— the directory SWE-bench/BeetleBox mirror into. Bench4BL's content lives in
`bench4bl_cache/<PROJECT>/gitrepo` instead, read directly by `dataset/bench4bl.py`'s own git
calls, which `repo_cache.py` had no idea existed. So `is_repo_cached("WEAVER")` was always
`False`, content was never fetched for **any** Bench4BL bug, and every representation
(BM25 skeleton/symbols, embedding chunking) silently fell back to path-only tokens or a
single path-token pseudo-chunk — not because Java failed to parse, but because no Java (or
any) content ever reached the parsing step in the first place. Fixed by extending
`is_repo_cached()`/`get_file_contents_batch()` in `dataset/repo_cache.py` with a Bench4BL
fallback path (same `git cat-file --batch` mechanism, pointed at
`bench4bl_cache/<repo>/gitrepo/.git` instead of a bare clone). Verified: content now fetches
correctly (spot-checked against 3 real bugs, all files readable), and the "Java support"
section below adds a real Java-aware lexical scanner on top so the now-reachable content
actually gets parsed into real symbol/skeleton tokens and per-method chunks. The BM25 and
hybrid results below are the corrected numbers, run after both fixes.

## Result (n=30, BM25 retrieval-only, free/offline)

`results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json` (30 instances, drawn from the
current 8-usable-project pool — see "Manifest note" further below), via
`scripts/compare_bm25_representations.py --pool-size 500`, run both locally and on MN5
(job `44554116`, matched exactly):

| Representation | Hit@1 | Hit@5 | Hit@10 | MRR | MAP |
|---|---:|---:|---:|---:|---:|
| path_only | 16.7% | 33.3% | 46.7% | 0.2716 | 0.2398 |
| **skeleton — best** | 30.0% | 50.0% | 73.3% | **0.4129** | 0.3306 |
| symbols_with_imports | 23.3% | 56.7% | 83.3% | 0.3658 | 0.3027 |
| symbols_no_imports | 20.0% | 46.7% | 70.0% | 0.3169 | 0.2759 |

All four representations now genuinely differ, and all three content-aware ones beat
path_only by a wide margin — real signal, not the old artifact-tie. Directional at n=30,
same small-sample caveat as every other n=30 result in this project.

## Hybrid result — VOID original n=30 number, corrected but too-small n=6 replacement, real n=30 rerun still outstanding

**The original n=30 hybrid number below (0.5424 MRR, "chunked_embedding alone wins") is
invalid, for the exact same root cause as the BM25 tie above.** `rank_files_embedding_chunked()`
in `method/embedding_retriever.py` also fetches content through
`is_repo_cached()`/`get_file_contents_batch()` — the same functions that were always `False`
for Bench4BL before the `dataset/repo_cache.py` fix. So every "chunk" in that run was actually
the fallback pseudo-chunk (`" ".join(_tokenize_path(path))` — the tokenized file *path*, zero
real code content), not a real code embedding at all. The "why the job ran so fast" explanation
originally written below (short Java files, 1500-char windows) was **also wrong** — it wasn't
about chunk size, no code was ever read in the first place. This was found while writing up
this correction, after already publishing the number. **Original (void) numbers, kept only for
the record**:

| Config | MRR (void — path-tokens only, not real content) |
|---|---:|
| bm25 alone | 0.1419 |
| chunked_embedding (Qwen3) alone | 0.5424 |
| rrf_1_50 | 0.5424 (bit-identical ranking to embedding-alone, see RRF-convergence note further below — that math still holds, only the *input* was wrong) |

**Corrected (real content + Java lexical scanner) result exists only at n=6** — small-scale
sanity check before a real n=30 rerun, run after the content-fetch fix, same manifest as the
corrected BM25 result (`bench4bl-multi-n30-s42-mn5-8proj.json`'s first 6 instances: 4x IO,
2x AMQP):

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | MAP |
|---|---:|---:|---:|---:|---:|
| bm25 alone | 0% | 16.7% | 66.7% | 0.1641 | 0.1231 |
| chunked_embedding (Qwen3) alone | 50.0% | 100% | 100% | 0.7500 | 0.6796 |
| **rrf_1_1 (unweighted) — best** | 66.7% | 100% | 100% | **0.8333** | 0.5468 |
| rrf_1_5 | 66.7% | 100% | 100% | 0.8333 | 0.6889 |

**This flips the (void) original finding**: now unweighted RRF beats embedding-alone, not the
other way around. But n=6 is far too small to trust on its own — this project has hit the
exact same "small n gives a different answer than n=30" pattern twice already (SWE-bench
n=6→n=30, and this same Bench4BL manifest's earlier n=21/n=30 pool-size mismatch). **Read this
n=6 run as "the pipeline works correctly end-to-end with real content" (confirmed — real
per-method Java chunk counts, e.g. 2302-2953 chunks over 200 files, a completely different
order of magnitude from the void run's 1-chunk-per-file path-token fallback), not as a
trustworthy MRR number.** A real n=30 hybrid rerun with the fixed content-fetching + Java
scanner is the honest next step, not yet done as of this writing — see "Next steps" below.

**Why the n=6 job took ~33 minutes on MN5 this time (vs. the void run's ~64s for all 30)**:
directly confirms the root-cause diagnosis above. Real per-method chunking on real Java content
means far more actual embedding calls per instance (e.g. `AMQP-468: 2302 chunks over 200
files`, ~6.5min just for that one instance) — completely different computational profile than
embedding a single tokenized path string per file. A first attempt at the real n=30 rerun
(before scaling down to n=6) was cancelled after discovering the first instance alone took 917s
— at that rate 30 instances would have blown MN5's 3hr job budget and produced zero output
(this script only writes its output once, at the end).

**RRF-convergence math (this part of the original writeup remains correct, just re-verify
against real numbers once the n=30 rerun exists)**: RRF's fused score per file is
`sum(weight_i / (k + rank_i))`, k=60. BM25's contribution is bounded — ranks only run 1 to the
candidate pool size (200), so its max possible term is `1/61` and its floor is `1/260`, a
narrow fixed band regardless of weight elsewhere. Once the embedding weight is scaled high
enough that its own term spread swamps that whole BM25 band, BM25 can no longer flip any file's
relative order — the fused ranking becomes bit-identical to sorting by embedding rank alone.
This was confirmed exactly (not approximately) in the void run's own numbers — rrf_1_50 matched
chunked_embedding-alone to 4 decimal places — and is expected behavior of weighted RRF
regardless of what's being embedded, so the mechanism itself isn't in question, only which
config wins on real content.

**Manifest note**: the n=30 manifest used for the BM25-only result above
(`bench4bl-multi-n30-s42-mn5-8proj.json`) was regenerated mid-session after discovering the
*original* n=30 manifest (`bench4bl-multi-n30-s42-9449c3b8a675.json`, generated when only 5
projects were mirrored) silently matched only 21/30 instances once 4 more projects had been
mirrored (all 9 missing ones were CODEC) — the same seed samples a different subset once the
underlying pool's size changes. Fixed by generating fresh, directly on MN5, against MN5's
actual current pool (`--pool-size 250`, comfortably above the 213-instance total so
`random.sample` isn't triggered by this kind of mismatch again as more projects get mirrored):
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

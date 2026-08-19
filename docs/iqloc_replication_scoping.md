# Scoping: IQLoc publication replication

Branch `iqloc-publication-replication`, mirroring the co-intern's branch of the same name
(`github.com/adisenaa/Bug-Report-Localization`). Scopes what a real replication of **IQLoc**
(`arxiv.org/html/2510.04468v2`, "Improving IR-based bug localization with semantics-driven
query reduction", JSS 2026) would take, before writing any code.

This is a distinct pathway from both of this project's existing tracks: the co-intern's
frozen-BM25-only baseline (`docs/presentations/bug_localization_progress_2026-08-17.pptx` on
their repo — no dense/hybrid retrieval, full-population P=4,621 eval, MN5-reproducibility
focused), and this project's own hybrid-RRF + Ollama relevance-feedback pipeline (embedding-
based hybrid retrieval, LLM chunk-level relevance judgment, raw-identifier-token
reformulation). IQLoc's actual published pipeline matches neither.

## The real IQLoc pipeline (from the paper, not the one-line summary)

1. **Retrieval**: Elasticsearch BM25 (default k/b), top-K=100, over the bug's correct
   project/version (not just latest).
2. **Cross-encoder relevance scoring**: CodeBERT base + a classification head, **fine-tuned**
   on Bench4BL-derived (bug report, method) pairs — positives = methods changed in the
   buggy→fixed diff (AST-parsed), negatives = random methods from other systems, 4:1 ratio.
   LR 1e-4, `ReduceLROnPlateau` (patience 2), batch size 8, 512 token limit, up to 100 epochs
   (stopped on plateau). Threshold 0.5 (~91-93% accuracy, ~81-89% precision on their splits).
   Every method in every Stage-1 document gets scored [0,1].
3. **Keyword extraction (bug report side)**: EmbedRank (via KeyBERT) using **CodeT5-small**
   embeddings — CodeT5 itself further pre-trained on 70,884 bug reports from 100 Java repos
   (MLM, 15% masking) before use here. MMR λ=0.5, top-15 keywords (swept 5/15/43 — 15 is the
   knee of the curve, see ablation below).
4. **Keyword extraction (code side)**: same EmbedRank/MMR process, but only over segments the
   cross-encoder scored >0.5.
5. **Query reformulation**: cosine similarity between bug-report keywords and code keywords
   (CodeT5 embeddings) to find overlapping/similar terms; original bug report query is
   expanded with the top-matching code-side keywords.
6. **Final rerank**: BM25 again, over the same Stage-1 top-100, with the reformulated query.

Two components not obvious from the abstract: **CodeT5 is domain-pretrained before use as the
keyword-extraction embedder** (this alone was worth 7-8% at N=5, per their ablation), and the
**cross-encoder is a real fine-tuned classifier, not zero-shot** (unlike BRaIn's zero-shot LLM
judge, which this project's own pipeline is closer to).

## Dataset

- **7,483 bug reports, 42 subject systems, 1,578 buggy/fixed versions** (Apache 11 systems/
  3,145 reports, JBoss 6/1,416, Spring 25/2,922) — their own refined+expanded Bench4BL, not
  the raw SourceForge-archived 51-project benchmark this project mirrors as-is.
- Ground truth is **method-level** (AST-parsed diff), average ~2 changed files/bug (1-7
  range) — finer-grained than this project's file-level ground truth.
- Two splits reported: random 70:10:20 (5 runs averaged) and time-wise 70:10:20
  (chronological) — time-wise is the more realistic/harder setting and the one already cited
  in `docs/sota_comparison.md` (MRR 0.553).
- Bug reports stratified by type: Stack Trace (~28%), Program Element (~30%), Natural-
  Language-only (~42%) — IQLoc's gain is heavily concentrated in ST/PE (+15-33%), much weaker
  on NL-only (+4-9%). Any comparison against our own numbers should control for this mix.

## Headline results (time-wise split, Table 7/9/10 in the paper)

| | MAP | MRR | Hit@1 | Hit@5 | Hit@10 |
|---|---|---|---|---|---|
| Elasticsearch BM25 (their baseline) | 0.471 | 0.496 | 0.396 | – | – |
| IQLoc (full pipeline) | 0.520 | 0.553 | 0.466 | 0.669 | 0.735 |
| IQLoc, Stack-Trace reports only | 0.565 | 0.601 | 0.535 | – | – |
| IQLoc, Program-Element reports only | 0.689 | 0.737 | 0.674 | – | – |
| IQLoc, NL-only reports | 0.460 | 0.487 | 0.382 | – | – |

Also beats 8 other published baselines (BLUiR, Blizzard, BRTracer, BLIA, LRBL, LLmiRQ,
DNNLoc, RLocator) on the random split, all differences Wilcoxon-significant (p<0.05).

## What's actually reusable vs. what needs rebuilding

Checked the paper's cited replication bundle (`github.com/asifsamir/IQLoc`) directly — **the
paper's claim of released "trained cross-encoder models" and "pre-trained CodeT5 models" does
not hold up against the actual repo contents**:
- Code is real and structured (5 stages, A-E, matching the pipeline above), but uses fixed
  internal paths, not CLI args.
- Models are pulled from Hugging Face **on first use** — base KeyBERT / sentence-transformers
  CrossEncoder / `Salesforce/codet5-small` — i.e. the *un-fine-tuned, un-pretrained* starting
  points, not IQLoc's actual trained artifacts. No checkpoints are distributed.
- The curated dataset (`Bench4BLExtended.json`, ~7,500 reports) is also not included — the repo
  expects it to already exist locally.
- No license declared.

**Practical implication**: a real replication means re-doing the expensive parts ourselves —
fine-tuning the CodeBERT cross-encoder (needs the AST-diff-derived positive/negative pairs
built from scratch) and domain-pretraining CodeT5 (needs the 70k-bug-report corpus, or an
honest substitute) — not just wiring up released weights. This is a materially bigger lift
than this project's own relevance-feedback prototype (1 zero-shot LLM call/bug, no training).

## Open scoping questions — need a decision before writing pipeline code

1. **Depth of replication**: full from-scratch retrain of both CodeBERT cross-encoder and
   CodeT5 pretraining (true replication, weeks of work, needs a labeled training set built
   from our own Bench4BL mirror's diffs), vs. an approximation that swaps in something
   equivalent to this project's existing local-LLM relevance judge for the cross-encoder step
   and an off-the-shelf embedding model for keyword extraction (much faster, but then it's not
   really replicating IQLoc's specific mechanism, just its stage structure).
2. **Dataset**: build our own version of `Bench4BLExtended.json` (7,483 reports, 42 systems)
   from this project's 46-project mirror, or request/locate the authors' actual curated file.
   Our own loader already found 16/46 systems contribute 0 usable instances after version
   resolution (`docs/bench4bl_result.md`) — worth checking whether that gap overlaps with
   IQLoc's 42-system list before assuming direct compatibility.
3. **Ground truth granularity**: IQLoc is method-level; this project's pipeline is currently
   file-level throughout (chunk-level judgments in the relevance-feedback prototype are the
   closest analog but aren't wired to method-level ground truth). Matters for any apples-to-
   apples metric comparison.
4. **What "replication" is meant to produce**: a comparison point for `docs/sota_comparison.md`
   (already cites IQLoc's published numbers, 0.553 MRR time-wise), or an actual working
   from-scratch reimplementation of the pipeline, or both.

Not yet decided — surfacing before starting implementation rather than guessing.

## Roadmap: closer to IQLoc, adopting the co-intern's rigor

Scoped 2026-08-19 after comparing this project's two tracks against both IQLoc's actual
pipeline and the co-intern's frozen-baseline deck (`bug_localization_progress_2026-08-17.pptx`
on `github.com/adisenaa/Bug-Report-Localization`, branch `iqloc-publication-replication`).
Neither track matched IQLoc cleanly: the co-intern is closer on the retrieval stage (plain
BM25, no dense/hybrid — matches IQLoc's Stage 1 exactly) and far ahead on rigor (full
population P=4,621, real Slurm accounting/telemetry, a verified local↔MN5 exact-match smoke
test); this project's own hybrid-RRF track is closer on everything downstream of retrieval
(the only one of the two with a working relevance-feedback/reformulation stage at all, and as
of today the `method/keyword_extraction.py` EmbedRank/MMR + cosine-similarity mechanism
genuinely matches IQLoc's Stage 3-4 design, not just BRaIn's simpler raw-token version). This
roadmap closes both gaps rather than picking one track to abandon.

### Phase 1 — Freeze a genuinely IQLoc-comparable retrieval baseline (this branch)

IQLoc's Stage 1 is plain BM25 top-100, no hybrid. This project's hybrid-RRF result (MRR 0.475
on the diverse n=30 manifest, see `results/hybrid_rrf_weighting_openai_skeleton_bench4bl_30_diverse.json`)
is a real, stronger number, but it is not what IQLoc's own architecture uses at this stage, so
it is not a like-for-like comparison point. Freeze **BM25 (skeleton representation, this
project's confirmed-best BM25 config) as the IQLoc-comparable retrieval baseline on this
branch specifically** — do not swap in hybrid retrieval here even though it scores higher,
the same "don't retune the frozen candidate generator" discipline the co-intern's slide 3
states explicitly as a research contract. Hybrid-RRF stays the project's own best-performing
config elsewhere (`docs/sota_comparison.md`, `docs/qwen3_rrf_result.md`), just not conflated
with the IQLoc-comparison track.

### Phase 2 — Full-population evaluation, not n=30 (adopt the co-intern's practice)

**Done.** Every number this project had produced on the IQLoc-approximation pipeline before
this was n=30 (or smaller, for smoke tests) — not remotely comparable in statistical weight to
IQLoc's own 1,497-1,501-instance test split, let alone their full 7,483. The full-population
BM25 run (4,418 instances, our own mirror, `run_bm25_comparison_shard.py` + a 50-way MN5 Slurm
array job 44806072) is now confirmed: skeleton wins again at full scale (MRR 0.2488, down from
0.310 at n=30 — same ranking, harder true distribution). Committed to `main`
(`results/bm25_comparison_bench4bl_full4418_summary.json`), matches the co-intern's own
full-population rigor. Re-running relevance feedback at this same scale is still open — see
Phase 4.

### Phase 3 — Fix the keyword-extraction contamination bug found in smoke testing

**Done, and turned out to be a bigger bug than first scoped.** Root cause found in
`method/java_parsing.py`'s `chunk_java_content`: the header chunk (package/imports/leading
comment, everything before the first method signature) was built from **raw** `content`
instead of the already noise-stripped `cleaned` text used everywhere else in that function —
so a file's leading Javadoc/license comment (the full Apache License boilerplate, for most
Bench4BL Java files) leaked into the header chunk verbatim. That chunk feeds
`method/embedding_retriever._chunk_file_content`, used by `rank_files_embedding_chunked`
**everywhere in this project**, not just the new IQLoc-approximation pipeline where the
contamination was first noticed. So this had likely been diluting chunk embeddings across
every confirmed hybrid-RRF result to date, to some degree — not re-running every prior sweep
to quantify how much, but worth remembering as a caveat on pre-fix numbers. Fixed with a
one-line change (`cleaned[:first_start]` instead of `content[:first_start]`), a regression
test added (`tests/test_java_parsing.py::test_chunk_java_content_header_excludes_comment_text`),
all 163 tests pass. Committed to `main` (`ca46680`) and merged into this branch.

**Re-smoke-tested and confirmed clean.** Same 2 real instances as the original smoke test:
code-side keywords are now real package/class/identifier terms (`org apache`, `smppclient`,
`classextension`, `springframework batch`, `listener factory`, `bean extends`) with zero
license-boilerplate tokens, vs. the pre-fix run's `warranties`, `licensed`, `permissions`,
`obtain`. Reformulation-term quality is real code vocabulary now, not license text. Safe to
move to Phase 4 (scale beyond n=2).

### Phase 4 — Full-population run of the IQLoc-approximation pipeline (real cost, scope first)

**Done at n=200, on MN5 GPU.** Since the default keyword-extraction model (`microsoft/
unixcoder-base`, local) makes the whole pipeline offline-capable end to end -- BM25, Ollama
relevance judge, keyword extraction -- it runs entirely on MN5, no two-phase split needed
(unlike the OpenAI-embedding hybrid-RRF pipeline). New infra built: `run_iqloc_approximation_
shard.py` + `aggregate_iqloc_shards.py` + `scripts/mn5/bench4bl_iqloc_approximation_array.
sbatch`, a 20-way Slurm array job where **each task starts its own local Ollama server**
(confirmed necessary -- no shared-server mode across nodes). GPU-verified first with a real
smoke test (5.4s per chat completion on an H100, vs. up to 460s/instance observed on this
CPU-only Mac) and a single-instance shard validation (25.6s including model load) before
committing to the full run.

**Real n=200 result** (`results/iqloc_approximation_bench4bl_200.json`, 24-repo diverse
manifest, job 44814164, completed in ~6 minutes wall-clock across 20 parallel GPU shards):

| config | Hit@1 | MRR | MAP |
|---|---:|---:|---:|
| retriever (skeleton-BM25 alone) | 0.195 | **0.3083** | 0.2191 |
| + relevance filtering | 0.110 | 0.2358 (&minus;23.5%) | 0.1730 |
| + EmbedRank/MMR reformulation | 0.150 | 0.2542 (&minus;17.5%) | 0.1754 |

**A fourth confirmed replication of the negative finding, now at real scale (n=200, not n=30)
and using IQLoc's own actual reformulation mechanism** (EmbedRank/MMR + cosine-similarity
keyword matching, not this project's simpler raw-identifier-token version) -- not just an
n=30 quirk, and not specific to the less faithful approximation. 3/200 LLM calls (1.5%) failed
to parse (JSON truncated mid-string, likely hitting the 4096-token completion limit on a
large chunk-heavy prompt) -- a real, small, silently-handled failure mode worth fixing before
this number is fully trusted, but not large enough to explain the negative result on its own.

### Phase 5 — Adopt MN5 reproducibility discipline project-wide, not just for this branch

**Done, with a real caveat surfaced.** Three pieces:

**1. Real Slurm accounting/wall-clock capture**, retroactively pulled via `sacct` for both
MN5 jobs run this session:

| Job | Tasks | Wall-clock (first start &rarr; last end) | Mean per-task elapsed |
|---|---|---|---|
| BM25 full-population (44806072) | 50 | 28m46s | 9m00s |
| IQLoc-approximation n=200 (44814164) | 20 | 5m56s | 3m33s |

**A real bug found in the process**: `sacct` showed **9/20 (45%) of the IQLoc job's tasks
marked FAILED**, despite every single one writing a valid shard file (confirmed: the 20/20
aggregation used all of them with real data). Root cause: the sbatch script's last command
was `kill "$OLLAMA_PID"`, and `kill` on an already-exited Ollama server returns nonzero --
that became the whole job's reported exit status instead of the actual Python pipeline's.
Fixed by capturing the pipeline's real exit code before the kill and exiting with that
explicitly (`scripts/mn5/bench4bl_iqloc_approximation_array.sbatch`). This is exactly the
kind of telemetry-accuracy issue the co-intern's rigor (their smoke-test slide checks
`ExitCode 0:0` explicitly) is meant to catch -- a false FAILED status could easily cause a
future run to be needlessly re-submitted, or worse, cast unwarranted doubt on valid results.

**2. Local↔MN5 exact-match verification** (`scripts/check_local_mn5_determinism.py`, on
`main`). Deliberately **not** run against this project's actual confirmed-best config
(hybrid-RRF, OpenAI `text-embedding-3-small` + skeleton-BM25) -- MN5 has no outbound
internet, so an OpenAI-API-dependent pipeline cannot execute there at all, making a same-
config comparison structurally impossible, not just impractical. Used Qwen3-Embedding-0.6B
instead (fully local/offline, runs identically on both machines) to validate the actual
shared code path -- repo content, BM25 skeleton generation, chunking, RRF fusion -- rather
than the specific embedding provider. **Result: exact sha256 match on both test instances
(AMQP-242, AMQP-243) between a local run and an MN5 GPU run.** The real scientific claim
this validates is "the code computes the same thing regardless of machine," which holds;
"the OpenAI-based confirmed-best config reproduces on MN5" is a different, currently
unanswerable claim given the internet constraint, and is not conflated with this result.

**3. Frozen-configuration contract** -- see the top of this document's Phase 1 (skeleton-BM25,
not hybrid, is the frozen retrieval baseline on this branch) plus `docs/qwen3_rrf_result.md`
for the main track's own "why hybrid, why this weight" contract. Not writing a new document
here since equivalent contracts already exist per-track; the addition this phase makes is
tying them to actual verified reproducibility evidence (telemetry + exact-match) rather than
just a stated intention not to retune.

### Live thread, not part of the original 5 phases: recall-over-precision (supervisor guidance, 2026-08-19)

Relayed mid-session: prioritize **recall over precision** in query reformulation and the
pre-LLM retrieval stage (precision may drop, that's an accepted tradeoff), and add a
**semantic layer to query reformulation**. Real diagnostic surfaced while scoping this: the
main-track relevance-feedback pipeline's `--candidate-pool-size 50` default caps Recall@100
at 0.576 vs. 0.751 at pool=200 on the same diverse manifest -- a true file beyond the judged
pool can never be recovered downstream no matter how good the judgment or reformulation is.
Being worked on branch `experiment/recall-oriented-reformulation`: raised the default pool to
100, and wired `method/keyword_extraction.py`'s EmbedRank/MMR + cosine-similarity mechanism
(already built for the IQLoc branch) into the main track's reformulation as a new
`--reformulation-mode semantic` option, deliberately recall-leaning (unions bug-report and
code-side keywords rather than only the narrow cosine-matched overlap). Not yet run at scale
-- see that branch for status.

### What this roadmap deliberately does NOT include yet

Real fine-tuning of a CodeBERT cross-encoder or domain-pretraining CodeT5 (the "full retrain"
option declined earlier this session in favor of "approximate with what we have") stays out of
scope unless revisited explicitly -- the phases above are about making the *approximation*
rigorous and comparable at real scale, not closing the remaining architectural gap to IQLoc's
actual trained components.

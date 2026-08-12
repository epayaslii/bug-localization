# Next steps

Living roadmap doc. Split into (A) direction changes from supervisor guidance (2026-08-12)
and (B) concrete in-flight technical work. Read (A) first — it reframes what "done" means
for several items in (B) and in `docs/PROGRESS_REPORT.md`.

## A. Supervisor guidance (2026-08-12) — direction changes

### The eventual production constraint: no cloud code transfer

The project this work eventually feeds into **cannot send source code to cloud systems**.
Everything built so far assumes cloud LLM calls are available (OpenRouter/OpenAI APIs) —
that assumption may not hold in the target deployment. Not an immediate blocker for current
experiments (still useful for benchmarking/research), but worth designing around going
forward: prefer approaches that can plausibly run with a local/on-prem model, don't build
deeper dependencies on cloud-only APIs without a fallback story.

### Our focus is localization, not repair

SWE-bench's headline metric is end-to-end *fix* success (does the generated patch resolve
the issue), which folds in code-generation quality on top of localization quality. **This
project's actual target is correct localization**, not APR (automated program repair). Keep
evaluating and reporting localization-specific metrics (Hit@k, MRR, MAP, retrieval Recall@k)
as the primary signal — end-to-end fix-success numbers (if ever computed) are secondary
context, not the headline.

### Target architecture shift: IR relevance-feedback loop

Two papers the supervisor flagged as close to the right shape for where this project should
head:

- **BRaIn (2025)** — *Improved IR-based Bug Localization with Intelligent Relevance
  Feedback*
- **IQLoc (2026)** — *Semantics-Driven IR-Based Bug Localization*, ACM
  (`10.1145/3786583.3786882`) — explicitly praised as folding in older IR techniques
  alongside newer ones, a "best of both eras" approach

The flow they both point toward:

```
Bug report -> IR retrieval -> LLM relevance feedback -> query reformulation -> reranking
```

This is a real architecture change from the current pipeline (`BM25/embedding retrieval ->
RRF fusion -> single-shot LLM rerank`, no feedback loop, no query reformulation). The
concrete next-lever framing: **after the IR step, use the LLM's relevance judgment to expand
the query/candidate set and iterate, rather than doing IR once and reranking once.** Two
sub-goals called out explicitly:
1. Expand what we have post-retrieval using different methods (not just pass the same
   top-k straight to a single rerank call).
2. Maximize the LLM relevance-feedback step itself — this is the step our current
   architecture doesn't have at all yet.

**Relation to work already done**: `research/reasoning-rerank` (RGFL-style: LLM writes
reasoning before ranking, single call) was a negative result — see the "Negative Results"
sheet in the findings workbook and `docs/reasoning_rerank_result.md`. That is NOT the same
architecture as BRaIn/IQLoc's flow above — RGFL was a *single-call* reasoning+ranking
combination; relevance feedback -> query reformulation -> reranking is a *multi-step loop*
with the LLM's judgment actively changing what gets retrieved next, not just how one fixed
candidate set gets ordered. The reasoning-rerank negative result doesn't invalidate this
direction; if anything it's evidence *for* separating the reasoning step from the ranking
step more thoroughly, which a feedback loop does by construction.

**Not yet scoped**: how big a build this is, which paper's exact loop to prototype first,
whether query reformulation targets BM25 only or also the embedding retriever. Needs its own
planning pass before implementation starts.

### New benchmarks to add

For continuity across past and future work, the supervisor wants these benchmarks covered
going forward, in addition to what's already used:

| Benchmark | Status here |
|---|---|
| SWE-bench Verified | Already primary benchmark, extensive results |
| Bench4BL | Scoped (2026-08-12), see below — not started |
| LocBench / MuLocBench | Not started |
| SWE-Explore | Not started |

None of the remaining three have been investigated yet for what they actually contain
(format, size, ground-truth granularity, whether existing `dataset/` loaders can be adapted
or need new ones). First step for each is a scoping pass, same as was done for BeetleBox/BugsInPy

#### Bench4BL scoping findings (2026-08-12)

Source: `github.com/exatoa/Bench4BL` (ISSTA 2018 reproducibility study). 10,017 bug reports,
51 Java projects (Apache/Commons/JBoss/Wildfly/Spring + 5 legacy Eclipse projects), JIRA-linked
(not GitHub issues).

**Target schema maps cleanly onto our existing `BugInstance`**: each bug becomes an XML
record with a summary/description (-> `bug_report`), a `<fixedFiles>` list (->
`ground_truths`), and version/fixedVersion fields (-> commit reference). Same shape as the
SWE-bench/BeetleBox loaders already in `dataset/`.

**The real cost is data preparation, not the schema.** There's no clean HuggingFace dataset
like SWE-bench/BeetleBox have. Getting to that XML requires: (1) downloading per-project tar
archives from SourceForge (51 projects, total size unconfirmed — likely multi-GB, several
are large long-lived projects like Hive/HBase; checking real sizes is the first concrete
step before committing further); (2) running the repo's own `GitInflator` script to check
out many git versions per project locally; (3) running `BugRepositoryMaker` (Python 2.7,
old deps: numpy/scipy/GitPython) to reformat already-scraped bug data into the XML. The good
news: the JIRA scraping itself was already done once by the original researchers and is
baked into the downloaded archives — no need to re-scrape JIRA live.

**Recommended approach**: run their legacy Python 2 pipeline once, isolated (throwaway
conda env or Docker), purely to materialize the XML corpus, then write a pure-Python-3
`dataset/bench4bl.py` that parses that XML output directly — same pattern as the existing
loaders, no legacy-toolchain dependency at runtime. Not started — next concrete step is
checking real archive sizes.
originally.

### Papers to read for method ideas

- **RGFL** (arxiv 2601.18044) — already implemented and evaluated
  (`research/reasoning-rerank`, negative result, see above)
- **IQLoc** (ACM `10.1145/3786583.3786882`) — not yet read in detail; supervisor's top pick
  for architecture direction
- **BRaIn** (2025) — not yet read in detail; supervisor's other top pick, same reasoning

## B. Concrete technical work — current state

### MN5 / HPC

- **CPU-only torch is still the open blocker** — `--gres=gpu:1` allocations are silently
  unused (`torch==2.6.0+cpu` installed). Fixing this is the single highest-leverage MN5 item:
  it's the entire reason to use the cluster over a local machine for the retrieval pipeline.
- **n=500 SWE-bench Verified run** — in progress as of 2026-08-12. Full manifest generated
  (`swebench-multi-n500-s42-b7a0108947df`, all 12 repos). Built as a Slurm array job (50
  shards x 10 instances, `scripts/run_hybrid_rrf_weighting_shard.py` +
  `scripts/mn5/qwen3_rrf_array_swebench500.sbatch` + `scripts/aggregate_rrf_shards.py`)
  rather than one ~46hr serial job, modeled on the co-intern's array-job pattern. A 2-task
  test array (job `44532092`) was run first to validate the mechanics before committing the
  full 50-shard array.
- **BeetleBox transferred to MN5** — the same 5 small repos already validated locally
  (dolphinscheduler, dagger, localstack, axios, act; ~574MB) plus an offline dataset export
  (`hf_datasets/beetlebox/`, 27MB, since MN5 has no internet for `load_dataset()`) and the
  existing n=15 manifest. `scripts/compare_bm25_beetlebox_mirrored.py` (new — the original
  n=15 run used an uncommitted ad hoc script; this is the committed, tested replacement,
  verified to reproduce the exact same numbers locally) submitted as job `44534587`.

### Branch hygiene

- **Still the single highest-leverage unstarted item outside of MN5/GPU**: almost nothing is
  merged to `main`. Embedding/hybrid/indexing/SOTA/failure-analysis/MN5-handbook work all
  sits on 12+ unmerged branches, some validated for weeks. This caused a real MN5 blocker
  once already (cluster's checkout from `main` was missing the whole embedding stack).

### Findings workbook

- `bug_localization_findings.xlsx` (Desktop) has the full current-state metrics table,
  dataset stats, MN5 transfer/smoke-test detail, and negative-results summary — the
  reference point for "where do we stand" without re-deriving numbers from scratch.

## Suggested sequencing

1. Let the in-flight MN5 jobs (n=500 array, BeetleBox) finish and land their results.
2. Fix the CPU-only torch blocker — unlocks real GPU speed for everything after this.
3. Merge validated branches to `main` — cheap, unblocks future MN5 transfers from breaking again.
4. Scope IQLoc and BRaIn in detail (read the papers, extract the concrete architecture),
   and scope the 3 new benchmarks (what format, what a loader would need) — both as their
   own planning pass before writing code.
5. Prototype the relevance-feedback loop (BRaIn/IQLoc-style) as a new branch, once scoped.

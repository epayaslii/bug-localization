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

**Relation to work already done — CORRECTED 2026-08-12 after actually reading the RGFL
paper** (`arxiv.org/html/2601.18044v1`, previously only known secondhand through this
project's own implementation). The real paper is a **4-stage pipeline with separate LLM
calls per stage** (file-level reasoning+rerank as one step, element-level reasoning+rerank
as a distinct later step, line-level unchanged from its Agentless baseline, then patch
generation) — not a single combined "reason then rank in one generation" call. And its
real results are strongly **positive**, not negative: SWE-bench Verified file-level Hit@1
71.4%->85%, MRR 81.8%->88.8%, +12.8% end-to-end repair success over the Agentless baseline.

`research/reasoning-rerank` (this project's own attempt, see `docs/reasoning_rerank_result.md`
on that branch) was a **single-call** reasoning+ranking combination — a simplified
approximation of RGFL's idea, not a faithful reproduction of its actual architecture. Its
negative result (36.7%->30.0%->23.3%) and its own diagnosed cause ("final ranking isn't
reliably grounded in the reasoning text in this single-call design... RGFL's actual design
is likely two separate calls") turns out to have correctly predicted exactly this gap,
now confirmed against the real paper. **Reading revised**: this isn't just "RGFL doesn't
help here" — the untried lever (separate reasoning-generation and ranking calls, as the
real paper does) is now confirmed as the likely fix, not a speculative one. Worth
re-attempting with a truer-to-paper two-call design before writing off reasoning-augmented
reranking entirely. `docs/reasoning_rerank_result.md` on `research/reasoning-rerank` still
describes the old (secondhand) understanding and needs this same correction next time that
branch is touched.

Separately, RGFL is NOT the same architecture as BRaIn/IQLoc's relevance-feedback flow
above — RGFL reasons and reranks a *fixed* candidate set (even across its multiple stages);
relevance feedback -> query reformulation -> reranking is a genuine *second retrieval pass*,
the LLM's judgment changes what gets retrieved next, not just how a fixed set gets ordered.
Both directions are worth pursuing; they're not competing explanations of the same gap.

**Scoped (2026-08-12)**, full detail in `docs/relevance_feedback_scoping.md` — both papers
read in full, not just the one-line flow. Key findings: both BRaIn and IQLoc evaluate on
Bench4BL (or a refined version of it), which is almost certainly why the supervisor named it
specifically — a real comparison point, not just another dataset. BRaIn's design (zero-shot
LLM, ~150-250 calls/bug at method-segment granularity) is feasible to adapt with this
project's existing stack; IQLoc needs a supervised fine-tuning pipeline (CodeBERT
cross-encoder + further-pretrained CodeT5) this project doesn't have, so BRaIn's shape is the
one to prototype first. Recommended scaled-down adaptation: file-level (not segment-level)
relevance, batched into one LLM call per bug (matching this project's existing call-volume
pattern) instead of BRaIn's literal per-segment-per-call design, algorithmic (no extra LLM
call) query reformulation reusing existing symbol-extraction code, re-run BM25 with the
reformulated query. Not implemented yet — next step is a small prototype (n=10-15,
SWE-bench Verified, retrieval-only) per the scoping doc's suggested first pass.

### New benchmarks to add

For continuity across past and future work, the supervisor wants these benchmarks covered
going forward, in addition to what's already used:

| Benchmark | Status here |
|---|---|
| SWE-bench Verified | Already primary benchmark, extensive results |
| Bench4BL | Loader built + first real result (2026-08-12), see below |
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

**Update (2026-08-12, later same day): the legacy pipeline turned out to be unnecessary
entirely.** Checked real archive sizes first (~5.6GB total across all 51 projects, via
SourceForge's file listing — far smaller than feared) then inspected one archive directly:
each one already contains the fully processed `bugrepo/repository.xml` and a real git repo
with tags, i.e. the legacy Python 2 pipeline's *output*, not just its inputs — confirmed
against the upstream README's own words ("we already offer the result of this step in
provided subject's archives", saved at `docs/bench4bl_reference/README.md`). Built
`dataset/bench4bl.py` (pure Python 3, XML parsing + `git ls-tree`/`git show`, no legacy
toolchain dependency) and `scripts/mirror_bench4bl.py`, wired into
`generate_evaluation_manifest.py`/`compare_bm25_representations.py`. First real result:
n=30, Hit@1=23.3%, MRR=0.3564, on 5 of 51 projects mirrored so far — see
`docs/bench4bl_result.md`. Next: mirror the remaining 46 projects, add test coverage,
eventually wire into the end-to-end (LLM rerank) path.
originally.

### Papers to read for method ideas

- **RGFL** (arxiv 2601.18044) — already implemented and evaluated
  (`research/reasoning-rerank`, negative result, see above)
- **IQLoc** (ACM `10.1145/3786583.3786882`) — not yet read in detail; supervisor's top pick
  for architecture direction
- **BRaIn** (2025) — not yet read in detail; supervisor's other top pick, same reasoning

## B. Concrete technical work — superseded by docs/PROGRESS_REPORT.md

This section (MN5/HPC status, branch hygiene, sequencing) described 2026-08-12 state and has
been superseded — the branch merge, Bench4BL work, and current gap list now live in
[`docs/PROGRESS_REPORT.md`](PROGRESS_REPORT.md) §16 (kept there rather than duplicated here, to
avoid two copies of the same status drifting apart again). **Section A above is not
superseded** — the supervisor's direction guidance (no-cloud-transfer constraint,
localization-vs-repair framing, target architecture, benchmarks/papers) is unique to this
document and still current as of this writing.

One item from this section specifically **is now done**: torch on MN5 is being upgraded to a
CUDA build (`torch==2.6.0+cu124`, in progress as of 2026-08-17) — see
`docs/mn5_execution_handbook.md` once that lands.

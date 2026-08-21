# Scoping: LLM relevance feedback + query reformulation

Scopes the architecture direction from supervisor guidance (`docs/next_steps.md`): `Bug
report -> IR retrieval -> LLM relevance feedback -> query reformulation -> reranking`.
Read both source papers in full rather than working off the one-line flow description.

**Update — a small prototype has since been run** (`scripts/run_relevance_feedback_test.py`,
n=6 smoke + n=12 real, SWE-bench, `gpt-4o-mini`, scoped to 1 LLM call/bug instead of BRaIn's
150-250 calls/bug): relevance filtering alone gave a real lift (MRR 0.023 → 0.192 over plain
BM25), but the full reformulate-and-rerun-BM25 step **made results worse, not better**
(MRR 0.066 — still beats plain BM25 but clearly loses to relevance-filtering alone). Too
small (n=12) to treat as final, but a real, specific, contrary-to-the-papers signal worth
flagging before investing more here: filtering seems to be doing the useful work, not
reformulation.

**Update 2026-08-18 — confirmed at n=30 on Bench4BL itself, with the design upgraded closer
to BRaIn/IQLoc's own methodology.** The two gaps flagged above are closed: chunk/method-level
relevance judgments (not whole-file — see "Open questions" below, this was the recommended
granularity but the first prototype skipped it), the hybrid BM25+embedding RRF retriever (not
BM25 alone), a local open model (`qwen2.5-coder:7b` via Ollama, not `gpt-4o-mini`) — and run
on Bench4BL, the actual target benchmark, at n=30. Result: **the negative finding holds, more
strongly.**

| Config | MRR | MAP |
|---|---:|---:|
| retriever (hybrid-RRF, no LLM) — best | **0.688** | **0.521** |
| + relevance feedback (filtering only) | 0.384 | 0.318 |
| + relevance feedback + reformulation + rerank | 0.675 | 0.498 |

Relevance filtering alone now *hurts* badly (−44% relative MRR) rather than helping — the
opposite of the SWE-bench n=12 result, where filtering was the one part that worked.
Reformulation partially recovers the loss but still lands under the no-LLM baseline. All
30/30 LLM calls succeeded cleanly, so this isn't a crash/fallback artifact. Full writeup:
`docs/bench4bl_result.md`'s "Relevance-feedback + query reformulation" section. This is now a
real, twice-replicated (different benchmark, different model, different granularity) finding
against this architecture direction as implemented — not proof the concept can't work, but
strong enough that the honest next step is diagnosing *why* (model judgment quality? the
single-batched-call prompt design vs. BRaIn's per-segment calls? the raw-identifier-token
reformulation vs. BRaIn's PageRank term graph?) rather than assuming a bigger n will flip it.

**Update 2026-08-21 — answers the "diagnosing why" question above: it's judgment quality,
not prompt design, and not output-format truncation.** (Context: by this point the project's
confirmed-best pipeline had already moved to `--relevance-mode embedding-cosine`, which
makes zero LLM calls and is confirmed positive at n=30/n=200/n=871 — this update is about the
original zero-shot-LLM path specifically, on branch `experiment/relevance-prompt-optimization`.)

Built a labeled chunk-relevance dev set (n=30 bugs, `scripts/build_relevance_dev_set.py`,
holdout-safe against every manifest that's ever produced a reported number) to test two
concrete hypotheses about *why* the LLM path fails, cheaply — classification accuracy
against derived ground-truth-file-implies-relevant labels, not a full retrieval+rerank cycle:

1. **Prompt wording** (BRaIn/SAMMO-style mutation search, `method/prompt_optimizer.py`) —
   never actually reached. A mandatory smoke test first (score the *current* production
   prompt against 2 dev bugs) surfaced a real, more fundamental bug: the model only judged
   10/211 chunks in one instance, `max_tokens=8192` clearly insufficient for a dense
   verdict-per-chunk response at this candidate-pool size (~211 chunks). Fixing that (raising
   `num_ctx` 16384→32768, confirming `max_tokens` matched production's actual default) helped
   coverage (37/211) but not accuracy.
2. **Output format** (dense verdict-per-chunk vs. sparse relevant-only list,
   `method/models.py`'s `SparseChunkRelevanceResponse` +
   `method/prompt.py`'s `generate_sparse_chunk_relevance_feedback_prompt`) — tested directly
   against the same 5 dev bugs as a clean before/after. Sparse output is ~3x faster (no
   wasted tokens restating "not relevant" for the majority) and has no truncation risk at
   all. **Accuracy was unchanged: F1=0.000, 0/20 true positives, both formats.**

**Reading the actual false positives/negatives by eye (not just aggregate counts) settles
which of the two original hypotheses was right.** False positives were genuine model
errors — e.g. flagging a file's raw `import` block and a generic `afterPropertiesSet()`
boilerplate method as relevant to a job-status-handling bug, with no real connection. False
negatives were mostly a different, known issue: Bench4BL has no chunk/line-level ground
truth (`BugInstance.line_mappings` is never populated for this dataset), so every chunk in a
changed *file* gets labeled "relevant" for scoring purposes, including unrelated setup/
boilerplate chunks the model was right to skip.

**Conclusion**: the negative finding is a genuine mechanism-quality problem with zero-shot
LLM chunk-relevance judgment on this benchmark, not an artifact of prompt wording, context
truncation, or output-format inefficiency — both plausible confounds were tested directly and
ruled out. Beam-search prompt mutation (the originally planned next step) has low expected
value given this and was not run. This raises the case for a fundamentally different
mechanism (a trained classifier, not zero-shot generation) over further LLM-prompt iteration
— see the fine-tuned-classifier scoping work that follows this update.

## What BRaIn and IQLoc actually do

Both target papers use **Bench4BL** (or a refined/expanded version of it) as their benchmark
— this is almost certainly why the supervisor specifically named Bench4BL: it's a real
apples-to-apples comparison point against these exact papers, not just "another dataset."

### BRaIn (2025, arxiv 2501.10542)

1. BM25 retrieves top-K (K=50) documents via Elasticsearch.
2. Each retrieved document is **segmented into methods/functions** (JavaParser).
3. **One LLM call per code segment** (not per file): "does this segment relate to the bug?"
   -> binary JSON `{"relevance": "yes"/"no"}`. Zero-shot, no fine-tuning. Tested with LLaMA-3
   8B / Mistral 7B / Qwen 1.5 7B.
4. **Query reformulation is fully algorithmic, no extra LLM calls**: extract
   class/method/field signatures from LLM-marked-relevant segments, split camelCase, filter
   stopwords, build a term co-occurrence graph, run PageRank, take the top-10 terms, append
   to the original bug report as an expanded query.
5. Re-run BM25 with the expanded query.
6. **Final score is a formula, not another LLM call**: `softmax(new BM25 score) x binary
   LLM relevance (0 or 1)`.

**Result on Bench4BL** (4,683 bugs/42 systems, refined split): MAP 0.537, MRR 0.571,
Hit@10 0.781 — +11% MAP over plain Elasticsearch/BM25.

**Cost**: ~150-250 LLM calls per bug report (one per code segment across the top-K
documents). This is the single biggest adaptation problem — see below.

### IQLoc (2026, arxiv 2510.04468)

1. BM25 retrieves top-K (K=100).
2. **Not a general LLM at all** — a *fine-tuned* CodeBERT cross-encoder scores each
   retrieved method for relevance (0-1), and a further-pretrained CodeT5 model produces
   embeddings used for keyword extraction (EmbedRank algorithm).
3. Query reformulation: extract keywords from the bug report and from LLM/cross-encoder-
   flagged-relevant code via embedding similarity, no extra inference calls beyond the
   embeddings already computed.
4. Re-run BM25 with the reformulated query for the final rerank.

**Result on their own refined/expanded Bench4BL** (7,483 bugs): MAP 0.520, MRR 0.553,
Hit@10 0.735.

**Cost/build**: requires training a cross-encoder and further-pretraining CodeT5 on ~70K bug
reports (their own preprocessing) — a real ML training pipeline (GPU training runs, dataset
curation, checkpointing), not just prompting an off-the-shelf model.

## Feasibility for this project's actual stack

**BRaIn's design is the practical one to adapt.** It only needs zero-shot prompting of an
off-the-shelf LLM — exactly what `method/openrouter_localizer.py` already does. IQLoc needs
a supervised training pipeline (labeled data, fine-tuning infra, GPU training) that doesn't
exist in this project at all — building that would be its own multi-week project, not a
prototype. **Recommendation: adapt BRaIn's shape, borrow IQLoc's idea only where cheap**
(e.g. reusing embeddings we already compute for keyword extraction, since
`method/embedding_retriever.py` already exists).

**The real blocker is BRaIn's call volume, not its concept.** ~150-250 LLM calls/bug at
segment granularity is 150-250x this project's current ~1-call-per-bug design. At n=30 that
would be 4,500-7,500 calls instead of 30 — a real cost/latency problem, and also a much
bigger structural change (needs code segmentation via a Java parser for Bench4BL, which we
don't have — javalang/JavaParser equivalent not yet in this project's dependencies).

## Recommended adaptation (scoped down for a first prototype)

1. **File-level relevance, not segment-level.** Skip method/function segmentation entirely
   for a first pass — ask relevance per *candidate file* from BM25's top-K, not per method.
   Cuts call volume by roughly the average methods-per-file factor (BRaIn's own numbers
   imply ~3-5x), and avoids needing a Java code parser we don't have yet.
2. **Batch the relevance judgments into one LLM call, not one-call-per-candidate.** This
   project's existing structured-output pattern (`ReasoningLocalizerResponse` style, one
   call returns a list of {file, judgment} pairs) already does this shape of thing — ask for
   binary relevance across all top-K candidates in a single call. This is the single biggest
   cost lever: turns "K calls per bug" into "1 call per bug," same call count as today's
   pipeline, just a different prompt/output shape.
3. **Keep query reformulation fully algorithmic (no LLM call)**, matching both papers —
   extract identifier-like tokens from LLM-marked-relevant files (reuse the existing
   `_tokenize_path`/symbol-extraction machinery already in `method/bm25_retriever.py`
   /`method/repository_index.py` rather than building a new PageRank term graph from
   scratch as a first pass; a real term-graph/PageRank step is a reasonable v2 if the simple
   version shows promise).
4. **Re-run BM25 (or the embedding retriever) with the reformulated query**, same as both
   papers' final rerank step.
5. **Final score**: start with BRaIn's simple formula (softmax(new score) x binary
   relevance) rather than a second LLM call — cheapest option, and directly comparable to
   BRaIn's own published numbers.

This keeps the total LLM call budget at **2 calls per bug** (1 relevance-feedback call + 1
final localization call, if a distinct final pick call is kept at all) or as few as **1
call** if the final ranking is derived purely from the formula in step 5 — much closer to
this project's existing cost profile than BRaIn's literal design, while preserving the real
structural difference from the already-rejected reasoning-rerank result: **a genuine second
retrieval pass with a reformulated query**, not just reordering the same fixed candidate set
in one call.

## Why this differs from the reasoning-rerank negative result

`research/reasoning-rerank` (this project's own single-call approximation of RGFL, see
`docs/next_steps.md` for the 2026-08-12 correction after actually reading the real RGFL
paper — it's a 4-stage multi-call pipeline with strongly *positive* published results, not
what our simplified version tested) had the LLM write reasoning then immediately produce a
final ranking, all in one continuous generation, over a *fixed* candidate set. Its diagnosed
failure: the final ranking wasn't reliably grounded in the reasoning text. This
relevance-feedback design is structurally different again from both: relevance feedback and
reformulation happen *before* a second retrieval pass, so the candidate set the final step
sees is actually different from the first pass, not just re-ordered — the "grounding" is
enforced by construction (BM25 mechanically re-scores based on the new query), not by hoping
a single LLM call's ranking respects its own earlier reasoning.

## Open questions before implementation

- File-level relevance granularity (recommended above) vs. a cheaper proxy for
  "segments" using existing chunking (`_chunk_file_content` in
  `method/embedding_retriever.py` already exists and could stand in for JavaParser
  segmentation without new parsing code) — worth a quick comparison once prototyping starts.
- Which retrieval to reformulate: BM25 only (matches both papers) or also the embedding
  retriever (this project's stronger signal) — probably BM25 first since that's what both
  papers validate against, embedding second as a follow-up.
- Evaluate on Bench4BL once fully mirrored (real comparison point against BRaIn/IQLoc's own
  published numbers) or on SWE-bench Verified first (faster iteration, matches this
  project's existing majority of results) — recommend SWE-bench first for fast iteration
  during prototyping, Bench4BL once the design is validated, since that's the real
  apples-to-apples comparison worth taking seriously.
- Model choice for the relevance-feedback call: reuse whatever's already configured
  (`gpt-4o-mini` via OpenRouter) rather than BRaIn's 7-8B open models, since this project
  already has that integration working and paid for.

## Suggested first prototype (small, cheap, fast to get a signal)

- n=10-15 instances, SWE-bench Verified (fast iteration), `gpt-4o-mini`.
- BM25 top-50 candidate pool (smaller than the usual top-100/200, to keep the relevance-
  feedback call's prompt size reasonable).
- One batched relevance-feedback call -> algorithmic keyword extraction from relevant files
  -> re-run BM25 with reformulated query -> compare Hit@1/MRR against plain BM25 top-50 on
  the same instances (retrieval-only, no cost from a separate final LLM pick yet).
- If retrieval-only numbers look promising, extend to the full formula/final-pick step and a
  larger n.

This suggested-prototype spec is what `scripts/run_relevance_feedback_test.py` actually
implemented (see the update note at the top) — n=12 rather than n=10-15, otherwise matching:
SWE-bench, `gpt-4o-mini`, one batched relevance-feedback call per bug, retrieval-only
comparison. The real result reverses this doc's own expectation that reformulation would
help on top of filtering — see the update note for the actual numbers.

# Scoping: LLM relevance feedback + query reformulation

Scopes the architecture direction from supervisor guidance (`docs/next_steps.md`): `Bug
report -> IR retrieval -> LLM relevance feedback -> query reformulation -> reranking`.
Read both source papers in full rather than working off the one-line flow description.
**Not implemented yet** — this is a design document, next step is prototyping.

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

`research/reasoning-rerank` (already tried, rejected — see `docs/reasoning_rerank_result.md`)
had the LLM write reasoning then immediately produce a final ranking, all in one
continuous generation, over a *fixed* candidate set. Its diagnosed failure: the final
ranking wasn't reliably grounded in the reasoning text. This design is structurally
different: relevance feedback and reformulation happen *before* a second retrieval pass, so
the candidate set the final step sees is actually different from the first pass, not just
re-ordered — the "grounding" is enforced by construction (BM25 mechanically re-scores based
on the new query), not by hoping the LLM's ranking call respects its own earlier reasoning.

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

Not started — this document is the scope, not the implementation.

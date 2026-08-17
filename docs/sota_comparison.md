# SOTA baseline comparison

Compares this project's own real results against the state-of-the-art numbers already
extracted in [`literature_review.md`](literature_review.md)'s "Quick reference" table. All
numbers below are either committed result artifacts under `results/` or direct quotes from
`literature_review.md` — nothing here is a new run.

## The comparison that's actually fair

SOTA papers report "Top-1" **after their own reranking/agent step** — CoRNStack's 68.2% is
"with retriever+reranker"; BugCerberus's 65.1% is its fine-tuned classifier's final output.
That is the same *kind* of number as this project's **end-to-end accuracy** (BM25 retrieval
→ LLM rerank), not its raw retrieval-only Hit@1. Comparing SOTA's post-rerank Top-1 against
this project's pre-rerank BM25 Hit@1 would be comparing different pipeline stages, not
different systems doing the same job — so both are reported below, but only the end-to-end
row is the fair SOTA comparison point.

| System | File Top-1 / Acc@1 | Benchmark | Stage |
|---|---:|---|---|
| MarsCode Agent | 88.3% | SWE-bench Lite | post-agent |
| Meta-RAG | 84.7%* | SWE-bench Lite | post-summarization (*own scoring, not strict Acc@1) |
| LocAgent (Claude-3.5) | 77.7% | SWE-bench Lite | post-agent |
| CoRNStack (retriever+reranker) | 68.2% | SWE-Bench Lite | post-rerank |
| Agentless-GPT-4o | 65.7–67.2% | SWE-bench Lite | post-agent |
| BugCerberus (hierarchical fine-tuned) | 65.1% | SWE-bench Lite | post-classifier |
| **This project — end-to-end (hybrid-RRF + gpt-4o-mini), n=30, Bench4BL** | **76.7%** | **Bench4BL** | **post-rerank** |
| This project — end-to-end (skeleton-BM25 + gpt-4o-mini), n=30, Bench4BL | 70.0% | Bench4BL | post-rerank |
| **This project — end-to-end (symbols-BM25 + gpt-4o-mini), n=60, SWE-bench** | **51.7%** | **SWE-bench Verified** | **post-rerank** |
| This project — end-to-end (symbols-BM25 + gpt-4o-mini), n=30 | 50.0% | SWE-bench Verified | post-rerank |
| SWE-Fixer | 30.2% (Pass@1, harder metric) | SWE-bench Verified | — |
| This project — retrieval-only, weighted-RRF hybrid (Qwen3, 1:5), n=30 | 30.0% | SWE-bench Verified | pre-rerank |
| This project — retrieval-only, symbols-BM25 alone, n=30 | 6.7% | SWE-bench Verified | pre-rerank |

## Bench4BL — the benchmark the supervisor actually named, with real SOTA to compare against

Unlike the SWE-bench rows above, Bench4BL has two directly-relevant SOTA papers that **use
this exact benchmark**, not a different split — BRaIn and IQLoc, both scoped in
`docs/relevance_feedback_scoping.md`. This is almost certainly why the supervisor named
Bench4BL specifically: it's a real apples-to-apples comparison point, not just "another
dataset."

| System | MRR | Benchmark | Notes |
|---|---:|---|---|
| **This project — retrieval-only, weighted-RRF hybrid (Qwen3, 1:5), n=30** | **0.714** | Bench4BL | pre-rerank, this project's own manifest (8 projects) |
| BRaIn (zero-shot LLM relevance feedback + query reformulation) | 0.571 | Bench4BL | 4,683 bugs / 42 systems, their own full split |
| IQLoc (fine-tuned CodeBERT cross-encoder + CodeT5) | 0.553 | Bench4BL | 7,483 bugs, their own refined split |

**This project's retrieval-only MRR already beats both papers' full pipelines** — though the
caveat below matters: this is a much smaller manifest (n=30 vs. thousands of bugs) and MRR
alone, not the papers' other metrics. Still a real, positive signal specifically on the
benchmark that motivated the whole relevance-feedback-pipeline direction — worth surfacing
before investing further in building BRaIn/IQLoc's own architecture, since the simpler
retrieval-only pipeline is already competitive with it here.

**This project also has real end-to-end numbers on Bench4BL — 76.7% (hybrid-RRF retrieval) and
70.0% (BM25-only) accuracy, both n=30** — these are in the SWE-bench-style table above, but
still not a fair comparison to BRaIn/IQLoc specifically: they both report Hit@10/MRR for their
*retrieval* step, not a single final-pick accuracy after an LLM narrows to one file. There's no
published Top-1/Acc@1 number from either paper to put next to 76.7% on this specific benchmark
— unlike the SWE-bench agentic/fine-tuned systems, which do report that exact metric.

## Reading

**On Bench4BL, this project's retrieval-only number already beats both directly-relevant SOTA
papers' full pipelines** (0.714 vs. 0.571/0.553 MRR) — the one clearly favorable comparison in
this document, on the one benchmark with a real apples-to-apples reference point.

**On the SWE-bench fair (post-rerank) comparison, this project sits well below every SOTA
baseline listed** — 51.7% vs. a 65–88% range. The gap is large enough that it isn't explained by
noise at n=60; it reflects real architectural differences: every SOTA system above either
fine-tunes a model specifically for localization (BugCerberus's 3 LoRA-tuned Llama-3-8Bs)
or runs a genuine multi-step agent loop (MarsCode Agent, LocAgent), while this project's
end-to-end pipeline is a single BM25-narrow → single LLM-rerank pass with an off-the-shelf
`gpt-4o-mini`.

**The retrieval-only numbers are not directly comparable to any row above** (none of the
SOTA papers report a pre-rerank number in the same units), but they're informative on their
own: BM25 alone only gets the correct file to rank 1 6.7% of the time; the weighted-RRF
hybrid more than triples that to 23.3%. That the *end-to-end* number (51.7%) is so much
higher than the *retrieval* Hit@1 (23.3%) shows the LLM reranker is doing real work beyond
just picking whatever BM25 ranked first — it's not a rubber stamp.

## Caveats — read before citing this comparison anywhere

- **Different benchmark split.** Every SOTA row uses **SWE-bench Lite** (300 instances);
  this project's own rows use **SWE-bench Verified** (500 instances, OpenAI's
  human-filtered-for-solvability set). These are related but not identical — Verified was
  curated for higher solution-quality confidence, not explicitly for difficulty, so the two
  splits aren't guaranteed comparable at face value.
- **n=30/n=60 vs. full-split evaluation.** SOTA papers report numbers over their entire
  benchmark split (300 instances); this project's numbers are seeded samples (n=30, n=60).
  Directional, not final, same caveat as every other n<full-split result in this project.
- **No SOTA system's exact code/candidate-set was reproduced here** — all SOTA numbers are
  taken as reported in their papers, not re-run against this project's own pipeline or data
  splits. A truly controlled comparison would need identical instances and identical
  candidate-file pools, which this comparison does not have.
- **The Bench4BL retrieval-only comparison uses MRR, a different metric than the Top-1/Acc@1
  numbers in the SWE-bench table above** — not directly comparable across the two tables,
  only within each one.
- **This project has real BeetleBox and Bench4BL results now** (see `docs/bench4bl_result.md`)
  — the "SWE-bench-family only" caveat that used to apply here no longer does, though BeetleBox
  itself has no SOTA baseline comparison yet (no papers found reporting on it specifically).

## Reproducing this project's own numbers

```bash
# End-to-end, Bench4BL (paid, gpt-4o-mini)
python main.py --method openrouter --dataset bench4bl --model gpt-4o-mini --manifest results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json --pool-size 500 --bm25-top-k 100 --bm25-skeleton --output results/e2e_gpt4o_mini_bench4bl_30_skeleton.json

# End-to-end, SWE-bench (paid, gpt-4o-mini)
python main.py --method openrouter --dataset swebench --model gpt-4o-mini --sample-size 60 --bm25-top-k 100 --bm25-symbols --output results/end_to_end_swebench_60_symbols.json

# Retrieval-only ceiling (free, offline)
python scripts/compare_bm25_representations.py --manifest results/manifests/swebench-multi-n30-s42-6757c7d8bb76.json --output results/bm25_comparison_swebench_30.json

# Retrieval-only hybrid, SWE-bench (free, local compute -- or MN5, see docs/mn5_execution_handbook.md)
python scripts/run_hybrid_rrf_weighting_test.py --manifest results/manifests/swebench-multi-n30-s42-1fb8f4b8d82f.json --candidate-pool-size 200 --model "Qwen/Qwen3-Embedding-0.6B" --output results/hybrid_rrf_qwen3_swebench_30_mn5.json

# Retrieval-only hybrid, Bench4BL (real Java-aware chunking is slow -- see the array-job pattern in docs/mn5_execution_handbook.md)
python scripts/run_hybrid_rrf_weighting_test.py --dataset bench4bl --manifest results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json --pool-size 250 --candidate-pool-size 200 --model "Qwen/Qwen3-Embedding-0.6B" --output results/hybrid_rrf_qwen3_bench4bl_30_array_mn5.json
```

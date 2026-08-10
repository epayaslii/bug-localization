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
| **This project — end-to-end (symbols-BM25 + gpt-4o-mini), n=60** | **51.7%** | **SWE-bench Verified** | **post-rerank** |
| This project — end-to-end (symbols-BM25 + gpt-4o-mini), n=30 | 50.0% | SWE-bench Verified | post-rerank |
| SWE-Fixer | 30.2% (Pass@1, harder metric) | SWE-bench Verified | — |
| This project — retrieval-only, weighted-RRF hybrid (1:10), n=30 | 23.3% | SWE-bench Verified | pre-rerank |
| This project — retrieval-only, symbols-BM25 alone, n=30 | 6.7% | SWE-bench Verified | pre-rerank |

## Reading

**On the fair (post-rerank) comparison, this project sits well below every SOTA baseline
listed** — 51.7% vs. a 65–88% range. The gap is large enough that it isn't explained by
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
- **This project has never been evaluated against BeetleBox or any non-Python benchmark**
  that appears in this table — everything above is SWE-bench-family only.

## Reproducing this project's own numbers

```bash
# End-to-end (paid, gpt-4o-mini)
python main.py --method openrouter --dataset swebench --model gpt-4o-mini --sample-size 60 --bm25-top-k 100 --bm25-symbols --output results/end_to_end_swebench_60_symbols.json

# Retrieval-only ceiling (free, offline)
python scripts/compare_bm25_representations.py --manifest results/manifests/swebench-multi-n30-s42-6757c7d8bb76.json --output results/bm25_comparison_swebench_30.json

# Retrieval-only hybrid (checkout experiment/hybrid-retrieval first, local compute only)
python scripts/run_hybrid_rrf_weighting_test.py --manifest results/manifests/swebench-multi-n30-s42-1fb8f4b8d82f.json --candidate-pool-size 200 --output results/hybrid_rrf_weighting_swebench_30.json
```

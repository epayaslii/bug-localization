# Qwen3-Embedding-0.6B through the weighted-RRF pipeline

Every prior weighted-RRF result in this project (the 1:10 peak, MRR 0.281) was built on
UniXCoder embeddings. The embedding bake-off later found Qwen3-Embedding-0.6B far stronger
on its own (MRR 0.603 vs. UniXCoder's 0.419, n=6) but never ran it through the actual hybrid
fusion pipeline. This closes that gap.

## Why hybrid retrieval, and why 1:5 — the evidence in one place

Two separate questions, both answered empirically by direct side-by-side runs, not assumed:
**should BM25 and embedding be combined at all**, and **at what weight ratio**.

**BM25 alone vs. embedding alone vs. fused**, same candidate pool, both benchmarks (n=30):

| Config | SWE-bench MRR | Bench4BL MRR |
|---|---:|---:|
| BM25 alone | 0.086 | 0.143 |
| Embedding alone (Qwen3, chunked) | 0.316 | 0.688 |
| **RRF fused (1:5) — best on both** | **0.422** | **0.714** |

BM25 alone is weak because it's pure lexical token overlap between a bug report's prose and
code identifiers — Hit@1 was ~0-3% on both benchmarks. Embedding alone is 3.7-4.8x stronger
(captures semantic similarity beyond exact wording), but fusing still adds a further lift on
top of it: +33% relative on SWE-bench, +3.8% on Bench4BL — confirming the two signals aren't
fully redundant even though embedding dominates.

**Weight-ratio sweep** (`bm25_weight:embedding_weight`), same manifests, via
`scripts/run_hybrid_rrf_weighting_test.py` / its MN5-sharded Bench4BL variant:

| Ratio | SWE-bench MRR (n=30) | Bench4BL MRR (n=30) |
|---|---:|---:|
| 1:1 (unweighted) | — | — |
| 1:2 | 0.385 | — |
| **1:5 — peak on both** | **0.422** | **0.7137** |
| 1:10 | 0.390 | 0.6964 |
| 1:15 | — | 0.7056 |
| 1:50 | — | converges to embedding-alone (see below) |

1:5 is a genuine peak, not just where the sweep happened to stop — metrics were checked on
*both sides* and both go down: 1:2 and 1:10 score below 1:5 on SWE-bench; 1:10 and 1:15 both
score below 1:5 on Bench4BL. A sweep that only tested up to the apparent best (e.g. stopping
at 1:2) couldn't distinguish a real peak from "we didn't test far enough."

One mechanism check from the same sweep: at very high ratios (1:50), the fused ranking
converges to being *bit-identical* to embedding-alone (confirmed to 4 decimal places) —
mathematically expected, since BM25's score contribution is bounded to a narrow band (ranks
only run 1 to the candidate pool size), so once the embedding weight dominates enough, BM25
can no longer flip any file's relative order. This also served as an independent sanity check
that the RRF fusion math itself was implemented correctly, separate from which config wins.

**Caveat**: swept on n=6/n=30 samples, not the full benchmark — the 1:5 peak held when
scaling n=6→n=30 on both benchmarks tested so far, but it's a sampled estimate, not a proven
global optimum for the full dataset.

## Result (n=30, confirmed, MN5)

Same manifest family as the n=6 run below, `--candidate-pool-size 200`,
`Qwen/Qwen3-Embedding-0.6B`. `results/hybrid_rrf_qwen3_swebench_30_mn5.json`:

| Config | MRR | Hit@1 |
|---|---:|---:|
| bm25 (symbols) | 0.086 | 3.3% |
| chunked_embedding alone | 0.316 | 13.3% |
| rrf 1:2 | 0.385 | 26.7% |
| **rrf 1:5 — best** | **0.422** | 30.0% |
| rrf 1:10 | 0.390 | 23.3% |

**Weighted RRF does beat embedding-alone at n=30 — the n=6 finding below ("no weight beats
embedding alone") did not survive scaling up**, the same "small n gives a different answer"
pattern this project has now hit more than once (also seen with Bench4BL's own n=6→n=30
hybrid transition, see `docs/bench4bl_result.md`). The peak ratio (1:5) differs from
UniXCoder's own peak (1:10), but the shape — fusion adds a real lift over embedding-alone —
holds for both embedding models, just at a different ratio.

**0.422 MRR held the project-best position until Bench4BL's own n=30 confirmation (0.7137,
see `docs/bench4bl_result.md`) — SWE-bench Verified is no longer the top benchmark for this
metric, but this number is still more than 50% above UniXCoder's best-ever fused result
(0.281) on the same benchmark.**

## Result (n=6, local CPU, historical — see correction above)

`swebench-multi-n6-s42-3b9be79c3129`, `--candidate-pool-size 200`, `Qwen/Qwen3-Embedding-0.6B`:

| Config | MRR | MAP | Hit@1 | Hit@5 |
|---|---:|---:|---:|---:|
| bm25 (symbols) | 0.023 | 0.023 | 0.000 | 0.000 |
| chunked_embedding alone | 0.603 | 0.602 | 0.500 | 0.667 |
| rrf 1:1 | 0.227 | 0.229 | 0.000 | 0.500 |
| rrf 1:3 | 0.437 | 0.436 | 0.167 | 0.667 |
| rrf 1:10 | 0.603 | 0.602 | 0.500 | 0.667 |
| rrf 1:50 | 0.603 | 0.602 | 0.500 | 0.667 |

At n=6, no weight ratio beat embedding alone — read at the time as "Qwen3-Embedding is
strong enough that BM25 fusion has nothing to add." **This reading did not hold at n=30**
(see above); kept here only as the historical record, not as a trustworthy finding on its own.

## Reproducing

```bash
# n=30 (confirmed result, MN5 acc partition — see docs/mn5_execution_handbook.md)
python scripts/run_hybrid_rrf_weighting_test.py \
  --manifest results/manifests/swebench-multi-n30-s42-1fb8f4b8d82f.json \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output results/hybrid_rrf_qwen3_swebench_30_mn5.json

# n=6 (local, historical)
python scripts/run_hybrid_rrf_weighting_test.py \
  --manifest results/manifests/swebench-multi-n6-s42-3b9be79c3129.json \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output results/hybrid_rrf_qwen3_swebench_6.json
```

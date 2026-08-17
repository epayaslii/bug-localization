# Qwen3-Embedding-0.6B through the weighted-RRF pipeline

Every prior weighted-RRF result in this project (the 1:10 peak, MRR 0.281) was built on
UniXCoder embeddings. The embedding bake-off later found Qwen3-Embedding-0.6B far stronger
on its own (MRR 0.603 vs. UniXCoder's 0.419, n=6) but never ran it through the actual hybrid
fusion pipeline. This closes that gap.

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

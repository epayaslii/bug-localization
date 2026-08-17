# Qwen3-Embedding-0.6B through the weighted-RRF pipeline

Every prior weighted-RRF result in this project (the 1:10 peak, MRR 0.281) was built on
UniXCoder embeddings. The embedding bake-off later found Qwen3-Embedding-0.6B far stronger
on its own (MRR 0.603 vs. UniXCoder's 0.419, n=6) but never ran it through the actual hybrid
fusion pipeline. This closes that gap.

## Result (n=6, local CPU, same manifest as the embedding bake-off)

`swebench-multi-n6-s42-3b9be79c3129`, `--candidate-pool-size 200`, `Qwen/Qwen3-Embedding-0.6B`:

| Config | MRR | MAP | Hit@1 | Hit@5 |
|---|---:|---:|---:|---:|
| bm25 (symbols) | 0.023 | 0.023 | 0.000 | 0.000 |
| **chunked_embedding alone** | **0.603** | **0.602** | 0.500 | 0.667 |
| rrf 1:1 | 0.227 | 0.229 | 0.000 | 0.500 |
| rrf 1:3 | 0.437 | 0.436 | 0.167 | 0.667 |
| rrf 1:10 | 0.603 | 0.602 | 0.500 | 0.667 |
| rrf 1:50 | 0.603 | 0.602 | 0.500 | 0.667 |

**No weight ratio beats embedding alone.** This is a different reading than the UniXCoder
case, where weighted RRF at 1:10 gave a real +20% lift over embedding-alone (0.233 -> 0.281).
Here, Qwen3-Embedding is strong enough on its own that BM25 fusion has nothing useful to
add at any weight -- ratios just converge back toward the embedding-alone number as BM25's
contribution becomes negligible, same shape as the UniXCoder sweep but with the "ceiling"
already at the embedding-alone point instead of past it.

**The headline isn't the fusion question, it's the absolute number**: 0.603 MRR is more
than double UniXCoder's *best fused* result (0.281) found anywhere in this project so far.
n=6, same small-sample caveat as the original bake-off -- directional, not final.

## Next: n=30 confirmation on MN5

Running at n=6 locally took ~46 minutes (Qwen3-Embedding is a decoder-based model, much
slower per-instance than UniXCoder on CPU). A larger n=30 run was moved to MN5's `acc`
(GPU) partition rather than run for hours more locally -- see the MN5 execution handbook
for the transfer/environment details. Result to be added here once that job completes.

## Reproducing (local)

```bash
python scripts/run_hybrid_rrf_weighting_test.py \
  --manifest results/manifests/swebench-multi-n6-s42-3b9be79c3129.json \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output results/hybrid_rrf_qwen3_swebench_6.json
```

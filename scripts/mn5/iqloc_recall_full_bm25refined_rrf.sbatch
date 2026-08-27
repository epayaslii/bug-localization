#!/bin/bash
#SBATCH --job-name=iqloc-full-bm25refined-rrf
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-349
#SBATCH --output=logs/iqloc_full_bm25refined_rrf_%A_%a.out
#SBATCH --error=logs/iqloc_full_bm25refined_rrf_%A_%a.err

# Full-population validation (2026-08-27) of the n=23 breakthrough: refined-BM25 (query-side
# summary-lite + code-aware tokenization, see method/bm25_retriever.py) fused with
# Qwen3-Embedding-0.6B via RRF. At n=23 (candidate_pool_size=100): bm25-alone(refined)=0.5032
# MRR, embedding-alone(within refined pool)=0.6757 MRR, best fused (rrf_1_5_k100)=0.7145 MRR /
# Hit@1 0.565 / MAP 0.4985 -- comfortably clears IQLoc's own published numbers (MRR up to
# 0.553, Hit@1 up to 46.6%) at small sample. Every prior small-sample winner this project has
# tested has regressed at full population (sometimes to a tie, sometimes below), so this run
# is the deciding test: does the refined-BM25 retrieval-side gain (not an LLM-relevance
# artifact, not a reformulation artifact -- pure retrieval fusion, no LLM in this pipeline at
# all) hold up at n=7091, or follow the same regression pattern as everything else.
#
# Retrieval-ONLY, no relevance-filtering, no reformulation, no LLM/Ollama -- run_hybrid_rrf_
# weighting_shard.py just computes bm25[refined]+chunked_embedding once per instance and
# sweeps all 9 weight ratios x k=100 from those same shared base rankings (free RRF math), so
# this one job reports the full weight sweep at full scale, not just the single winning ratio.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_full_bm25refined_rrf/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_hybrid_rrf_weighting_shard.py \
  --manifest results/manifests/iqloc-multi-n7091-s42-f8a5d2c401a4.json \
  --dataset iqloc \
  --iqloc-strict \
  --pool-size 7091 \
  --candidate-pool-size 100 \
  --bm25-repr refined \
  --rrf-k-values 100 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --num-shards 350 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_full_bm25refined_rrf/shards

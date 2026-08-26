#!/bin/bash
#SBATCH --job-name=iqloc-rrf-weighting
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_rrf_weighting_%j.out
#SBATCH --error=logs/iqloc_rrf_weighting_%j.err

# Retrieval-ONLY comparison (2026-08-26): bm25-alone, embedding-alone, and BM25+embedding
# RRF fused at every (weight, k) combination -- 9 weights (1:1 unweighted through 1:50) x
# 4 k's (10/30/60/100) = 36 fused configs -- for ONE --bm25-repr per invocation. NO
# commit-history, NO relevance-filtering, NO query reformulation. Those get layered on top
# of whichever (repr, weight, k) wins THIS comparison, as a separate second-stage sweep,
# not conflated into the same run.
# Computes BM25 + chunked-embedding exactly once per instance and reuses across every
# weight x k config (see run_hybrid_rrf_weighting_test.py) -- the k sweep is pure RRF math
# reusing the same base rankings, so it costs nothing extra beyond the single shared pass.
#
# n=23 not n=30: 7/30 of this manifest's instances don't resolve when re-deriving the
# pool (confirmed identical under both --iqloc-strict and loose -- a real, pre-existing
# gap unrelated to today's changes), so every run against this manifest is actually n=23.

BM25_REPR="${1:-symbols_with_imports}"

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_hybrid_rrf_weighting_test.py \
  --dataset iqloc \
  --manifest results/manifests/iqloc-multi-n30-s42-df69b06700bd.json \
  --pool-size 100 \
  --candidate-pool-size 100 \
  --bm25-repr "$BM25_REPR" \
  --rrf-k-values 10,30,60,100 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output "results/iqloc_rrf_weighting_n23_${BM25_REPR}.json"

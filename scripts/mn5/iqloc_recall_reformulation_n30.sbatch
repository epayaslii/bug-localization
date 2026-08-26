#!/bin/bash
#SBATCH --job-name=iqloc-recall-reform-n30
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_recall_reform_n30_%j.out
#SBATCH --error=logs/iqloc_recall_reform_n30_%j.err

# Same recall-oriented pipeline as bench4bl_recall_reformulation_array_n871.sbatch (hybrid-RRF
# -> embedding-cosine relevance filtering, no LLM call -> semantic EmbedRank/MMR
# reformulation), pointed at IQLoc's own dataset (dataset/iqloc.py's IQLocExtended) instead of
# this project's from-scratch Bench4BL reconstruction. n=30 sanity check, not sharded (small
# enough for a single GPU task).
#
# Retriever, manifest, weights, bm25-repr, candidate-pool-size, and RRF k all parameterized
# ($1-$6) so one script covers the whole 2026-08-26 sweep (bm25-repr skeleton vs.
# symbols_with_imports, candidate-pool-size 100 vs. 200, RRF k away from the never-touched
# default of 60, and weight x k combinations) without editing the file per run. Defaults
# match the confirmed n=30 winner (2026-08-25): hybrid-rrf-history, unweighted 1:1:1,
# symbols_with_imports, pool=100, k=60.

RETRIEVER="${1:-hybrid-rrf-history}"
MANIFEST="${2:-iqloc-multi-n30-s42-df69b06700bd}"
RRF_WEIGHTS="${3:-1,1,1}"
BM25_REPR="${4:-symbols_with_imports}"
CANDIDATE_POOL_SIZE="${5:-100}"
RRF_K="${6:-60}"

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_recall_reform_n30/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_relevance_feedback_shard.py \
  --manifest "results/manifests/${MANIFEST}.json" \
  --dataset iqloc \
  --pool-size 100 \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --retriever "$RETRIEVER" \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights "$RRF_WEIGHTS" \
  --rrf-k "$RRF_K" \
  --bm25-repr "$BM25_REPR" \
  --granularity chunk \
  --reformulation-mode semantic \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode embedding-cosine \
  --relevance-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --keep-fraction 0.5 \
  --num-shards 1 \
  --shard-index 0 \
  --output-dir "results/hpc_mn5_iqloc_recall_reform_n30/shards_${RETRIEVER}_${RRF_WEIGHTS//,/_}_${BM25_REPR}_pool${CANDIDATE_POOL_SIZE}_k${RRF_K}"

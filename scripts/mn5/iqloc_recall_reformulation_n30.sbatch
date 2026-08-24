#!/bin/bash
#SBATCH --job-name=iqloc-recall-reform-n30
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_recall_reform_n30_%j.out
#SBATCH --error=logs/iqloc_recall_reform_n30_%j.err

# Same recall-oriented pipeline as bench4bl_recall_reformulation_array_n871.sbatch (hybrid-RRF
# -> embedding-cosine relevance filtering, no LLM call -> semantic EmbedRank/MMR
# reformulation), pointed at IQLoc's own dataset (dataset/iqloc.py's IQLocExtended) instead of
# this project's from-scratch Bench4BL reconstruction. n=30 sanity check, not sharded (small
# enough for a single GPU task) -- the local CPU run of the same config stalled on instance 1
# for 15+ minutes (Qwen3-Embedding-0.6B doing 2-3 embedding passes/instance with no GPU), so
# this exists to get the same result in minutes instead of hours.
#
# Needs bench4bl_cache/Bench4BLExtended.json and the 7 new-clone + 18 tag-refreshed
# sub-project mirrors synced to MN5 first -- these are newer than what MN5's bench4bl_cache
# had for the original Bench4BL work. See the rsync command given alongside this script.

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
  --manifest results/manifests/iqloc-multi-n30-s42-df69b06700bd.json \
  --dataset iqloc \
  --pool-size 100 \
  --candidate-pool-size 100 \
  --retriever hybrid-rrf \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights 1,5 \
  --bm25-repr symbols_with_imports \
  --granularity chunk \
  --reformulation-mode semantic \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode embedding-cosine \
  --relevance-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --keep-fraction 0.5 \
  --num-shards 1 \
  --shard-index 0 \
  --output-dir results/hpc_mn5_iqloc_recall_reform_n30/shards

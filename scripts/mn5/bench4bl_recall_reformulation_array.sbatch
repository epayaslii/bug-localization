#!/bin/bash
#SBATCH --job-name=bench4bl-recall-reform-n200
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --output=logs/bench4bl_recall_reform_array_%A_%a.out
#SBATCH --error=logs/bench4bl_recall_reform_array_%A_%a.err

# Recall-oriented relevance-feedback pipeline (hybrid-RRF retriever -> embedding-cosine
# relevance filtering, no LLM call -> semantic EmbedRank/MMR reformulation), n=200 diverse
# manifest, sharded 20-way. Uses Qwen3-Embedding-0.6B (fully local/offline, GPU-accelerated)
# for everything -- not this project's confirmed-best local config (OpenAI text-embedding-3-
# small), since MN5 has no outbound internet and an OpenAI-dependent pipeline cannot execute
# there at all. This is validating whether the *mechanism* (embedding-cosine filter + semantic
# reformulation) generalizes to a different embedding model, not reproducing the exact n=30
# breakthrough config.
#
# No Ollama server needed at all -- relevance-mode=embedding-cosine makes zero LLM calls,
# unlike the IQLoc-approximation array job.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_recall_reform_n200/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_relevance_feedback_shard.py \
  --manifest results/manifests/bench4bl-multi-n200-s42.json \
  --dataset bench4bl \
  --pool-size 4418 \
  --candidate-pool-size 100 \
  --retriever hybrid-rrf \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights 1,5 \
  --bm25-repr skeleton \
  --granularity chunk \
  --reformulation-mode semantic \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode embedding-cosine \
  --relevance-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --keep-fraction 0.5 \
  --num-shards 20 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_recall_reform_n200/shards

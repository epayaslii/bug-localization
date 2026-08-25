#!/bin/bash
#SBATCH --job-name=iqloc-cosine-retriever
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_cosine_retriever_%j.out
#SBATCH --error=logs/iqloc_cosine_retriever_%j.err

# Cosine-similarity relevance filtering (no LLM) on IQLoc n=30, isolating the retriever
# variable (plain hybrid-rrf vs. hybrid-rrf-history) while holding everything else -- including
# --reformulation-mode literal -- identical to the qwen2.5-coder LLM comparison job
# (iqloc_llm_relevance_compare.sbatch), so all three configs (LLM, cosine-plain, cosine-history)
# are directly comparable on the exact same n=30 sample.
#
# Uses run_relevance_feedback_test.py directly (not the sharded version) to match the LLM
# job's script family exactly, even though cosine mode doesn't need sharding for n=30.

RETRIEVER="${1:?Usage: sbatch iqloc_cosine_retriever_compare.sbatch <retriever> <output-suffix> [rrf-weights] [manifest-basename]}"
SUFFIX="${2:?Usage: sbatch iqloc_cosine_retriever_compare.sbatch <retriever> <output-suffix> [rrf-weights] [manifest-basename]}"
RRF_WEIGHTS="${3:-1,5}"
MANIFEST="${4:-iqloc-multi-n30-s42-df69b06700bd}"

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_relevance_feedback_test.py \
  --manifest "results/manifests/${MANIFEST}.json" \
  --dataset iqloc \
  --pool-size 100 \
  --retriever "$RETRIEVER" \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights "$RRF_WEIGHTS" \
  --bm25-repr symbols_with_imports \
  --granularity chunk \
  --reformulation-mode literal \
  --relevance-mode embedding-cosine \
  --relevance-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --keep-fraction 0.5 \
  --output "results/iqloc_cosine_${SUFFIX}_n30_literal.json"

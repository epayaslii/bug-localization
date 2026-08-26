#!/bin/bash
#SBATCH --job-name=iqloc-recall-reform-full-cosine-pool200
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-349
#SBATCH --output=logs/iqloc_recall_reform_full_cosine_pool200_%A_%a.out
#SBATCH --error=logs/iqloc_recall_reform_full_cosine_pool200_%A_%a.err

# Full-population isolation test (2026-08-26): SAME strong retrieval config as the LLM
# full-scale run (results/hpc_mn5_iqloc_recall_reform_full_llm_pool200/, MRR 0.518, tied with
# IQLoc's random-split MRR) -- hybrid-rrf-history, symbols_with_imports, RRF 1:15:1, k=30,
# candidate-pool-size 200, semantic reformulation -- but with embedding-cosine relevance
# filtering instead of LLM. The LLM run showed relevance-filtering ALONE badly hurting at full
# scale (MRR 0.519 retriever -> 0.366 filtered -> 0.518 reformulated, net ~0 gain), directly
# contradicting the n=23 finding that LLM beats cosine. This run answers: is that a real
# LLM-specific failure at scale, or does cosine ALSO net out to ~0 gain on this same strong
# retrieval base (i.e. the whole relevance-filtering+reformulation stage just doesn't help at
# scale, regardless of method)? No Ollama needed -- embedding-cosine makes zero LLM calls, so
# this should run faster than the LLM version despite pool=200.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_recall_reform_full_cosine_pool200/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_relevance_feedback_shard.py \
  --manifest results/manifests/iqloc-multi-n7091-s42-f8a5d2c401a4.json \
  --dataset iqloc \
  --iqloc-strict \
  --pool-size 7091 \
  --candidate-pool-size 200 \
  --retriever hybrid-rrf-history \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights 1,15,1 \
  --rrf-k 30 \
  --bm25-repr symbols_with_imports \
  --granularity chunk \
  --reformulation-mode semantic \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode embedding-cosine \
  --relevance-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --keep-fraction 0.5 \
  --num-shards 350 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_recall_reform_full_cosine_pool200/shards

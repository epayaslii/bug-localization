#!/bin/bash
#SBATCH --job-name=bench4bl-candidates-array30
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=20
#SBATCH --array=0-9
#SBATCH --output=logs/bench4bl_candidates_array_%A_%a.out
#SBATCH --error=logs/bench4bl_candidates_array_%A_%a.err

# Phase 1 of a two-phase Bench4BL hybrid-RRF end-to-end eval: computes retrieval candidates
# only (BM25 -> chunked embedding -> weighted RRF 1:5, Qwen3-Embedding-0.6B), no LLM call --
# the LLM call needs live internet, which MN5 doesn't have, so it runs locally afterward
# (main.py --candidates-file) against this job's aggregated output. Same 10-shard split as
# the retrieval-only MRR confirmation (bench4bl_qwen3_rrf_array.sbatch), same reason: real
# per-instance Java-aware chunking is too slow for one serial job (~900s/instance).

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_array_bench4bl_candidates30/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_hybrid_retrieval_candidates_shard.py \
  --manifest results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json \
  --dataset bench4bl \
  --pool-size 250 \
  --candidate-pool-size 200 \
  --retrieval-top-k 100 \
  --rrf-weights 1,5 \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --num-shards 10 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_array_bench4bl_candidates30/shards

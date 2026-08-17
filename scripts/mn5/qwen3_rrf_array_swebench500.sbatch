#!/bin/bash
#SBATCH --job-name=qwen3-rrf-array500
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=20
#SBATCH --array=0-49
#SBATCH --output=logs/rrf_array_%A_%a.out
#SBATCH --error=logs/rrf_array_%A_%a.err

# CPU-account run (no --gres=gpu:1) -- torch on MN5 is currently torch==2.6.0+cpu, so a GPU
# allocation would sit unused anyway (see docs/mn5_execution_handbook.md, Blocker 5). This
# array job is the CPU-scale replacement for one long serial run of
# run_hybrid_rrf_weighting_test.py: 500 SWE-bench Verified instances split into 50 shards of
# 10 instances each (~55min/shard at the observed ~5.5min/instance), run in parallel instead
# of one ~46-hour serial job with no incremental output.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_array_swebench500/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SWEBENCH_LOCAL_PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_datasets/swebench_verified_test
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_hybrid_rrf_weighting_shard.py \
  --manifest results/manifests/swebench-multi-n500-s42-b7a0108947df.json \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --num-shards 50 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_array_swebench500/shards

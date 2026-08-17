#!/bin/bash
#SBATCH --job-name=bench4bl-rrf-array30
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=20
#SBATCH --array=0-9
#SBATCH --output=logs/bench4bl_rrf_array_%A_%a.out
#SBATCH --error=logs/bench4bl_rrf_array_%A_%a.err

# Array-job replacement for bench4bl_qwen3_rrf.sbatch's serial n=30 run, which was killed
# before completion: real per-instance Java-aware chunking (method/java_parsing.py) runs
# ~900s/instance, so a serial n=30 run risks ~7.5hrs -- longer than acc_ehpc's practical
# budget for one job and produces zero output if killed partway (see
# docs/mn5_execution_handbook.md, "only writes output once" pattern). This splits the same
# n=30 manifest into 10 shards of 3 instances each (~45min/shard at the observed rate) run in
# parallel via run_hybrid_rrf_weighting_shard.py, then aggregate_rrf_shards.py merges them.
# CPU-only (torch==2.6.0+cpu on MN5, Blocker 5), no --gres=gpu:1 requested.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_array_bench4bl30/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_hybrid_rrf_weighting_shard.py \
  --manifest results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json \
  --dataset bench4bl \
  --pool-size 250 \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --num-shards 10 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_array_bench4bl30/shards

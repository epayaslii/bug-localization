#!/bin/bash
#SBATCH --job-name=bench4bl-dense-probe
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=logs/bench4bl_dense_probe_%A_%a.out
#SBATCH --error=logs/bench4bl_dense_probe_%A_%a.err

# True dense-retrieval-only probe: chunked Qwen3-Embedding ranking over the FULL repository
# corpus per bug (no BM25 pre-filter at all), 8-way sharded on the n=30 diverse manifest --
# matching the co-intern's own 8-bug bounded-probe scale for their Jina dense-retrieval test.
# GPU-accelerated (H100); full-corpus embedding is real compute per instance (every file in
# the repo, not a ~100-file BM25-narrowed pool), so this stays a small probe, not a full run,
# until the numbers say it's worth scaling.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_dense_probe/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_dense_retrieval_shard.py \
  --manifest results/manifests/bench4bl-multi-n30-s42-8c4f91c33f21.json \
  --dataset bench4bl \
  --pool-size 4418 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --num-shards 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_dense_probe/shards

#!/bin/bash
#SBATCH --job-name=bench4bl-bm25-full
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-49
#SBATCH --output=logs/bench4bl_bm25_full_array_%A_%a.out
#SBATCH --error=logs/bench4bl_bm25_full_array_%A_%a.err

# Full-population (4,418 instances, no diversity sampling -- see
# results/manifests/bench4bl-full-4418-s42.json) BM25 4-representation comparison, sharded
# across 50 array tasks via run_bm25_comparison_shard.py. Retrieval-only, offline, CPU-bound
# lexical work -- no --gres=gpu:1 requested, since BM25/Elasticsearch-style scoring has no
# GPU path and a GPU reservation here would only add queue time for zero speedup. Real
# per-instance cost is dominated by content-fetching (git cat-file) the first time a repo/
# commit is touched, not the BM25 scoring itself.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_bm25_full/shards

export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_bm25_comparison_shard.py \
  --manifest results/manifests/bench4bl-full-4418-s42.json \
  --dataset bench4bl \
  --pool-size 4418 \
  --num-shards 50 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_bm25_full/shards

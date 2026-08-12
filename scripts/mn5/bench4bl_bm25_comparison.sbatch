#!/bin/bash
#SBATCH --job-name=bench4bl-bm25
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/bench4bl_bm25_%j.out
#SBATCH --error=logs/bench4bl_bm25_%j.err

# Retrieval-only, offline, no embedding/LLM/GPU needed -- same profile as the BeetleBox
# comparison job (finished in ~1min there), acc_debug's 2hr cap is a generous margin.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results

export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/compare_bm25_representations.py \
  --dataset bench4bl \
  --manifest results/manifests/bench4bl-multi-n30-s42-9449c3b8a675.json \
  --output results/bm25_comparison_bench4bl_30_mn5.json

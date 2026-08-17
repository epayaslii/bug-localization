#!/bin/bash
#SBATCH --job-name=beetlebox-bm25
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/beetlebox_bm25_%j.out
#SBATCH --error=logs/beetlebox_bm25_%j.err

# Retrieval-only, offline, no embedding/LLM/GPU needed -- BM25 over 15 already-mirrored
# BeetleBox instances runs in seconds locally, so acc_debug's 2hr cap is a generous margin
# for a 30min request.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results

export BEETLEBOX_LOCAL_PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_datasets/beetlebox
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/compare_bm25_beetlebox_mirrored.py \
  --manifest results/manifests/beetlebox-multi-n26-s42-cfb046f258fc.json \
  --output results/bm25_comparison_beetlebox_15_mn5.json

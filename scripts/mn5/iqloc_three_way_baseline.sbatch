#!/bin/bash
#SBATCH --job-name=iqloc-three-way
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_three_way_%j.out
#SBATCH --error=logs/iqloc_three_way_%j.err

# BM25-alone vs. chunked-embedding-alone (Qwen3-Embedding-0.6B) vs. hybrid-RRF, all computed
# in one pass over the same candidate pool -- for a co-intern-style 3-row baseline comparison
# table (Method | MAP | MRR | Hit@1 | Hit@5 | Hit@10), on the corrected-dataset n=30 sample.

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

.venv_mn5/bin/python scripts/run_hybrid_retrieval_test.py \
  --manifest results/manifests/iqloc-multi-n30-s42-ca06c4695c47.json \
  --dataset iqloc \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output results/iqloc_three_way_baseline_n30.json

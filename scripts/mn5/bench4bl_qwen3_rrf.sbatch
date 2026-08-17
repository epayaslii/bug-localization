#!/bin/bash
#SBATCH --job-name=bench4bl-qwen3-rrf
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/bench4bl_qwen3_rrf_%j.out
#SBATCH --error=logs/bench4bl_qwen3_rrf_%j.err

# BM25 + Qwen3-Embedding-0.6B reranking, full RRF weight sweep (bm25-alone,
# embedding-alone, 1:1 through 1:50), on the n=30 Bench4BL manifest -- same
# weight-sweep script and model already validated on SWE-bench
# (results/hybrid_rrf_qwen3_swebench_30_mn5.json, MRR 0.4216 at 1:5). CPU-only
# (torch==2.6.0+cpu on MN5, see docs/mn5_execution_handbook.md Blocker 5), no
# --gres=gpu:1 requested since it would sit unused. acc_ehpc/3hr (not
# acc_debug/2hr) since real per-method Java chunking (method/java_parsing.py)
# now produces far more embedding calls per file than the earlier run, which
# had been silently chunking almost nothing (see docs/bench4bl_result.md,
# "Content-fetching gap").

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_hybrid_rrf_weighting_test.py \
  --dataset bench4bl \
  --manifest results/manifests/bench4bl-multi-n6-s42-mn5-8proj.json \
  --pool-size 500 \
  --candidate-pool-size 200 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --output results/hybrid_rrf_qwen3_bench4bl_6_mn5_java_aware.json

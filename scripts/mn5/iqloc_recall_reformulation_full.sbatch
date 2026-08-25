#!/bin/bash
#SBATCH --job-name=iqloc-recall-reform-full
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-349
#SBATCH --output=logs/iqloc_recall_reform_full_%A_%a.out
#SBATCH --error=logs/iqloc_recall_reform_full_%A_%a.err

# Full-population run: same recall-oriented pipeline confirmed positive on Bench4BL (n=871,
# MRR 0.563->0.580) and now on IQLoc's own dataset at n=23 (MRR 0.5116->0.5364), scaled to the
# strict 7,108-instance "ALL GT localizable" cohort (--iqloc-strict, dataset/iqloc.py's
# include_partial=False -- excludes the 310 instances where only some fixed_files resolved).
#
# Qwen3-Embedding-0.6B chosen over OpenAI for this MN5-executed stage despite OpenAI scoring
# higher in the n=30 embedding bake-off (0.6342 vs 0.5827 MRR standalone) -- MN5 has no
# internet, so OpenAI cannot run here at all; this is a real tradeoff (GPU speed/parallelism
# vs. best-available quality), not an oversight. Jina was also bake-off-tested and ruled out
# (0.1940 MRR, not competitive).
#
# 350 shards (MaxSubmitPU=366 on this QOS, confirmed via sacctmgr -- this is comfortably under
# that ceiling): ~20.3 instances/shard, ~14.5 min/shard at the real per-instance mean (42.87s,
# from the n=23 sanity run's actual timings, which included a 298s outlier) -- --time=00:45:00
# leaves real buffer for shards that draw more than one large-repo outlier.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_recall_reform_full/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_relevance_feedback_shard.py \
  --manifest results/manifests/iqloc-multi-n7108-s42-bcf7195dbff1.json \
  --dataset iqloc \
  --iqloc-strict \
  --pool-size 7108 \
  --candidate-pool-size 100 \
  --retriever hybrid-rrf \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights 1,5 \
  --bm25-repr symbols_with_imports \
  --granularity chunk \
  --reformulation-mode semantic \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode embedding-cosine \
  --relevance-embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --keep-fraction 0.5 \
  --num-shards 350 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_recall_reform_full/shards

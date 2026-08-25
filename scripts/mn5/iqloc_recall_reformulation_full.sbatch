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

# Full-population run on the AUTHOR'S CORRECTED dataset (7,483 raw records, 7,091 strict
# "ALL GT localizable" cohort -- superseded the earlier 7,493/7,108-record version once the
# IQLoc author confirmed it was an earlier/wrong export). Winning config from the 2026-08-25
# n=30 comparison matrix (9 configs tested on this corrected dataset): commit-history RRF
# reranking (--retriever hybrid-rrf-history) + embedding-cosine relevance filter + semantic
# EmbedRank/MMR reformulation -- Hit@1 0.433, MRR 0.5738, MAP 0.4518, beating both the plain
# retriever and every literal-reformulation variant tested, and beating IQLoc's own published
# random-split Hit@1 (0.423) and both splits' MRR (0.520/0.553).
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
mkdir -p logs results/hpc_mn5_iqloc_recall_reform_full_corrected/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_relevance_feedback_shard.py \
  --manifest results/manifests/iqloc-multi-n7091-s42-f8a5d2c401a4.json \
  --dataset iqloc \
  --iqloc-strict \
  --pool-size 7091 \
  --candidate-pool-size 100 \
  --retriever hybrid-rrf-history \
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
  --output-dir results/hpc_mn5_iqloc_recall_reform_full_corrected/shards

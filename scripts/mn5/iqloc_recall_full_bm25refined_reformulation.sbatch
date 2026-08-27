#!/bin/bash
#SBATCH --job-name=iqloc-full-refined-reform
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-349
#SBATCH --output=logs/iqloc_full_refined_reform_%A_%a.out
#SBATCH --error=logs/iqloc_full_refined_reform_%A_%a.err

# Full-population run (2026-08-27) of the reformulation-added refined-BM25 pipeline:
# refined BM25 + Qwen3-Embedding-0.6B + commit-history (hybrid-rrf-history) + qwen3-coder:30b
# LLM relevance filtering + semantic reformulation.
#
# Speed levers applied (per direct request) on top of the truncation fix
# (--relevance-batch-size, added to scripts/run_relevance_feedback_test.py /
# run_relevance_feedback_shard.py this session -- batches chunk-relevance judgment into
# groups instead of one call for the whole candidate pool, which is what actually stopped
# qwen3-coder's reasoning chain-of-thought from truncating the JSON):
#   1. --max-chunks-per-file 3 (was 5)      -- fewer chunks per candidate file
#   2. --candidate-pool-size 100 (was 200)  -- fewer candidate files per instance
#   3. --relevance-batch-size 50 (was 25)   -- fewer LLM calls per instance (untested at this
#      size before this run; batch=25 was confirmed truncation-free at n=23, batch=50 is a
#      deliberate speed/risk tradeoff, not yet independently re-validated at n=23)
# Together these cut total chunks-per-instance roughly 3x and halve the per-chunk-group call
# count on top of that -- observed per-instance time at pool=200/chunks=5/batch=25 was
# 300-390s; this config is expected (not yet confirmed) to land around 100-150s/instance.
# --time bumped to 2h per shard (vs qwen3-coder's earlier 90min) for headroom given batch=50
# is unproven and per-instance time could still run long on outlier instances.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
module load ollama/0.11.8
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_full_refined_reform/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1
export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models

ollama serve > "logs/ollama_serve_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
    break
  fi
  sleep 2
done

.venv_mn5/bin/python scripts/run_relevance_feedback_shard.py \
  --manifest results/manifests/iqloc-multi-n7091-s42-f8a5d2c401a4.json \
  --dataset iqloc \
  --iqloc-strict \
  --pool-size 7091 \
  --candidate-pool-size 100 \
  --retriever hybrid-rrf-history \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights 1,5,1 \
  --rrf-k 100 \
  --bm25-repr refined \
  --granularity chunk \
  --max-chunks-per-file 3 \
  --reformulation-mode semantic \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode llm \
  --method ollama \
  --model qwen3-coder:30b \
  --num-ctx 65536 \
  --max-tokens 32768 \
  --relevance-batch-size 50 \
  --num-shards 350 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_full_refined_reform/shards
PIPELINE_EXIT=$?

kill "$OLLAMA_PID" 2>/dev/null
exit $PIPELINE_EXIT

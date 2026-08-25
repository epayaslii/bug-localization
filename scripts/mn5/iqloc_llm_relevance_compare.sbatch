#!/bin/bash
#SBATCH --job-name=iqloc-llm-relevance
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_llm_relevance_%j.out
#SBATCH --error=logs/iqloc_llm_relevance_%j.err

# Cosine-vs-LLM relevance-filtering comparison on IQLoc's own dataset (n=30), using
# run_relevance_feedback_test.py's --relevance-mode llm path -- unlike the recall-oriented
# array job, this script isn't sharded, so this runs as a single GPU task.
#
# Model name passed as $1 (e.g. "qwen2.5-coder:7b" or "qwen3.8"). Each task starts its OWN
# local `ollama serve` (no shared-server mode across nodes), same lifecycle already proven in
# scripts/mn5/bench4bl_iqloc_approximation_array.sbatch -- poll /api/tags until ready, run the
# pipeline, capture its exit code BEFORE killing the server (a nonzero kill status on an
# already-dead process becoming the job's reported FAILED status was a real bug fixed
# 2026-08-21 on that job).
#
# Bench4BL finding this is checking against IQLoc's own data: LLM-based relevance filtering
# hurt badly in 4 separate replications there; embedding-cosine (already run, n=23: MRR
# 0.5364 reformulated) was the confirmed-positive alternative. Not yet tested whether the
# LLM-negative finding also holds on IQLoc's dataset specifically.

MODEL="${1:?Usage: sbatch iqloc_llm_relevance_compare.sbatch <ollama-model-name> <retriever> <manifest-basename>}"
RETRIEVER="${2:-hybrid-rrf}"
MANIFEST="${3:-iqloc-multi-n30-s42-df69b06700bd}"

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
module load ollama/0.11.8
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1
export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models

ollama serve > "logs/ollama_serve_${SLURM_JOB_ID}_${MODEL//[:\/]/_}.log" 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
    break
  fi
  sleep 2
done

SAFE_MODEL="${MODEL//[:\/]/_}"
.venv_mn5/bin/python scripts/run_relevance_feedback_test.py \
  --manifest "results/manifests/${MANIFEST}.json" \
  --dataset iqloc \
  --pool-size 100 \
  --retriever "$RETRIEVER" \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights 1,5 \
  --bm25-repr symbols_with_imports \
  --granularity chunk \
  --relevance-mode llm \
  --method ollama \
  --model "$MODEL" \
  --num-ctx 16384 \
  --max-tokens 8192 \
  --output "results/iqloc_llm_relevance_${SAFE_MODEL}_${RETRIEVER}_n30.json"
PIPELINE_EXIT=$?

kill "$OLLAMA_PID" 2>/dev/null
exit $PIPELINE_EXIT

#!/bin/bash
#SBATCH --job-name=iqloc-step2-qwen3coder
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_step2_qwen3coder_%j.out
#SBATCH --error=logs/iqloc_step2_qwen3coder_%j.err

# n=23 test of Qwen3-Coder-30B-A3B (real MoE, 30B total / ~3B active params per token --
# NOT the qwen3.8 hybrid-SSM architecture, so this should be well-supported by the standard
# module-loaded Ollama 0.11.8, unlike qwen3.8 which needed a manually-transferred newer
# binary) as the LLM relevance judge, in place of qwen2.5-coder:7b. Same best-known config
# otherwise: symbols_with_imports, RRF k=30, semantic reformulation, candidate-pool-size 200.
# Retriever passed as $1 (hybrid-rrf or hybrid-rrf-history) so one script covers both arms
# the user asked for.

RETRIEVER="${1:?Usage: sbatch iqloc_step2_qwen3coder.sbatch <retriever hybrid-rrf|hybrid-rrf-history> [candidate-pool-size] [reformulation-mode semantic|literal]}"
CANDIDATE_POOL_SIZE="${2:-200}"
REFORMULATION_MODE="${3:-semantic}"
if [ "$RETRIEVER" = "hybrid-rrf-history" ]; then
  RRF_WEIGHTS="1,15,1"
else
  RRF_WEIGHTS="1,15"
fi

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
module load ollama/0.11.8
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1
export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models

ollama serve > "logs/ollama_serve_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
    break
  fi
  sleep 2
done

.venv_mn5/bin/python scripts/run_relevance_feedback_test.py \
  --manifest "results/manifests/iqloc-multi-n30-s42-df69b06700bd.json" \
  --dataset iqloc \
  --pool-size 100 \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --retriever "$RETRIEVER" \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights "$RRF_WEIGHTS" \
  --rrf-k 30 \
  --bm25-repr symbols_with_imports \
  --granularity chunk \
  --reformulation-mode "$REFORMULATION_MODE" \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode llm \
  --method ollama \
  --model qwen3-coder:30b \
  --num-ctx 32768 \
  --max-tokens 16384 \
  --output "results/iqloc_step2_qwen3coder_${RETRIEVER}_pool${CANDIDATE_POOL_SIZE}_${REFORMULATION_MODE}_n23.json"
PIPELINE_EXIT=$?

kill "$OLLAMA_PID" 2>/dev/null
exit $PIPELINE_EXIT

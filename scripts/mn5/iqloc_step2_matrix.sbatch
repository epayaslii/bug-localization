#!/bin/bash
#SBATCH --job-name=iqloc-step2
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_step2_%j.out
#SBATCH --error=logs/iqloc_step2_%j.err

# Step 2 (2026-08-26): on top of Step 1's winning retrieval config (Qwen3-Embedding-0.6B,
# symbols_with_imports, RRF 1:15, k=30 -- results/iqloc_rrf_weighting_n23_symbols_with_imports.json),
# test whether commit-history (RETRIEVER) and/or LLM-vs-cosine relevance filtering
# (RELEVANCE_MODE) help, with semantic reformulation always on. Each invocation's single run
# yields 3 stages internally (retriever / relevance_filtered / reformulated -- see
# run_relevance_feedback_test.py's _run_one), so 4 jobs (2 retrievers x 2 relevance modes)
# cover the full step-2 matrix.
#
# hybrid-rrf-history's 3rd (history) weight is added at parity (1) on top of the already-
# validated 1:15 bm25:embedding ratio, not independently re-swept -- step 2 asks "does adding
# this signal help at all", not "what's its optimal weight" (that would be a separate sweep).
#
# Ollama only actually needed for --relevance-mode llm; started unconditionally here since
# the module load / server startup cost is small and this keeps one script covering all 4
# jobs rather than forking cosine/llm into separate files.

RETRIEVER="${1:?Usage: sbatch iqloc_step2_matrix.sbatch <retriever hybrid-rrf|hybrid-rrf-history> <relevance-mode cosine|llm> [reformulation-mode semantic|literal] [candidate-pool-size] [bm25-repr] [rrf-k] [llm-model]}"
RELEVANCE_MODE="${2:?Usage: sbatch iqloc_step2_matrix.sbatch <retriever> <relevance-mode cosine|llm> [reformulation-mode semantic|literal] [candidate-pool-size] [bm25-repr] [rrf-k] [llm-model]}"
REFORMULATION_MODE="${3:-semantic}"
CANDIDATE_POOL_SIZE="${4:-100}"
BM25_REPR="${5:-refined}"
RRF_K="${6:-100}"
MODEL="${7:-qwen3-coder:30b}"
MANIFEST="iqloc-multi-n30-s42-df69b06700bd"

# Base bm25:embedding ratio is the refined-BM25 winner (1:5, see results/
# iqloc_rrf_weighting_n23_refined.json); history is added at parity (1) on top, not
# independently re-swept -- this step asks "does adding history help at all", not "what's
# its optimal weight" (a separate sweep).
if [ "$RETRIEVER" = "hybrid-rrf-history" ]; then
  RRF_WEIGHTS="1,5,1"
else
  RRF_WEIGHTS="1,5"
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

if [ "$RELEVANCE_MODE" = "llm" ]; then
  ollama serve > "logs/ollama_serve_${SLURM_JOB_ID}.log" 2>&1 &
  OLLAMA_PID=$!
  for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
      break
    fi
    sleep 2
  done
  RELEVANCE_ARGS="--relevance-mode llm --method ollama --model ${MODEL} --num-ctx 32768 --max-tokens 16384"
else
  RELEVANCE_ARGS="--relevance-mode embedding-cosine --relevance-embedding-model Qwen/Qwen3-Embedding-0.6B --keep-fraction 0.5"
fi

.venv_mn5/bin/python scripts/run_relevance_feedback_test.py \
  --manifest "results/manifests/${MANIFEST}.json" \
  --dataset iqloc \
  --pool-size 100 \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --retriever "$RETRIEVER" \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights "$RRF_WEIGHTS" \
  --rrf-k "$RRF_K" \
  --bm25-repr "$BM25_REPR" \
  --granularity chunk \
  --reformulation-mode "$REFORMULATION_MODE" \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  $RELEVANCE_ARGS \
  --output "results/iqloc_step2_${RETRIEVER}_${RELEVANCE_MODE}_${REFORMULATION_MODE}_pool${CANDIDATE_POOL_SIZE}_${BM25_REPR}_n23.json"
PIPELINE_EXIT=$?

if [ "$RELEVANCE_MODE" = "llm" ]; then
  kill "$OLLAMA_PID" 2>/dev/null
fi
exit $PIPELINE_EXIT

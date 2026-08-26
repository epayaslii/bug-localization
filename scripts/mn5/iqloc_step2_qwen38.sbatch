#!/bin/bash
#SBATCH --job-name=iqloc-step2-qwen38
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_step2_qwen38_%j.out
#SBATCH --error=logs/iqloc_step2_qwen38_%j.err

# Step-2-style sample test (2026-08-26) for qwen3.8 (really Qwen3-27B hybrid attention+SSM,
# architecture tag "qwen35") as the LLM relevance judge, now that a manually-transferred
# Ollama v0.33.0 (not the module-loaded 0.11.8, which predates qwen35 support added in
# v0.17.1) has been confirmed to load it correctly and run at normal speed once warm
# (~2000 tok/s prompt eval, ~90 tok/s generation -- see diagnose_qwen38_v3 logs). Fixed at
# the best config found so far: hybrid-rrf-history, symbols_with_imports, RRF 1:15:1, k=30,
# semantic reformulation, candidate-pool-size 200 (the pool-size winner over 100).
#
# Uses the SAME OLLAMA_MODELS store as everything else (qwen3.8 was already transferred and
# is registered there) but points PATH/LD_LIBRARY_PATH at the new binary instead of `module
# load ollama/0.11.8`, since only the new binary can load this architecture.

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
export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models
export PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_v0.33.0/bin:$PATH
export LD_LIBRARY_PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_v0.33.0/lib/ollama:$LD_LIBRARY_PATH

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
  --candidate-pool-size 200 \
  --retriever hybrid-rrf-history \
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" \
  --rrf-weights 1,15,1 \
  --rrf-k 30 \
  --bm25-repr symbols_with_imports \
  --granularity chunk \
  --reformulation-mode semantic \
  --keyword-model "Qwen/Qwen3-Embedding-0.6B" \
  --relevance-mode llm \
  --method ollama \
  --model qwen3.8 \
  --num-ctx 65536 \
  --max-tokens 32768 \
  --output "results/iqloc_step2_qwen38_hybrid-rrf-history_pool200_n23.json"
PIPELINE_EXIT=$?

kill "$OLLAMA_PID" 2>/dev/null
exit $PIPELINE_EXIT

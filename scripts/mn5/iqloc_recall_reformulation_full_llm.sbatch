#!/bin/bash
#SBATCH --job-name=iqloc-recall-reform-full-llm
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-349
#SBATCH --output=logs/iqloc_recall_reform_full_llm_%A_%a.out
#SBATCH --error=logs/iqloc_recall_reform_full_llm_%A_%a.err

# Full-population run, STEP 3 (2026-08-26): winning config from Step 2's n=23 matrix
# (results/iqloc_step2_hybrid-rrf-history_llm_n23.json) -- hybrid-rrf-history (commit-history
# as a 3rd RRF signal, weight 1:15:1) + LLM relevance filtering (qwen2.5-coder:7b via Ollama,
# NOT embedding-cosine) + semantic reformulation, RRF k=30. This is a genuine reversal of
# every earlier LLM-vs-cosine result on this project (7-8 prior replications all favored
# cosine) -- confirmed specifically at Step 1's winning retrieval config (symbols_with_imports,
# Qwen3-Embedding-0.6B), not assumed to transfer from elsewhere. MRR 0.607, Hit@1 47.8%,
# MAP 0.421 at n=23; this run is the real test of whether that holds at full scale, matching
# this project's own repeated "small sample vs. full population" caution.
#
# Same 350-shard/GPU array-job architecture as iqloc_recall_reformulation_full.sbatch (the
# confirmed cosine-mode run, results/iqloc_recall_reform_full_corrected_final.json), but each
# task now ALSO starts its own local Ollama server (same lifecycle as
# scripts/mn5/iqloc_llm_relevance_compare.sbatch) since relevance judging needs a real LLM
# call per instance now, not just embedding math. --time bumped 45->50 min for Ollama startup
# overhead (~10-30s/shard) plus a small buffer given LLM latency is less uniform than
# embedding-cosine's.
#
# --iqloc-strict / manifest / pool-size unchanged from the cosine-mode full run -- this is a
# relevance-filtering + reformulation config change only, not a different dataset cohort.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
module load ollama/0.11.8
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_recall_reform_full_llm_pool200/shards

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
  --model qwen2.5-coder:7b \
  --num-ctx 16384 \
  --max-tokens 8192 \
  --num-shards 350 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_recall_reform_full_llm/shards
PIPELINE_EXIT=$?

kill "$OLLAMA_PID" 2>/dev/null
exit $PIPELINE_EXIT

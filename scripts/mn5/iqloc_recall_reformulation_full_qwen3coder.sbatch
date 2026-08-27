#!/bin/bash
#SBATCH --job-name=iqloc-recall-reform-full-qwen3coder
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-349
#SBATCH --output=logs/iqloc_recall_reform_full_qwen3coder_%A_%a.out
#SBATCH --error=logs/iqloc_recall_reform_full_qwen3coder_%A_%a.err

# Full-population run, STEP 3 rerun (2026-08-26): Qwen3-Coder-30B-A3B (real MoE, 30B total /
# ~3B active params per token) as the LLM relevance judge, in place of qwen2.5-coder:7b.
# Switched BACK to pool=100 (2026-08-26): pool=200's full-scale run (8/350 shards completed,
# results/hpc_mn5_iqloc_recall_reform_full_qwen3coder/shards/) was cancelled mid-run to trade
# the pool=200 quality edge (n=23 MRR 0.632) for speed -- pool=100's own best config
# (hybrid-rrf-history + semantic, MRR 0.617, Hit@1 47.8%, MAP 0.425) is a real but modest
# step down, at roughly half the per-instance LLM judging cost (fewer candidates per prompt).
# This is a deliberate speed/quality tradeoff, not a correction of a mistake.
#
# Uses the standard module-loaded Ollama 0.11.8 (qwen3-coder's MoE architecture is
# well-supported, unlike qwen3.8's brand-new hybrid-SSM architecture which needed a manually
# transferred newer binary). --time bumped to 90 min (vs qwen2.5-coder's 45-50 min) since
# qwen3-coder runs ~60-120s/instance vs qwen2.5-coder's ~13-30s (bigger model, ~4x qwen2.5-
# coder:7b's parameter count even with MoE's active-parameter speedup). One CUDA OOM was seen
# during the n=23 test (transient GPU contention on one node, resolved on retry) -- expect a
# somewhat higher per-shard retry/timeout rate than the qwen2.5-coder run given less GPU
# memory headroom per node (18GB model vs 4.7GB).

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
module load ollama/0.11.8
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_recall_reform_full_qwen3coder/shards

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
  --model qwen3-coder:30b \
  --num-ctx 32768 \
  --max-tokens 16384 \
  --num-shards 350 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_recall_reform_full_qwen3coder/shards
PIPELINE_EXIT=$?

kill "$OLLAMA_PID" 2>/dev/null
exit $PIPELINE_EXIT

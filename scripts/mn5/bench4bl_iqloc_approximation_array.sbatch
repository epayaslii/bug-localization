#!/bin/bash
#SBATCH --job-name=bench4bl-iqloc-n200
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --output=logs/bench4bl_iqloc_array_%A_%a.out
#SBATCH --error=logs/bench4bl_iqloc_array_%A_%a.err

# IQLoc-approximation pipeline (BM25 skeleton -> Ollama chunk relevance judge -> EmbedRank/MMR
# keyword extraction -> cosine reformulation -> BM25 rerank), n=200 diverse manifest, sharded
# 20-way. GPU-accelerated Ollama (qwen2.5-coder:7b, confirmed working this session: real chat
# completion in 5.4s on the H100, vs. up to 460s/instance observed on a CPU-only local Mac) and
# GPU-accelerated UniXCoder for keyword-extraction embeddings -- both fully offline, model
# weights already transferred to ollama_models/ and hf_cache_mn5/ from earlier sessions.
#
# Each array task starts its OWN local `ollama serve` (Ollama has no shared-server mode across
# nodes) before running its shard, and kills it after -- this is genuinely new plumbing, no
# prior MN5 sbatch script in this project combines Ollama + Slurm array sharding.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
module load ollama/0.11.8
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_n200/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1
export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models

ollama serve > "logs/ollama_serve_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!

# Wait for the server to actually accept requests before starting the shard -- model load
# onto the GPU takes a few seconds, and a request against a not-yet-ready server hard-fails
# rather than queuing.
for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
    break
  fi
  sleep 2
done

.venv_mn5/bin/python scripts/run_iqloc_approximation_shard.py \
  --manifest results/manifests/bench4bl-multi-n200-s42.json \
  --dataset bench4bl \
  --pool-size 4418 \
  --candidate-pool-size 50 \
  --keyword-model "microsoft/unixcoder-base" \
  --model qwen2.5-coder-7b \
  --num-shards 20 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_n200/shards

# $? from the python run above is the job's real exit status; the kill on the last line
# would otherwise become the last command Slurm inspects, and `kill` on an Ollama server
# that already exited on its own returns nonzero -- Slurm reports that as a job FAILURE
# even though the pipeline itself completed and wrote a valid shard file (confirmed on
# job 44814164: 9/20 tasks marked FAILED in sacct despite 20/20 shards aggregating
# cleanly with real results). Capture the real status, kill best-effort, then exit with
# the status that actually reflects whether the pipeline succeeded.
PIPELINE_STATUS=$?
kill "$OLLAMA_PID" 2>/dev/null
exit "$PIPELINE_STATUS"

#!/bin/bash
#SBATCH --job-name=diagnose-qwen38
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/diagnose_qwen38_%j.out
#SBATCH --error=logs/diagnose_qwen38_%j.err

# One-off diagnostic (2026-08-26): qwen3.8 (really Qwen3-27B q4_K_M, 16.8GB blob -- confirmed
# intact, size matches manifest exactly) has failed with a generic "unable to load model"
# API error every time it's been tried on MN5. Capturing Ollama's own server-side log (not
# just the API's terse error response) to find the real cause -- likely GPU VRAM, since this
# is a much bigger model (27B) than qwen2.5-coder (7B), which has always worked fine.

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

export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

ollama serve > "logs/ollama_serve_diagnose_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
    break
  fi
  sleep 2
done

echo "=== Attempting to load qwen3.8 via /api/generate ==="
curl -s http://localhost:11434/api/generate -d '{"model": "qwen3.8", "prompt": "hello", "stream": false}'
echo
echo "=== nvidia-smi after load attempt ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

sleep 3
echo "=== Ollama server log (last 60 lines) ==="
tail -n 60 "logs/ollama_serve_diagnose_${SLURM_JOB_ID}.log"

kill "$OLLAMA_PID" 2>/dev/null

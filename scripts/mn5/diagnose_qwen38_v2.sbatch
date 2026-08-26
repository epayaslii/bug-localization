#!/bin/bash
#SBATCH --job-name=diagnose-qwen38-v2
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/diagnose_qwen38_v2_%j.out
#SBATCH --error=logs/diagnose_qwen38_v2_%j.err

# Retest qwen3.8 with a manually-transferred Ollama v0.33.0 (not the module-loaded 0.11.8) --
# root cause confirmed 2026-08-26: qwen3.8 is really Qwen3-27B (hybrid attention+SSM,
# architecture tag "qwen35"), which Ollama only added support for at v0.17.1+. Using the
# newer binary directly by path, not via `module load ollama/...`.

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs

export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models
export PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_v0.33.0/bin:$PATH
export LD_LIBRARY_PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_v0.33.0/lib/ollama:$LD_LIBRARY_PATH

echo "=== ollama version ==="
ollama --version

ollama serve > "logs/ollama_serve_diagnose_v2_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
    break
  fi
  sleep 2
done

echo "=== Attempting to load qwen3.8 via /api/generate ==="
curl -s http://localhost:11434/api/generate -d '{"model": "qwen3.8", "prompt": "Say hello in one word.", "stream": false}'
echo
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

sleep 3
echo "=== Ollama server log (last 60 lines) ==="
tail -n 60 "logs/ollama_serve_diagnose_v2_${SLURM_JOB_ID}.log"

kill "$OLLAMA_PID" 2>/dev/null

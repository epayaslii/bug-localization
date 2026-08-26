#!/bin/bash
#SBATCH --job-name=diagnose-qwen38-v3
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/diagnose_qwen38_v3_%j.out
#SBATCH --error=logs/diagnose_qwen38_v3_%j.err

# Follow-up to diagnose_qwen38_v2 (worked! but prompt eval was 0.65 tok/s on a 16-token
# prompt, 24.5s just for prefill). Testing whether that's one-time warmup (CUDA graph
# compile, cache init) or a real per-token bottleneck, by sending a SECOND request in the
# SAME warm server session with a long, realistic prompt (~2500 tokens, matching real
# chunk-relevance prompt sizes for a 100-candidate pool).

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs

export OLLAMA_MODELS=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_models
export PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_v0.33.0/bin:$PATH
export LD_LIBRARY_PATH=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/ollama_v0.33.0/lib/ollama:$LD_LIBRARY_PATH

ollama serve > "logs/ollama_serve_diagnose_v3_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 30); do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags | grep -q 200; then
    break
  fi
  sleep 2
done

echo "=== Call 1 (short prompt, warm-up) ==="
time curl -s http://localhost:11434/api/generate -d '{"model": "qwen3.8", "prompt": "Say hello in one word.", "stream": false}' | python3 -c "import json,sys; d=json.load(sys.stdin); print('prompt_eval_count:', d.get('prompt_eval_count'), 'prompt_eval_duration_s:', d.get('prompt_eval_duration',0)/1e9, 'eval_count:', d.get('eval_count'), 'eval_duration_s:', d.get('eval_duration',0)/1e9)"

echo "=== Building a ~2500-token realistic prompt ==="
python3 -c "
import json
lorem = ' '.join(['public class Example { void method' + str(i) + '() { return processData(input, ' + str(i) + '); } }' for i in range(120)])
prompt = 'You are judging code relevance. Bug report: NullPointerException when processing empty input. Here are 100 candidate code chunks, judge each for relevance:\n' + lorem
with open('/tmp/long_prompt.json', 'w') as f:
    json.dump({'model': 'qwen3.8', 'prompt': prompt, 'stream': False}, f)
print('prompt chars:', len(prompt))
"

echo "=== Call 2 (long realistic prompt, server already warm) ==="
time curl -s http://localhost:11434/api/generate -d @/tmp/long_prompt.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('prompt_eval_count:', d.get('prompt_eval_count'), 'prompt_eval_duration_s:', d.get('prompt_eval_duration',0)/1e9, 'eval_count:', d.get('eval_count'), 'eval_duration_s:', d.get('eval_duration',0)/1e9)"

kill "$OLLAMA_PID" 2>/dev/null

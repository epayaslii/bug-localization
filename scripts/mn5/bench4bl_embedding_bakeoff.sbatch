#!/bin/bash
#SBATCH --job-name=bench4bl-embed-bakeoff
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bench4bl_embed_bakeoff_%j.out
#SBATCH --error=logs/bench4bl_embed_bakeoff_%j.err

# Embedding-model bake-off on Bench4BL, n=30, GPU-accelerated -- the 4 local HF models
# only (unixcoder, codebert, bge-code-v1, qwen3-embedding-0.6b). OpenAI/Voyage are API-based
# and can't run here (MN5 has no outbound internet) -- those run locally instead, see
# scripts/mn5/../compare_embedding_models.py invocation without --dataset override needed.
# bge-code-v1 (2B params) was previously excluded on CPU (~18min/instance) -- now viable
# given the confirmed ~30-40x GPU speedup on the embedding step for other models.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/compare_embedding_models.py \
  --manifest results/manifests/bench4bl-multi-n30-s42-mn5-8proj.json \
  --dataset bench4bl \
  --pool-size 50000 \
  --candidate-pool-size 200 \
  --models unixcoder codebert bge-code-v1 qwen3-embedding-0.6b \
  --output results/embedding_bakeoff_bench4bl_30_gpu.json

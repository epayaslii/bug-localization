#!/bin/bash
#SBATCH --job-name=iqloc-embed-bakeoff
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/iqloc_embed_bakeoff_%j.out
#SBATCH --error=logs/iqloc_embed_bakeoff_%j.err

# Embedding-model bake-off on IQLoc's own dataset, n=30, before committing an embedding
# model to the full ~7,100-instance run. Qwen3 + Jina here since both are fully local/
# offline (GPU-accelerated); OpenAI needs internet and runs as a separate local-Mac step
# (scripts/compare_embedding_models.py --models openai-3-small), since MN5 has none.

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
  --manifest results/manifests/iqloc-multi-n30-s42-df69b06700bd.json \
  --dataset iqloc \
  --pool-size 100 \
  --candidate-pool-size 200 \
  --models qwen3-embedding-0.6b jina-embeddings-v2-base-code \
  --output results/embedding_bakeoff_iqloc_30_mn5.json

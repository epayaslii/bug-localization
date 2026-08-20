#!/bin/bash
#SBATCH --job-name=bench4bl-embed-jina
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bench4bl_embed_jina_%j.out
#SBATCH --error=logs/bench4bl_embed_jina_%j.err

# Jina embedding bake-off on Bench4BL, n=30, GPU-accelerated. Split out from
# bench4bl_embedding_bakeoff.sbatch since jina-embeddings-v2-base-code needs
# trust_remote_code=True (a second HF repo, jinaai/jina-bert-v2-qk-post-norm, for its
# custom modeling code) plus einops -- both packaged and transferred separately
# (hf_cache_mn5_jina.tar.gz, wheelhouse_mn5/einops-0.8.2-py3-none-any.whl) since MN5 has
# no outbound internet (HF_HUB_OFFLINE=1 below).

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
  --manifest results/manifests/bench4bl-multi-n30-s42-8c4f91c33f21.json \
  --dataset bench4bl \
  --pool-size 100000 \
  --candidate-pool-size 200 \
  --models jina-embeddings-v2-base-code \
  --output results/embedding_bakeoff_bench4bl_30_jina_mn5.json

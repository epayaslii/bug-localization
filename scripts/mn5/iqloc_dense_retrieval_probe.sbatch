#!/bin/bash
#SBATCH --job-name=iqloc-dense-probe
#SBATCH --account=ehpc680
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=logs/iqloc_dense_probe_%A_%a.out
#SBATCH --error=logs/iqloc_dense_probe_%A_%a.err

# Same true dense-retrieval-only probe as bench4bl_dense_retrieval_probe.sbatch (chunked
# Qwen3-Embedding ranking over the FULL repo corpus per bug, no BM25 pre-filter at all),
# pointed at IQLoc's corrected-dataset n=30 manifest -- added 2026-08-26 so the comparison
# matrix has a genuine embedding-alone number to sit next to BM25-alone (results/
# commit_history_rerank_iqloc_30.json's "content" config, MRR 0.2046) and the fused
# hybrid-rrf-history result, instead of only ever showing the fused number.

module load intel
module load mkl
module load impi
module load hdf5
module load python/3.11.5-gcc
unset PYTHONHOME
unset PYTHONPATH

cd /gpfs/projects/ehpc680/comm842299/repos/bug-localization/
mkdir -p logs results/hpc_mn5_iqloc_dense_probe/shards

export HF_HOME=/gpfs/projects/ehpc680/comm842299/repos/bug-localization/hf_cache_mn5
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export USE_TORCH=1

.venv_mn5/bin/python scripts/run_dense_retrieval_shard.py \
  --manifest results/manifests/iqloc-multi-n30-s42-df69b06700bd.json \
  --dataset iqloc \
  --pool-size 100 \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --num-shards 8 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir results/hpc_mn5_iqloc_dense_probe/shards

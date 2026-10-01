#!/bin/bash
#SBATCH --job-name=rankgpt-pg
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err

# Usage:  sbatch slurm/rerank.sbatch configs/llama31_8b_top20.yaml
set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/rerank.sbatch <config.yaml>}

mkdir -p results/logs

module purge
# module load cuda/12.1       # <-- adjust to your cluster's module names
source /home/cccp/25m2125/VAMSI/envs/rankgpt/bin/activate

# Compute nodes have no internet: force every library into offline mode so a
# stray network call fails loudly at startup instead of hanging for 10 minutes.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/cccp/25m2125/VAMSI/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "node      : $(hostname)"
echo "job id    : ${SLURM_JOB_ID}"
echo "config    : ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONPATH=src python scripts/run_rerank.py --config "${CONFIG}"

echo "done at $(date)"

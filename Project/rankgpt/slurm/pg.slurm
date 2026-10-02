#!/bin/bash
#SBATCH --job-name=rankgpt_pg
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/cccp/25m2125/VAMSI/cs6101/project/CS6101/Project/rankgpt/results/logs/pg_%j.out
#SBATCH --error=/home/cccp/25m2125/VAMSI/cs6101/project/CS6101/Project/rankgpt/results/logs/pg_%j.err

set -eo pipefail

PROJECT=/home/cccp/25m2125/VAMSI/cs6101/project/CS6101/Project/rankgpt
cd "$PROJECT"
mkdir -p results/logs

CONFIG="${1:?usage: sbatch slurm/pg.slurm configs/<run>.yaml [extra run_pg.py args]}"
shift

# --- Environment ---
source /home/cccp/25m2125/miniconda3/etc/profile.d/conda.sh
conda activate rankgpt

# Compute nodes have no internet: everything must resolve from local cache.
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Silences the TRANSFORMERS_CACHE deprecation warning you saw.
export HF_HOME="$PROJECT/.hf_home"
unset TRANSFORMERS_CACHE

export TOKENIZERS_PARALLELISM=false

echo "=== job $SLURM_JOB_ID on $(hostname) | $(date -Is) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "config : $CONFIG"
echo "extra  : $*"
echo "commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

python scripts/run_pg.py "$CONFIG" "$@"

echo
echo "=== done | $(date -Is) ==="
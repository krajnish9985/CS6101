#!/bin/bash
# RUN ON THE LOGIN NODE (needs internet). One time only.
set -euo pipefail

ENV_DIR=/home/cccp/25m2125/VAMSI/envs/rankgpt

module purge
# module load python/3.10 cuda/12.1    # <-- adjust to whatever `module avail` shows

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install vllm transformers accelerate
pip install pyyaml pytrec_eval pandas tqdm pytest

python -c "import vllm, transformers; print('vllm', vllm.__version__); print('transformers', transformers.__version__)"
echo "Environment ready at $ENV_DIR"

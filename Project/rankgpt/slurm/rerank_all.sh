#!/bin/bash
# Submit every midterm experiment at once. Run from the repo root.
set -euo pipefail

for cfg in \
    configs/llama31_8b_top20.yaml \
    configs/qwen25_7b_top20.yaml \
    configs/llama31_8b_top20_shortprompt.yaml \
    configs/llama31_8b_top20_multiturn.yaml \
    configs/llama31_8b_top20_words60.yaml \
    configs/llama31_8b_top20_shuffled.yaml
do
    name=$(basename "$cfg" .yaml)
    echo "submitting $name"
    sbatch --job-name="pg-${name}" slurm/rerank.sbatch "$cfg"
done

# The top-100 sliding-window run is ~9x the LLM calls: give it the longer limit.
sbatch --job-name=pg-sliding --time=06:00:00 \
    slurm/rerank.sbatch configs/llama31_8b_top100_sliding.yaml

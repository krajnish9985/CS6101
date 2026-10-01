#!/bin/bash
# RUN ON THE LOGIN NODE (needs internet). Compute nodes are offline, so every
# weight must be on disk before any job is submitted.
set -euo pipefail

MODEL_DIR=/home/cccp/25m2125/VAMSI/models
mkdir -p "$MODEL_DIR"

source /home/cccp/25m2125/VAMSI/envs/rankgpt/bin/activate
pip install -q "huggingface_hub[cli]"

# Llama is a gated repo: run `hf auth login` once and accept the licence on the
# model page first, otherwise the download 401s.
hf download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir "$MODEL_DIR/Llama-3.1-8B-Instruct"

# Qwen is ungated.
hf download Qwen/Qwen2.5-7B-Instruct \
    --local-dir "$MODEL_DIR/Qwen2.5-7B-Instruct"

# Optional stretch model for the size-scaling slide (~30 GB, DGX only).
# hf download Qwen/Qwen2.5-32B-Instruct \
#     --local-dir "$MODEL_DIR/Qwen2.5-32B-Instruct"

du -sh "$MODEL_DIR"/*

# RankGPT reimplementation - LLM Permutation Generation (Member 3)

Listwise re-ranking with a local open-source instruct LLM, following
*Is ChatGPT Good at Search? Investigating Large Language Models as
Re-Ranking Agents* (Sun et al., 2023).

This module takes the BM25 candidate lists produced upstream, asks an LLM to
emit a **permutation** of those candidates, parses and repairs that
permutation, and writes a TREC run file for evaluation.

```
reranking_data.jsonl  ->  prompt  ->  LLM  ->  "[3] > [1] > [2] ..."  ->  parse/repair  ->  run.trec
```

---

## Layout

```
configs/                    one YAML per experiment
src/rankgpt/
  data.py                   loads reranking_data.jsonl into Candidate/QueryRecord
  prompts.py                three prompt styles for the ablation
  permutation.py            parse + repair LLM output into a valid permutation
  llm.py                    batched vLLM wrapper (+ offline stub for dry runs)
  rerank.py                 single-window and sliding-window PG
  runfile.py                TREC run writing + metrics
scripts/
  smoke_test.py             inspect prompts and a couple of real outputs
  run_rerank.py             main entry point: config in, run file out
  compare_runs.py           BM25 vs cross-encoder vs LLM table
  qualitative_examples.py   biggest wins and losses for the slides
slurm/                      env setup, model download, sbatch templates
tests/                      unit tests for the parser
```

Nothing here writes to Member 1's, 2's or 4's files, so merges stay clean.

---

## Inputs expected from the team

| From | File | Used for |
|---|---|---|
| Member 1 | `data/scifact/reranking_data.jsonl` | queries, BM25 candidates, passage text, relevance |
| Member 1 | `data/scifact/qrels.tsv` | evaluation |
| Member 2 | `results/runs/bm25.trec`, `results/runs/minilm_ce.trec` | comparison table |

`reranking_data.jsonl` already contains everything needed (passage title and
text are embedded per candidate), so this module never touches the corpus or
the Lucene index.

---

## Setup

Everything that needs the network happens on the **login node**; compute nodes
are offline.

```bash
# 1. environment (login node, once)
bash slurm/setup_env.sh

# 2. model weights (login node, once; accept the Llama licence on HF first)
bash slurm/download_models.sh

# 3. copy the data from Member 1 into data/scifact/
```

---

## Running

```bash
source /home/cccp/25m2125/VAMSI/envs/rankgpt/bin/activate

# A. no GPU, no model: check the prompt looks right
PYTHONPATH=src python scripts/smoke_test.py \
    --config configs/llama31_8b_top20.yaml --show-prompt --no-model

# B. no GPU: exercise the whole pipeline with a stub model
PYTHONPATH=src python scripts/run_rerank.py \
    --config configs/llama31_8b_top20.yaml --dry-run --limit 5

# C. real model, 2 queries, interactive GPU
srun --partition=interactive --gres=gpu:1 --cpus-per-task=8 --mem=64G \
     --time=01:00:00 --pty bash
PYTHONPATH=src python scripts/smoke_test.py \
    --config configs/llama31_8b_top20.yaml -n 2

# D. full run as a batch job
sbatch slurm/rerank.sbatch configs/llama31_8b_top20.yaml

# E. everything at once
bash slurm/rerank_all.sh
```

Each run writes three things:

- `results/runs/<run_name>.trec` - the ranking, for evaluation
- `results/raw/<run_name>.jsonl` - prompt stats, raw LLM text, parse diagnostics
- `results/metrics/<run_name>.json` - metrics plus parse-quality rates

---

## Comparison table

```bash
PYTHONPATH=src python scripts/compare_runs.py \
    --qrels data/scifact/qrels.tsv \
    --run BM25=results/runs/bm25.trec \
    --run CrossEncoder=results/runs/minilm_ce.trec \
    --run Llama31-8B=results/runs/llama31-8b-top20-single.trec \
    --out results/metrics/comparison.csv
```

SciFact BM25 baseline to beat (300 queries): nDCG@1 0.5533, nDCG@10 0.6789,
MRR@10 0.6457, Recall@100 0.9253.

---

## Experiment grid

| Config | Varies |
|---|---|
| `llama31_8b_top20.yaml` | main run |
| `qwen25_7b_top20.yaml` | model family |
| `llama31_8b_top20_shortprompt.yaml` | minimal instructions |
| `llama31_8b_top20_multiturn.yaml` | the paper's original prompt |
| `llama31_8b_top20_words60.yaml` | passage truncation |
| `llama31_8b_top20_shuffled.yaml` | input-order sensitivity |
| `llama31_8b_top100_sliding.yaml` | top-100, window 20 / step 10 |

---

## Design notes

**Only ranks 1-20 are reranked, but the run file still holds 100 documents.**
Ranks 21-100 are appended in their original BM25 order, so Recall@100 and MAP
remain directly comparable with the baselines instead of collapsing to a
20-document ceiling.

**Output repair is deterministic and measured.** Small models emit duplicated,
missing and out-of-range identifiers. The parser drops out-of-range ids, keeps
the first occurrence of duplicates, and appends anything missing in its
original BM25 order. A completely unusable response therefore degrades
gracefully to the BM25 ordering rather than crashing. Every run reports
`clean_rate`, `any_missing_rate` and friends, which is a result in its own
right: it separates "the model ranks badly" from "the model cannot follow the
format".

**Every LLM call is batched across queries**, including inside the sliding
window, where all queries advance through window index *i* together. A full
300-query SciFact top-20 run is a single vLLM batch.

**Greedy decoding** (`temperature: 0.0`) because permutation generation is a
deterministic task and we want runs to be reproducible.

---

## Tests

```bash
PYTHONPATH=src pytest tests/ -q
```

Covers duplicates, missing and out-of-range identifiers, prose around the
ranking, refusals, empty output, bare-number fallback, and the window
schedule's coverage of the full list.

---

## Hand-off to Member 4

`rerank.py` exposes `window_schedule(n, window_size, step)` and
`rerank_sliding_window(...)` as a reference implementation of the paper's
bottom-up sliding window. Integration into the end-to-end pipeline is Member
4's call: either import these or replace them, but keep
`results/runs/*.trec` as the interface to evaluation.

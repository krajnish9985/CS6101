#!/usr/bin/env python
"""Generate the run-config matrix so ten YAMLs stay consistent by construction.

Writing them by hand invites a copy-paste error that silently changes one
cell of the results table. This emits every (dataset x model x prompt)
combination from a single definition.

Usage
-----
    python scripts/make_configs.py --list
    python scripts/make_configs.py            # writes configs/
    python scripts/make_configs.py --force    # overwrite existing
"""

from __future__ import annotations

import argparse
from pathlib import Path

# name -> (data filename stem, note about what makes this dataset different)
DATASETS = {
    "scifact": ("scifact", "5.2k corpus, 300 queries, ~1.1 judged per query, binary labels"),
    "nfcorpus": ("nfcorpus", "3.6k corpus, 323 queries, ~38 judged per query, graded labels"),
    "covid": ("covid", "171k corpus, 50 queries, ~1327 judged per query, graded labels"),
    "dbpedia": ("dbpedia", "4.6M corpus, 400 queries, ~109 judged per query, graded labels"),
}

MODELS = {
    "llama8b": ("models/Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct", 1),
    "qwen7b": ("models/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct", 1),
}

# Both prompt variants on every dataset: the compact-vs-paper comparison
# then rests on eight cells rather than one, which is what makes it a
# finding rather than an anecdote.
PROMPTS = ("paper", "compact")
ABLATION_PROMPTS = {}

TEMPLATE = """\
run_name: {run_name}
dataset: {dataset}
data: data/{dataset}/{stem}_reranking_data.jsonl

depth: {depth}
prompt: {prompt}
max_passage_words: 300

limit: null
chunk_size: {chunk_size}

notes: >
  {notes}

model:
  model_path: {model_path}
  model_label: {model_label}
  dtype: bfloat16
  max_model_len: {max_model_len}
  gpu_memory_utilization: 0.90
  tensor_parallel_size: {tensor_parallel_size}
  enforce_eager: false
  seed: 0
  temperature: 0.0
  top_p: 1.0
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="configs", help="output directory")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--list", action="store_true",
                        help="print the matrix and the sbatch lines, write nothing")
    return parser.parse_args()


def combinations():
    for dataset, (stem, dataset_note) in DATASETS.items():
        prompts = ABLATION_PROMPTS.get(dataset, PROMPTS)
        for model_key, (model_path, model_label, tensor_parallel) in MODELS.items():
            for prompt in prompts:
                yield {
                    "dataset": dataset,
                    "stem": stem,
                    "dataset_note": dataset_note,
                    "model_key": model_key,
                    "model_path": model_path,
                    "model_label": model_label,
                    "tensor_parallel_size": tensor_parallel,
                    "prompt": prompt,
                }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)

    planned = []
    for combo in combinations():
        run_name = f"{combo['model_key']}_{combo['dataset']}_top{args.depth}_{combo['prompt']}"
        filename = out_dir / f"{run_name}.yaml"
        notes = (
            f"{combo['model_label']} on {combo['dataset']} "
            f"({combo['dataset_note']}), {combo['prompt']} prompt, "
            f"top-{args.depth}, greedy decoding, single window."
        )
        body = TEMPLATE.format(
            run_name=run_name,
            dataset=combo["dataset"],
            stem=combo["stem"],
            depth=args.depth,
            prompt=combo["prompt"],
            chunk_size=args.chunk_size,
            notes=notes,
            model_path=combo["model_path"],
            model_label=combo["model_label"],
            max_model_len=args.max_model_len,
            tensor_parallel_size=combo["tensor_parallel_size"],
        )
        planned.append((filename, body, run_name))

    if args.list:
        print(f"{len(planned)} runs:\n")
        for filename, _, run_name in planned:
            print(f"  {run_name:44} -> {filename}")
        print("\nsbatch lines:\n")
        for filename, _, run_name in planned:
            print(f"sbatch --job-name={run_name} slurm/pg.slurm {filename}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for filename, body, _ in planned:
        if filename.exists() and not args.force:
            print(f"  skip (exists): {filename}")
            skipped += 1
            continue
        filename.write_text(body, encoding="utf-8")
        print(f"  wrote: {filename}")
        written += 1

    print(f"\n{written} written, {skipped} skipped. Use --force to overwrite.")
    print("Run with --list to print the matching sbatch lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
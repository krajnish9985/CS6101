#!/usr/bin/env python
"""Score several TREC run files side by side and emit the comparison table.

    PYTHONPATH=src python scripts/compare_runs.py \
        --qrels data/scifact/qrels.tsv \
        --run BM25=results/runs/bm25.trec \
        --run CrossEncoder=results/runs/minilm_ce.trec \
        --run "RankLLM (Llama-3.1-8B)"=results/runs/llama31-8b-top20.trec \
        --out results/metrics/comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rankgpt.runfile import evaluate_run, load_qrels, read_trec_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qrels", required=True)
    p.add_argument("--run", action="append", required=True, help="NAME=path.trec")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    qrels = load_qrels(args.qrels)
    rows = []
    for spec in args.run:
        name, _, path = spec.partition("=")
        run = read_trec_run(path)
        sub = {q: v for q, v in qrels.items() if q in run}
        scores = evaluate_run(run, sub)
        rows.append({"System": name, **scores})

    cols = ["System", "Queries", "nDCG@1", "nDCG@10", "MRR@10", "MAP", "Recall@100"]
    header = " | ".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:>14.4f}" if isinstance(v, float) else f"{str(v):>14}")
        print(" | ".join(cells))

    if args.out:
        import csv

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()

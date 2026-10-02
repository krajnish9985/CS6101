#!/usr/bin/env python
"""Join every run's summary.json into the comparison tables for the report.

Reads results/runs/*/summary.json and emits Markdown (and optionally CSV)
grouped by dataset, so BM25 and each model sit side by side.

Usage
-----
    python scripts/compare_runs.py
    python scripts/compare_runs.py --standard
    python scripts/compare_runs.py --csv results/comparison.csv
    python scripts/compare_runs.py --out results/comparison.md
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

METRIC_KEYS = ("nDCG@1", "nDCG@10", "MRR@10")

# Datasets appear in this order when present; anything else is appended.
DATASET_ORDER = ["scifact", "nfcorpus", "covid", "dbpedia"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("--out", default=None, help="write Markdown here as well as stdout")
    parser.add_argument("--csv", default=None, help="also write a flat CSV")
    parser.add_argument("--standard", action="store_true",
                        help="use standard BEIR nDCG instead of pool-relative")
    parser.add_argument("--skip", nargs="*", default=["probe"],
                        help="substrings of run names to exclude")
    return parser.parse_args()


def load_runs(root: Path, skip: List[str]) -> List[Dict[str, Any]]:
    runs = []
    for summary_path in sorted(root.glob("*/summary.json")):
        run_name = summary_path.parent.name
        if any(token in run_name for token in skip):
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        config_path = summary_path.parent / "config.json"
        config: Dict[str, Any] = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8")).get("config", {})

        runs.append({
            "run_name": run_name,
            "dataset": config.get("dataset", "unknown"),
            "model_label": config.get("model", {}).get("model_label", "unknown"),
            "prompt": config.get("prompt", "unknown"),
            "depth": config.get("depth", 0),
            "summary": summary,
        })
    return runs


def metric_block(summary: Dict[str, Any], standard: bool) -> tuple:
    if standard:
        return summary.get("bm25_standard"), summary.get("llm_standard")
    return summary.get("bm25"), summary.get("llm")


def build_markdown(runs: List[Dict[str, Any]], standard: bool) -> str:
    flavour = ("standard BEIR nDCG (ideal DCG from full qrels)" if standard
               else "pool-relative nDCG (ideal DCG from the candidate pool)")

    lines = [
        "# Permutation generation: cross-dataset comparison",
        "",
        f"Metric flavour: **{flavour}**.",
        "",
        "`delta` is the LLM re-ranker minus the BM25 baseline it re-ranked.",
        "",
    ]

    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        by_dataset.setdefault(run["dataset"], []).append(run)

    ordered = [d for d in DATASET_ORDER if d in by_dataset]
    ordered += [d for d in sorted(by_dataset) if d not in ordered]

    for dataset in ordered:
        dataset_runs = by_dataset[dataset]
        first = dataset_runs[0]["summary"]

        lines += [
            f"## {dataset}",
            "",
            f"- queries: {first.get('n_queries', '?')}"
            f" | with a relevant doc in the top-{dataset_runs[0]['depth']} window: "
            f"{first.get('n_queries_with_gold_in_window', '?')}"
            f" | mean judged relevant per query: "
            f"{first.get('mean_gold_judged_per_query', '?')}",
            "",
            "| method | prompt | nDCG@1 | nDCG@10 | MRR@10 | clean % | degenerate % |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        bm25, _ = metric_block(first, standard)
        if bm25:
            lines.append(
                f"| BM25 (stage 1) | — | {bm25['nDCG@1']:.4f} | "
                f"{bm25['nDCG@10']:.4f} | {bm25['MRR@10']:.4f} | — | — |"
            )

        for run in sorted(dataset_runs, key=lambda r: (r["model_label"], r["prompt"])):
            summary = run["summary"]
            _, llm = metric_block(summary, standard)
            if not llm:
                continue
            parse = summary.get("parse_quality", {})
            lines.append(
                f"| {run['model_label']} | {run['prompt']} | "
                f"{llm['nDCG@1']:.4f} | {llm['nDCG@10']:.4f} | {llm['MRR@10']:.4f} | "
                f"{parse.get('pct_clean', float('nan')):.1f} | "
                f"{parse.get('pct_degenerate_output', float('nan')):.1f} |"
            )

        lines += ["", "Deltas against BM25:", "",
                  "| method | prompt | nDCG@1 | nDCG@10 | MRR@10 |",
                  "| --- | --- | --- | --- | --- |"]

        for run in sorted(dataset_runs, key=lambda r: (r["model_label"], r["prompt"])):
            summary = run["summary"]
            base, llm = metric_block(summary, standard)
            if not (base and llm):
                continue
            cells = " | ".join(f"{llm[key] - base[key]:+.4f}" for key in METRIC_KEYS)
            lines.append(f"| {run['model_label']} | {run['prompt']} | {cells} |")

        lines.append("")

    return "\n".join(lines) + "\n"


def write_csv(runs: List[Dict[str, Any]], path: Path, standard: bool) -> None:
    fieldnames = [
        "run_name", "dataset", "model_label", "prompt", "depth",
        "n_queries", "mean_gold_judged_per_query",
        "bm25_nDCG@1", "bm25_nDCG@10", "bm25_MRR@10",
        "llm_nDCG@1", "llm_nDCG@10", "llm_MRR@10",
        "delta_nDCG@1", "delta_nDCG@10", "delta_MRR@10",
        "pct_clean", "pct_degenerate_output", "pct_with_missing",
        "mean_identifier_inflation", "pct_context_overflow",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run in runs:
            summary = run["summary"]
            base, llm = metric_block(summary, standard)
            if not (base and llm):
                continue
            parse = summary.get("parse_quality", {})

            row = {
                "run_name": run["run_name"],
                "dataset": run["dataset"],
                "model_label": run["model_label"],
                "prompt": run["prompt"],
                "depth": run["depth"],
                "n_queries": summary.get("n_queries"),
                "mean_gold_judged_per_query": summary.get("mean_gold_judged_per_query"),
            }
            for key in METRIC_KEYS:
                row[f"bm25_{key}"] = round(base[key], 4)
                row[f"llm_{key}"] = round(llm[key], 4)
                row[f"delta_{key}"] = round(llm[key] - base[key], 4)
            for key in ("pct_clean", "pct_degenerate_output", "pct_with_missing",
                        "mean_identifier_inflation", "pct_context_overflow"):
                row[key] = parse.get(key)
            writer.writerow(row)

    print(f"wrote {path}")


def main() -> int:
    args = parse_args()

    runs = load_runs(Path(args.results_root), args.skip)
    if not runs:
        raise SystemExit(f"no summary.json found under {args.results_root}")

    markdown = build_markdown(runs, args.standard)
    print(markdown)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"wrote {out_path}")

    if args.csv:
        write_csv(runs, Path(args.csv), args.standard)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python
"""Inspect the raw LLM outputs of a finished run.

After a job completes, this is how you find out *why* the metrics look
the way they do: it prints what the model actually generated, what the
parser recovered, and where the relevant document ended up.

Usage
-----
    python scripts/show_outputs.py probe_llama8b
    python scripts/show_outputs.py llama8b_top20_paper --only-broken -n 10
    python scripts/show_outputs.py llama8b_top20_paper --only-helped -n 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_name", help="directory name under results/runs/")
    parser.add_argument("--results-root", default="results/runs")
    parser.add_argument("-n", "--num", type=int, default=5, help="how many queries to show")
    parser.add_argument("--chars", type=int, default=300, help="raw output characters to print")

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only-broken", action="store_true",
                           help="only queries whose permutation needed repair")
    selection.add_argument("--only-clean", action="store_true",
                           help="only queries that parsed cleanly")
    selection.add_argument("--only-helped", action="store_true",
                           help="only queries where nDCG@10 improved")
    selection.add_argument("--only-hurt", action="store_true",
                           help="only queries where nDCG@10 got worse")
    selection.add_argument("--qid", help="one specific query_id")

    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Did the job finish? Check results/logs/ for the job log."
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.qid:
        return [row for row in rows if row["query_id"] == str(args.qid)]
    if args.only_broken:
        return [row for row in rows if not row["parse"]["is_clean"]]
    if args.only_clean:
        return [row for row in rows if row["parse"]["is_clean"]]
    if args.only_helped:
        return [row for row in rows
                if row["llm_metrics"]["nDCG@10"] > row["bm25_metrics"]["nDCG@10"]]
    if args.only_hurt:
        return [row for row in rows
                if row["llm_metrics"]["nDCG@10"] < row["bm25_metrics"]["nDCG@10"]]
    return rows


def main() -> int:
    args = parse_args()

    run_dir = Path(args.results_root) / args.run_name
    rows = load_rows(run_dir / "rankings.jsonl")
    chosen = select(rows, args)

    summary_path = run_dir / "summary.md"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))

    print("=" * 78)
    print(f"RAW OUTPUTS  ({len(chosen)} of {len(rows)} queries match; "
          f"showing {min(args.num, len(chosen))})")
    print("=" * 78)

    for row in chosen[: args.num]:
        parse = row["parse"]
        bm25_ndcg = row["bm25_metrics"]["nDCG@10"]
        llm_ndcg = row["llm_metrics"]["nDCG@10"]

        print(f"\nquery_id {row['query_id']}  |  {row['n_candidates']} candidates, "
              f"{row['n_gold_in_window']} gold in window, "
              f"{row['n_prompt_tokens']} prompt tokens")
        print(f"  query: {row['query_text'][:100]}")
        print(f"  parse: strategy={parse['strategy']} clean={parse['is_clean']} "
              f"ids={parse['n_identifiers_parsed']} dup={parse['n_duplicates']} "
              f"oor={parse['n_out_of_range']} missing={parse['n_missing']}")
        if parse["notes"]:
            for note in parse["notes"]:
                print(f"         - {note}")
        print(f"  nDCG@10: BM25 {bm25_ndcg:.4f} -> LLM {llm_ndcg:.4f} "
              f"({llm_ndcg - bm25_ndcg:+.4f})")
        print(f"  top-10 now comes from BM25 ranks: "
              f"{row['llm_rank_of_bm25_rank'][:10]}")
        print(f"  raw output:")
        raw = row["raw_output"]
        shown = raw if args.chars == 0 else raw[: args.chars]
        print(f"    {shown!r}")
        if args.chars and len(raw) > args.chars:
            print(f"    ... [{len(raw) - args.chars} more characters]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
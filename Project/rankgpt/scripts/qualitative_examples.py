#!/usr/bin/env python
"""Pull the biggest wins and losses from a run, for the presentation slide.

    PYTHONPATH=src python scripts/qualitative_examples.py \
        --raw results/raw/llama31-8b-top20.jsonl \
        --data data/scifact/reranking_data.jsonl --top 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rankgpt.data import iter_reranking_data


def first_relevant_rank(doc_ids, relevant):
    for i, d in enumerate(doc_ids, start=1):
        if d in relevant:
            return i
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--top", type=int, default=3)
    args = p.parse_args()

    meta = {}
    for obj in iter_reranking_data(args.data):
        qid = str(obj["query_id"])
        meta[qid] = {
            "query_text": obj["query_text"],
            "relevant": {str(d["doc_id"]) for d in obj.get("relevant_docs", [])
                         if int(d.get("relevance", 0)) > 0},
            "titles": {str(d["doc_id"]): (d.get("title") or "")[:90]
                       for d in obj.get("retrieved_docs", [])},
        }

    rows = []
    with open(args.raw, encoding="utf-8") as f:
        for line in f:
            res = json.loads(line)
            qid = res["query_id"]
            m = meta.get(qid)
            if not m or not m["relevant"]:
                continue
            before = first_relevant_rank(res["input_doc_ids"], m["relevant"])
            after = first_relevant_rank(res["ranked_doc_ids"], m["relevant"])
            if before is None or after is None:
                continue
            rows.append({"qid": qid, "before": before, "after": after,
                         "delta": before - after, "res": res, "meta": m})

    rows.sort(key=lambda r: -r["delta"])

    def show(title, items):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        for r in items:
            m, res = r["meta"], r["res"]
            print(f"\nquery {r['qid']}: {m['query_text'][:100]}")
            print(f"  first relevant doc: BM25 rank {r['before']} -> LLM rank {r['after']}")
            print(f"  raw LLM output: {res['windows'][0]['raw_output'][:140]!r}")
            for lbl, ids in (("BM25 top-3", res["input_doc_ids"][:3]),
                             ("LLM  top-3", res["ranked_doc_ids"][:3])):
                print(f"  {lbl}:")
                for d in ids:
                    mark = " *" if d in m["relevant"] else "  "
                    print(f"   {mark} {d}  {m['titles'].get(d, '')}")

    show(f"TOP {args.top} IMPROVEMENTS", rows[: args.top])
    show(f"TOP {args.top} REGRESSIONS", rows[-args.top:][::-1])
    print("\n(* marks a relevant document)")


if __name__ == "__main__":
    main()

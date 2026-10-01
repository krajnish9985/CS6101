#!/usr/bin/env python
"""Inspect prompts and a couple of real model outputs before any big run.

Login node (no GPU), prompt inspection only:
    PYTHONPATH=src python scripts/smoke_test.py --config configs/llama31_8b_top20.yaml --show-prompt

Interactive GPU session, 2 real queries:
    PYTHONPATH=src python scripts/smoke_test.py --config configs/llama31_8b_top20.yaml -n 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rankgpt.data import load_reranking_data
from rankgpt.prompts import build_messages
from rankgpt.permutation import parse_permutation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("-n", type=int, default=2)
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--no-model", action="store_true", help="prompts only, no GPU")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    mcfg, pcfg, rcfg, gcfg = cfg["model"], cfg["prompt"], cfg["rerank"], cfg["generation"]

    records = load_reranking_data(cfg["data_path"], top_k=rcfg["top_k"], limit=args.n)
    print(f"loaded {len(records)} queries, {records[0].num_candidates} candidates each\n")

    messages = [
        build_messages(
            style=pcfg["style"],
            query=r.query_text,
            candidates=r.candidates,
            max_words=pcfg.get("max_passage_words", 120),
            include_title=pcfg.get("include_title", True),
        )
        for r in records
    ]

    if args.show_prompt:
        print("=" * 70)
        print("PROMPT FOR QUERY", records[0].query_id)
        print("=" * 70)
        for m in messages[0]:
            print(f"\n--- {m['role']} ---\n{m['content']}")
        print("=" * 70)

    if args.no_model:
        approx = len(" ".join(m["content"] for m in messages[0]).split())
        print(f"\napprox prompt length: {approx} words "
              f"(~{int(approx * 1.4)} tokens; max_model_len is "
              f"{mcfg.get('max_model_len', 8192)})")
        return

    from rankgpt.llm import VLLMRanker

    ranker = VLLMRanker(
        model_path=mcfg["name_or_path"],
        max_model_len=mcfg.get("max_model_len", 8192),
        tensor_parallel_size=mcfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=mcfg.get("gpu_memory_utilization", 0.90),
        dtype=mcfg.get("dtype", "bfloat16"),
    )

    for i, m in enumerate(messages):
        print(f"prompt {i} tokens: {ranker.count_tokens(m)}")

    outputs = ranker.generate(
        messages,
        temperature=gcfg.get("temperature", 0.0),
        max_tokens=gcfg.get("max_tokens", 256),
    )

    for rec, raw in zip(records, outputs):
        parse = parse_permutation(raw, rec.num_candidates)
        print("\n" + "=" * 70)
        print(f"query {rec.query_id}: {rec.query_text[:90]}")
        print(f"raw output   : {raw!r}")
        print(f"parsed ids   : {parse.raw_ids}")
        print(f"clean?       : {parse.is_clean}  "
              f"(dupes={parse.num_duplicates}, missing={parse.num_missing}, "
              f"oor={parse.num_out_of_range}, fallback={parse.used_fallback_regex})")
        gold = [c.doc_id for c in rec.candidates if c.relevance > 0]
        new_top5 = [rec.candidates[j].doc_id for j in parse.order[:5]]
        print(f"gold docs    : {gold}")
        print(f"BM25 top-5   : {[c.doc_id for c in rec.candidates[:5]]}")
        print(f"LLM top-5    : {new_top5}")


if __name__ == "__main__":
    main()

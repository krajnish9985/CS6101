#!/usr/bin/env python
"""Run LLM permutation-generation reranking from a YAML config.

Usage:
    PYTHONPATH=src python scripts/run_rerank.py --config configs/llama31_8b_top20.yaml
    PYTHONPATH=src python scripts/run_rerank.py --config ... --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rankgpt.data import load_full_candidate_lists, load_reranking_data
from rankgpt.permutation import aggregate_parse_stats
from rankgpt.rerank import rerank_single_window, rerank_sliding_window
from rankgpt.runfile import build_run, evaluate_run, load_qrels, write_trec_run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--limit", type=int, default=None, help="override: only N queries")
    p.add_argument("--run-name", default=None, help="override run_name")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build prompts with the offline stub, never load the model",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    run_name = args.run_name or cfg["run_name"]
    limit = args.limit if args.limit is not None else cfg.get("limit_queries")

    mcfg, pcfg, rcfg, gcfg = cfg["model"], cfg["prompt"], cfg["rerank"], cfg["generation"]
    out_dir = Path(cfg.get("output_dir", "results"))
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    print(f"=== {run_name} ===")
    print(f"config: {args.config}")

    records = load_reranking_data(
        cfg["data_path"], top_k=rcfg["top_k"], limit=limit
    )
    print(f"queries: {len(records)}  top_k: {rcfg['top_k']}")

    if args.dry_run:
        from rankgpt.llm import EchoRanker

        ranker = EchoRanker(num_candidates_hint=rcfg["top_k"])
    else:
        from rankgpt.llm import VLLMRanker

        ranker = VLLMRanker(
            model_path=mcfg["name_or_path"],
            max_model_len=mcfg.get("max_model_len", 8192),
            tensor_parallel_size=mcfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=mcfg.get("gpu_memory_utilization", 0.90),
            dtype=mcfg.get("dtype", "bfloat16"),
            seed=cfg.get("seed", 42),
        )

    common = dict(
        prompt_style=pcfg["style"],
        max_passage_words=pcfg.get("max_passage_words", 120),
        include_title=pcfg.get("include_title", True),
        temperature=gcfg.get("temperature", 0.0),
        max_tokens=gcfg.get("max_tokens", 256),
    )

    start = time.time()
    if rcfg.get("mode", "single_window") == "sliding_window":
        results = rerank_sliding_window(
            ranker,
            records,
            window_size=rcfg.get("window_size", 20),
            step=rcfg.get("step", 10),
            **common,
        )
    else:
        results = rerank_single_window(
            ranker,
            records,
            shuffle_candidates=rcfg.get("shuffle_candidates", False),
            seed=cfg.get("seed", 42),
            **common,
        )
    elapsed = time.time() - start
    print(f"reranking finished in {elapsed:.1f}s "
          f"({elapsed / max(len(records), 1):.2f}s per query)")

    # --- raw outputs (for debugging and the qualitative slide) ---
    raw_path = out_dir / "raw" / f"{run_name}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
    print(f"raw outputs -> {raw_path}")

    # --- parse quality ---
    all_parses = [w.parse for res in results for w in res.windows]
    parse_stats = aggregate_parse_stats(all_parses)
    print("\nparse quality:")
    for k, v in parse_stats.items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    prompt_tokens = [
        w.prompt_tokens for res in results for w in res.windows if w.prompt_tokens
    ]
    if prompt_tokens:
        print(f"  prompt tokens  min/mean/max   "
              f"{min(prompt_tokens)} / {sum(prompt_tokens) // len(prompt_tokens)} / {max(prompt_tokens)}")

    # --- TREC run ---
    full = load_full_candidate_lists(cfg["data_path"]) if cfg.get("pad_to_full_list", True) else None
    run = build_run(results, full_candidates=full)
    run_path = out_dir / "runs" / f"{run_name}.trec"
    write_trec_run(run, run_path, tag=run_name)
    print(f"\nrun file -> {run_path}")

    # --- metrics ---
    metrics = {"run_name": run_name, "elapsed_sec": round(elapsed, 1)}
    qrels_path = cfg.get("qrels_path")
    if qrels_path and Path(qrels_path).exists() and limit is None:
        try:
            qrels = load_qrels(qrels_path)
            qrels = {q: v for q, v in qrels.items() if q in run}
            scores = evaluate_run(run, qrels)
            metrics.update(scores)
            print("\nmetrics:")
            for k, v in scores.items():
                print(f"  {k:<12} {v:.4f}" if isinstance(v, float) else f"  {k:<12} {v}")
        except ImportError:
            print("\npytrec_eval not installed; skipping evaluation "
                  "(hand the run file to Member 2 instead)")
    metrics.update({f"parse_{k}": v for k, v in parse_stats.items()})

    metrics_path = out_dir / "metrics" / f"{run_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()

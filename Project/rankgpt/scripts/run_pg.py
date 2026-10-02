#!/usr/bin/env python
"""Run permutation generation over a whole dataset and write results.

Usage
-----
    python scripts/run_pg.py configs/llama8b_top20_paper.yaml
    python scripts/run_pg.py configs/llama8b_top20_paper.yaml --limit 5

Outputs, under ``results/runs/<run_name>/``:

    config.json     resolved configuration, git commit, host, GPU, timing
    rankings.jsonl  one line per query: both orders, parse stats, metrics,
                    and the model's raw output
    summary.json    aggregate metrics and parse quality, machine-readable
    summary.md      the same, as a table to paste into the report

Prompts are rendered and length-checked for every query before the model
is loaded, so a context-length mistake fails in seconds rather than after
the weights are in GPU memory.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rankgpt_pg.config import RunConfig, git_commit, git_is_dirty, load_run_config  # noqa: E402
from rankgpt_pg.data import (  # noqa: E402
    describe_dataset,
    gold_grades,
    load_records,
    take_candidates,
)
from rankgpt_pg.llm import PermutationGenerator, budget_for_tokens  # noqa: E402
from rankgpt_pg.metrics import (  # noqa: E402
    evaluate_ranking,
    evaluate_ranking_standard,
    mean_metrics,
)
from rankgpt_pg.permutation import apply_permutation, parse_permutation  # noqa: E402
from rankgpt_pg.prompts import get_prompt_builder  # noqa: E402

METRIC_KEYS = ("nDCG@1", "nDCG@10", "MRR@10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="path to a run YAML under configs/")

    # Overrides. Left as None unless passed, so the YAML wins by default.
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N queries (rehearse a run)")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--prompt", default=None, choices=["paper", "compact"])
    parser.add_argument("--run-name", default=None,
                        help="override the output directory name")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="render and measure every prompt, then stop "
                             "without loading the model")
    return parser.parse_args()


def environment_info() -> Dict[str, Any]:
    """Record where a run happened, for the log and for reproducibility."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as error:
        info["torch"] = f"unavailable: {error}"

    try:
        import vllm

        info["vllm"] = vllm.__version__
    except Exception as error:
        info["vllm"] = f"unavailable: {error}"

    return info


def build_tasks(config: RunConfig, generator: PermutationGenerator) -> List[Dict[str, Any]]:
    """Render one prompt per query and measure it, before any GPU work.

    A query whose prompt cannot fit in the context window is flagged here
    and skipped at generation time: it keeps its BM25 order and is counted
    as a context overflow rather than silently crashing the run.
    """
    records = load_records(config.data, limit=config.limit)
    builder = get_prompt_builder(config.prompt)

    tasks: List[Dict[str, Any]] = []
    for record in records:
        docs = take_candidates(record, config.depth)
        messages = builder(record["query_text"], docs, config.max_passage_words)
        prompt = generator.render(messages)

        n_prompt_tokens = generator.count_tokens(prompt)
        budget = budget_for_tokens(max(len(docs), 1))
        fits = n_prompt_tokens + budget <= config.model.max_model_len

        tasks.append({
            "record": record,
            "docs": docs,
            "prompt": prompt,
            "n_prompt_tokens": n_prompt_tokens,
            "max_new_tokens": budget,
            "fits": fits,
            # Stage-1 retrieval returned nothing for this query. There is
            # no permutation to generate, so it is never sent to the model
            # and scores zero for both BM25 and the re-ranker.
            "is_empty": len(docs) == 0,
        })

    return tasks


def summarise_prompts(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [task["n_prompt_tokens"] for task in tasks]
    return {
        "n_tasks": len(tasks),
        "prompt_tokens_min": min(lengths) if lengths else 0,
        "prompt_tokens_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "prompt_tokens_max": max(lengths) if lengths else 0,
        "n_context_overflow": sum(1 for task in tasks if not task["fits"]),
        "n_empty_candidate_sets": sum(1 for task in tasks if task["is_empty"]),
    }


def run_generation(
    config: RunConfig,
    generator: PermutationGenerator,
    tasks: List[Dict[str, Any]],
) -> List[str]:
    """Generate in chunks so the Slurm log shows progress as it goes.

    Overflowing prompts are never sent to the model; they receive an
    empty string, which the parser turns into the BM25 order.
    """
    outputs: List[str] = [""] * len(tasks)

    runnable = [
        index for index, task in enumerate(tasks)
        if task["fits"] and not task["is_empty"]
    ]    
    if not runnable:
        print("  no runnable prompts; nothing to generate", flush=True)
        return outputs

    # One budget for the whole run: within a fixed depth every task asks
    # for the same number of tokens, so this is the max over the batch.
    budget = max(tasks[index]["max_new_tokens"] for index in runnable)

    total_chunks = (len(runnable) + config.chunk_size - 1) // config.chunk_size
    start = time.time()

    for chunk_number, offset in enumerate(range(0, len(runnable), config.chunk_size), start=1):
        indices = runnable[offset: offset + config.chunk_size]
        prompts = [tasks[index]["prompt"] for index in indices]

        chunk_start = time.time()
        completions = generator.generate(prompts, max_new_tokens=budget)
        chunk_elapsed = time.time() - chunk_start

        for index, completion in zip(indices, completions):
            outputs[index] = completion

        done = offset + len(indices)
        rate = done / (time.time() - start)
        print(
            f"  chunk {chunk_number}/{total_chunks}: {len(indices)} queries "
            f"in {chunk_elapsed:.1f}s | {done}/{len(runnable)} done "
            f"| {rate:.2f} queries/s",
            flush=True,
        )

    return outputs


def score_tasks(tasks: List[Dict[str, Any]], outputs: List[str]) -> List[Dict[str, Any]]:
    """Parse each completion, apply it, and evaluate both orders."""
    rows: List[Dict[str, Any]] = []

    for task, raw in zip(tasks, outputs):
        record = task["record"]
        docs = task["docs"]

        grades = gold_grades(record)

        if task["is_empty"]:
            # No candidates to rank. Record the query so n_queries matches
            # the dataset and stage-1 recall failures are not hidden.
            zero = {key: 0.0 for key in METRIC_KEYS}
            rows.append({
                "query_id": str(record["query_id"]),
                "query_text": record["query_text"],
                "n_candidates": 0,
                "n_gold_in_window": 0,
                "n_prompt_tokens": task["n_prompt_tokens"],
                "context_overflow": False,
                "empty_candidate_set": True,
                "bm25_order": [],
                "llm_order": [],
                "llm_rank_of_bm25_rank": [],
                "parse": {
                    "strategy": "skipped", "n_identifiers_parsed": 0,
                    "n_duplicates": 0, "n_out_of_range": 0, "n_missing": 0,
                    "is_clean": False, "is_total_failure": False,
                    "notes": ["no candidates retrieved; not sent to the model"],
                    "identifier_inflation": 0.0,
                },
                "n_gold_judged": len(grades),
                "bm25_metrics": dict(zero), "llm_metrics": dict(zero),
                "bm25_metrics_standard": dict(zero),
                "llm_metrics_standard": dict(zero),
                "raw_output": "",
            })
            continue

        result = parse_permutation(raw, len(docs))
        reranked = apply_permutation(docs, result.order)

        bm25_metrics = evaluate_ranking(docs)
        llm_metrics = evaluate_ranking(reranked)
        bm25_std = evaluate_ranking_standard(docs, grades)
        llm_std = evaluate_ranking_standard(reranked, grades)

        rows.append({
            "query_id": str(record["query_id"]),
            "query_text": record["query_text"],
            "n_candidates": len(docs),
            "n_gold_in_window": sum(1 for d in docs if int(d.get("relevance", 0)) > 0),
            "n_prompt_tokens": task["n_prompt_tokens"],
            "context_overflow": not task["fits"],
            "empty_candidate_set": False,
            "bm25_order": [str(d["doc_id"]) for d in docs],
            "llm_order": [str(d["doc_id"]) for d in reranked],
            "llm_rank_of_bm25_rank": [d["rank"] for d in reranked],
            "parse": result.stats(),
            "n_gold_judged": len(grades),
            "bm25_metrics": bm25_metrics,
            "llm_metrics": llm_metrics,
            "bm25_metrics_standard": bm25_std,
            "llm_metrics_standard": llm_std,
            "raw_output": raw,
        })

    return rows


def aggregate(rows: List[Dict[str, Any]], depth: int) -> Dict[str, Any]:
    """Mean metrics plus the parse-quality table."""
    n = len(rows)
    if n == 0:
        return {"n_queries": 0}

    def share(predicate) -> float:
        return round(100.0 * sum(1 for row in rows if predicate(row)) / n, 2)

    bm25 = mean_metrics([row["bm25_metrics"] for row in rows])
    llm = mean_metrics([row["llm_metrics"] for row in rows])

    scored = [row for row in rows if row["n_gold_in_window"] > 0]
    bm25_scored = mean_metrics([row["bm25_metrics"] for row in scored])
    llm_scored = mean_metrics([row["llm_metrics"] for row in scored])

    bm25_std = mean_metrics([row["bm25_metrics_standard"] for row in rows])
    llm_std = mean_metrics([row["llm_metrics_standard"] for row in rows])

    return {
        "n_queries": n,
        "n_queries_with_gold_in_window": len(scored),
        "n_empty_candidate_sets": sum(1 for row in rows if row["empty_candidate_set"]),
        "n_queries_short_of_depth": sum(
            1 for row in rows if 0 < row["n_candidates"] < depth
        ),
        "mean_gold_judged_per_query": round(
            sum(row["n_gold_judged"] for row in rows) / n, 2
        ),
        "bm25": bm25,
        "llm": llm,
        "delta": {key: round(llm[key] - bm25[key], 4) for key in METRIC_KEYS},
        "bm25_standard": bm25_std,
        "llm_standard": llm_std,
        "delta_standard": {
            key: round(llm_std[key] - bm25_std[key], 4) for key in METRIC_KEYS
        },
        "bm25_gold_only": bm25_scored,
        "llm_gold_only": llm_scored,
        "parse_quality": {
            "pct_clean": share(lambda r: r["parse"]["is_clean"]),
            "pct_bracket_strategy": share(lambda r: r["parse"]["strategy"] == "bracket"),
            "pct_bare_int_strategy": share(lambda r: r["parse"]["strategy"] == "bare_int"),
            "pct_total_failure": share(lambda r: r["parse"]["is_total_failure"]),
            "pct_with_duplicates": share(lambda r: r["parse"]["n_duplicates"] > 0),
            "pct_with_out_of_range": share(lambda r: r["parse"]["n_out_of_range"] > 0),
            "pct_with_missing": share(lambda r: r["parse"]["n_missing"] > 0),
            "mean_n_missing": round(
                sum(r["parse"]["n_missing"] for r in rows) / n, 2
            ),
            "pct_context_overflow": share(lambda r: r["context_overflow"]),
            "pct_empty_candidate_set": share(lambda r: r["empty_candidate_set"]),
            "pct_degenerate_output": share(
                lambda r: r["parse"]["identifier_inflation"] >= 1.5
            ),
            "mean_identifier_inflation": round(
                sum(r["parse"]["identifier_inflation"] for r in rows) / n, 2
            ),
        },
    }


def write_summary_markdown(path: Path, config: RunConfig, summary: Dict[str, Any],
                           env: Dict[str, Any], timing: Dict[str, Any]) -> None:
    lines = [
        f"# {config.run_name}",
        "",
        f"- dataset: `{config.dataset}` ({summary['n_queries']} queries, depth {config.depth})",
        f"- model: `{config.model.model_label}`",
        f"- prompt: `{config.prompt}`, passages truncated to {config.max_passage_words} words",
        f"- decoding: temperature {config.model.temperature}, seed {config.model.seed}",
        f"- gpu: {env.get('gpu_name', 'n/a')} x{env.get('gpu_count', 0)}",
        f"- commit: `{env['git_commit'][:12]}`"
        + (" (dirty tree)" if env["git_dirty"] else ""),
        f"- wall clock: {timing['total_seconds']:.1f}s "
        f"({timing['seconds_per_query']:.2f}s/query)",
        "",
        "## Ranking metrics (all queries)",
        "",
        "| metric | BM25 | LLM | delta |",
        "| --- | --- | --- | --- |",
    ]

    for key in METRIC_KEYS:
        lines.append(
            f"| {key} | {summary['bm25'][key]:.4f} | {summary['llm'][key]:.4f} "
            f"| {summary['delta'][key]:+.4f} |"
        )

    lines += [
        "",
        "## Ranking metrics, standard BEIR nDCG (ideal DCG from full qrels)",
        "",
        f"- mean judged relevant documents per query: "
        f"{summary['mean_gold_judged_per_query']}",
        "",
        "| metric | BM25 | LLM | delta |",
        "| --- | --- | --- | --- |",
    ]

    for key in METRIC_KEYS:
        lines.append(
            f"| {key} | {summary['bm25_standard'][key]:.4f} "
            f"| {summary['llm_standard'][key]:.4f} "
            f"| {summary['delta_standard'][key]:+.4f} |"
        )


    lines += [
        "",
        f"## Ranking metrics (the {summary['n_queries_with_gold_in_window']} queries "
        "with a relevant document in the window)",
        "",
        "| metric | BM25 | LLM | delta |",
        "| --- | --- | --- | --- |",
    ]

    for key in METRIC_KEYS:
        bm25_value = summary["bm25_gold_only"][key]
        llm_value = summary["llm_gold_only"][key]
        lines.append(
            f"| {key} | {bm25_value:.4f} | {llm_value:.4f} | {llm_value - bm25_value:+.4f} |"
        )

    lines += ["", "## Parse quality", "", "| statistic | value |", "| --- | --- |"]
    for key, value in summary["parse_quality"].items():
        lines.append(f"| {key} | {value} |")

    if config.notes:
        lines += ["", "## Notes", "", config.notes]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    overrides = {
        "limit": args.limit,
        "depth": args.depth,
        "prompt": args.prompt,
        "run_name": args.run_name,
        "chunk_size": args.chunk_size,
        "model_path": args.model_path,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
    }
    config = load_run_config(args.config, overrides)

    print("=" * 78)
    print(f"RUN {config.run_name}")
    print("=" * 78)
    print(json.dumps(config.to_dict(), indent=2))

    env = environment_info()
    print("\nENVIRONMENT")
    print(json.dumps(env, indent=2))

    generator = PermutationGenerator(config.model)

    print("\nRENDERING PROMPTS")
    build_start = time.time()
    tasks = build_tasks(config, generator)
    prompt_summary = summarise_prompts(tasks)
    dataset_summary = describe_dataset([task["record"] for task in tasks], config.depth)
    print(json.dumps({"dataset": dataset_summary, "prompts": prompt_summary}, indent=2))
    print(f"  rendered in {time.time() - build_start:.1f}s")

    if prompt_summary["n_context_overflow"]:
        print(
            f"  WARNING: {prompt_summary['n_context_overflow']} prompt(s) exceed "
            f"max_model_len={config.model.max_model_len}; those queries keep "
            f"their BM25 order"
        )

    if prompt_summary["n_empty_candidate_sets"]:
        print(
            f"  NOTE: {prompt_summary['n_empty_candidate_sets']} query/queries have "
            f"no BM25 candidates at all; they are not sent to the model and "
            f"score 0 for both BM25 and the re-ranker (a stage-1 recall failure)"
        )

    if args.dry_run:
        print("\nDRY RUN: prompts rendered and measured; model not loaded.")
        return 0

    print("\nLOADING MODEL")
    load_start = time.time()
    generator.load()
    load_seconds = time.time() - load_start
    print(f"  loaded in {load_seconds:.1f}s")

    print("\nGENERATING")
    gen_start = time.time()
    outputs = run_generation(config, generator, tasks)
    gen_seconds = time.time() - gen_start

    print("\nSCORING")
    rows = score_tasks(tasks, outputs)
    summary = aggregate(rows, config.depth)

    timing = {
        "load_seconds": round(load_seconds, 1),
        "generate_seconds": round(gen_seconds, 1),
        "total_seconds": round(load_seconds + gen_seconds, 1),
        "seconds_per_query": round(gen_seconds / max(len(rows), 1), 3),
    }

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config.json").write_text(
        json.dumps(
            {"config": config.to_dict(), "environment": env, "timing": timing},
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / "rankings.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps({"run_name": config.run_name, **summary, "timing": timing}, indent=2),
        encoding="utf-8",
    )

    write_summary_markdown(output_dir / "summary.md", config, summary, env, timing)

    print("\n" + "=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  {'metric':<10} {'BM25':>10} {'LLM':>10} {'delta':>10}")
    print("  " + "-" * 44)
    for key in METRIC_KEYS:
        print(
            f"  {key:<10} {summary['bm25'][key]:>10.4f} "
            f"{summary['llm'][key]:>10.4f} {summary['delta'][key]:>+10.4f}"
        )

    print(f"\n  standard BEIR nDCG (ideal from full qrels):")
    print(f"  {'metric':<10} {'BM25':>10} {'LLM':>10} {'delta':>10}")
    print("  " + "-" * 44)
    for key in METRIC_KEYS:
        print(
            f"  {key:<10} {summary['bm25_standard'][key]:>10.4f} "
            f"{summary['llm_standard'][key]:>10.4f} "
            f"{summary['delta_standard'][key]:>+10.4f}"
        )

    print("\n  parse quality:")
    for key, value in summary["parse_quality"].items():
        print(f"    {key}: {value}")

    print(f"\n  wrote {output_dir}/")
    for name in ("config.json", "rankings.jsonl", "summary.json", "summary.md"):
        print(f"    {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
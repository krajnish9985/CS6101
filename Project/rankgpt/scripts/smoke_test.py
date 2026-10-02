#!/usr/bin/env python
"""Run permutation generation on ONE query and show every intermediate step.

This is the debugging tool for the whole pipeline. It prints the rendered
prompt, its exact token count, the model's raw output, what the parser
made of it, how the ranking moved, and the metric change for that single
query. Nothing is hidden, so when something looks wrong it is obvious
which stage produced it.

Run it with --dry-run on a login node to check prompts and token counts
without a GPU, then without --dry-run inside an interactive allocation.

Examples
--------
    # Login node: no GPU, no model load.
    python scripts/smoke_test.py --dry-run

    # Interactive GPU session: real generation.
    python scripts/smoke_test.py \
        --model models/Llama-3.1-8B-Instruct \
        --pick-query-with-gold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable when the script is run directly as
# `python scripts/smoke_test.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rankgpt_pg.data import (  # noqa: E402
    describe_dataset,
    find_record,
    gold_doc_ids,
    load_records,
    take_candidates,
)
from rankgpt_pg.llm import LLMConfig, PermutationGenerator, budget_for_tokens  # noqa: E402
from rankgpt_pg.metrics import evaluate_ranking  # noqa: E402
from rankgpt_pg.permutation import apply_permutation, parse_permutation  # noqa: E402
from rankgpt_pg.prompts import MAX_PASSAGE_WORDS, get_prompt_builder  # noqa: E402

DEFAULT_DATA = "data/scifact/scifact_reranking_data.jsonl"


def rule(title: str = "", width: int = 78) -> None:
    if title:
        print("\n" + "=" * width)
        print(title)
        print("=" * width)
    else:
        print("-" * width)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data", default=DEFAULT_DATA, help="Member 1's candidate JSONL")
    parser.add_argument("--model", default="models/Llama-3.1-8B-Instruct", help="local model directory")
    parser.add_argument("--depth", type=int, default=20, help="number of BM25 candidates to re-rank")
    parser.add_argument("--prompt", default="paper", choices=["paper", "compact"])
    parser.add_argument("--max-passage-words", type=int, default=MAX_PASSAGE_WORDS)

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--index", type=int, default=0, help="0-based line number in the JSONL")
    selection.add_argument("--qid", help="select by query_id instead")
    selection.add_argument(
        "--pick-query-with-gold",
        action="store_true",
        help="pick the first query whose top-k contains a relevant document "
             "(otherwise every metric is 0 and the test shows nothing)",
    )

    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="skip CUDA graph capture: slower generation, much faster startup, "
             "which is what you want while debugging",
    )
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--show-prompt-chars", type=int, default=1500,
                        help="how much of the rendered prompt to print; 0 prints all of it")
    parser.add_argument("--dry-run", action="store_true",
                        help="render the prompt and count tokens, but do not load the model")

    return parser.parse_args()


def select_record(records, args, depth):
    if args.qid:
        return find_record(records, args.qid)

    if args.pick_query_with_gold:
        for record in records:
            if any(int(d.get("relevance", 0)) > 0 for d in record["retrieved_docs"][:depth]):
                return record
        raise SystemExit(f"no query has a relevant document in its top-{depth}")

    if args.index >= len(records):
        raise SystemExit(f"--index {args.index} out of range ({len(records)} records)")
    return records[args.index]


def print_candidates(docs, gold_ids) -> None:
    print(f"{'BM25':>5}  {'doc_id':<12} {'score':>8}  {'rel':>3}  title")
    rule()
    for doc in docs:
        marker = "*" if str(doc["doc_id"]) in gold_ids else " "
        title = (doc.get("title") or "")[:44]
        print(
            f"{doc['rank']:>5}  {str(doc['doc_id']):<12} "
            f"{doc['bm25_score']:>8.3f}  {int(doc.get('relevance', 0)):>3}{marker} {title}"
        )


def print_movement(before, after, gold_ids) -> None:
    print(f"{'new':>4} {'was':>4} {'move':>6}  {'doc_id':<12} {'rel':>3}  title")
    rule()
    for doc in after:
        delta = doc["rank"] - doc["llm_rank"]
        arrow = f"+{delta}" if delta > 0 else (str(delta) if delta < 0 else "0")
        marker = "*" if str(doc["doc_id"]) in gold_ids else " "
        title = (doc.get("title") or "")[:40]
        print(
            f"{doc['llm_rank']:>4} {doc['rank']:>4} {arrow:>6}  "
            f"{str(doc['doc_id']):<12} {int(doc.get('relevance', 0)):>3}{marker} {title}"
        )


def main() -> int:
    args = parse_args()

    rule("DATASET")
    records = load_records(args.data)
    summary = describe_dataset(records, args.depth)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    record = select_record(records, args, args.depth)
    docs = take_candidates(record, args.depth)
    gold_ids = gold_doc_ids(record)

    rule("QUERY")
    print(f"  query_id : {record['query_id']}")
    print(f"  query    : {record['query_text']}")
    print(f"  candidates: {len(docs)} (requested depth {args.depth})")
    print(f"  gold doc_ids: {sorted(gold_ids) or 'none'}")
    n_gold_in_window = sum(1 for d in docs if int(d.get('relevance', 0)) > 0)
    print(f"  gold in this window: {n_gold_in_window}")
    if n_gold_in_window == 0:
        print("  NOTE: no relevant document in this window, so every metric "
              "will be 0.0 regardless of the ranking.")
        print("        Re-run with --pick-query-with-gold to see a real change.")

    rule("BM25 CANDIDATES  (* = relevant)")
    print_candidates(docs, gold_ids)

    config = LLMConfig(
        model_path=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        max_new_tokens=budget_for_tokens(len(docs)),
    )
    generator = PermutationGenerator(config)

    builder = get_prompt_builder(args.prompt)
    messages = builder(record["query_text"], docs, args.max_passage_words)
    prompt = generator.render(messages)

    rule("PROMPT")
    print(f"  variant      : {args.prompt}")
    print(f"  chat turns   : {len(messages)}")
    print(f"  prompt tokens: {generator.count_tokens(prompt)}")
    print(f"  gen budget   : {config.max_new_tokens}")
    print(f"  max_model_len: {config.max_model_len}")
    print(f"  fits context : {generator.fits_in_context(prompt)}")
    rule()
    if args.show_prompt_chars == 0:
        print(prompt)
    else:
        print(prompt[: args.show_prompt_chars])
        if len(prompt) > args.show_prompt_chars:
            print(f"\n... [{len(prompt) - args.show_prompt_chars} more characters] ...\n")
            print(prompt[-600:])

    if args.dry_run:
        rule("DRY RUN")
        print("  Prompt rendered and measured. Model not loaded.")
        print("  Drop --dry-run inside a GPU allocation to generate.")
        return 0

    if not generator.fits_in_context(prompt):
        raise SystemExit(
            "prompt does not fit in max_model_len; raise --max-model-len "
            "or lower --max-passage-words"
        )

    rule("LOADING MODEL")
    print(f"  {args.model}")
    print("  (first load takes a minute or two)")
    generator.load()

    rule("GENERATING")
    raw = generator.generate_one(prompt)
    print(f"  raw output ({len(raw)} chars):")
    rule()
    print(repr(raw))

    rule("PARSE")
    result = parse_permutation(raw, len(docs))
    for key, value in result.stats().items():
        print(f"  {key}: {value}")
    print(f"  order (0-indexed): {result.order}")

    reranked = apply_permutation(docs, result.order)

    rule("RANK MOVEMENT  (* = relevant, move = positions gained)")
    print_movement(docs, reranked, gold_ids)

    rule("METRICS FOR THIS QUERY")
    bm25_metrics = evaluate_ranking(docs)
    llm_metrics = evaluate_ranking(reranked)
    print(f"  {'metric':<10} {'BM25':>10} {'LLM':>10} {'delta':>10}")
    rule()
    for key in ("nDCG@1", "nDCG@10", "MRR@10"):
        delta = llm_metrics[key] - bm25_metrics[key]
        print(f"  {key:<10} {bm25_metrics[key]:>10.4f} {llm_metrics[key]:>10.4f} {delta:>+10.4f}")

    rule("DONE")
    print("  If the parse is clean and the movement looks sensible, the "
          "pipeline is ready for a full run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
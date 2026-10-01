"""TREC run-file writing and evaluation.

The run file is the hand-off format to Member 2's evaluation code:
    query_id Q0 doc_id rank score tag
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .rerank import RerankResult


def build_run(
    results: Sequence[RerankResult],
    full_candidates: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """query_id -> ordered doc_ids.

    If `full_candidates` is given (the complete BM25 top-100 per query), any
    document that was NOT reranked is appended after the reranked block in its
    original BM25 order. This keeps Recall@100 and MAP directly comparable with
    the BM25 and cross-encoder baselines even when we only rerank the top 20.
    """
    run: Dict[str, List[str]] = {}
    for res in results:
        ordered = list(res.ranked_doc_ids)
        if full_candidates and res.query_id in full_candidates:
            seen = set(ordered)
            tail = [d for d in full_candidates[res.query_id] if d not in seen]
            ordered = ordered + tail
        run[res.query_id] = ordered
    return run


def write_trec_run(run: Dict[str, List[str]], path, tag: str = "rankgpt") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_ids in run.items():
            n = len(doc_ids)
            for rank, doc_id in enumerate(doc_ids, start=1):
                # Strictly decreasing score so any downstream sort-by-score
                # reproduces exactly this order.
                score = float(n - rank + 1)
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.4f} {tag}\n")


def read_trec_run(path) -> Dict[str, List[str]]:
    run: Dict[str, List[tuple]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, rank, score, _ = parts[:6]
            run.setdefault(qid, []).append((float(score), doc_id))
    return {q: [d for _, d in sorted(v, key=lambda x: -x[0])] for q, v in run.items()}


def load_qrels(path) -> Dict[str, Dict[str, int]]:
    """Read the qrels.tsv written by Member 1: qid \\t 0 \\t docid \\t rel."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, doc_id, rel = parts[:4]
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def evaluate_run(
    run: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    metrics: Sequence[str] = ("ndcg_cut.1", "ndcg_cut.10", "recall.100", "map"),
) -> Dict[str, float]:
    """Metrics matching Member 2's definitions, so numbers go in the same table."""
    import pytrec_eval

    run_scores = {
        qid: {doc_id: float(len(docs) - i) for i, doc_id in enumerate(docs)}
        for qid, docs in run.items()
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
    per_query = evaluator.evaluate(run_scores)
    n = len(per_query)
    if n == 0:
        return {}

    def mean(key):
        return sum(v[key] for v in per_query.values()) / n

    out = {
        "Queries": n,
        "nDCG@1": mean("ndcg_cut_1"),
        "nDCG@10": mean("ndcg_cut_10"),
        "Recall@100": mean("recall_100"),
        "MAP": mean("map"),
    }

    rr = []
    for qid, docs in run.items():
        relevant = {d for d, r in qrels.get(qid, {}).items() if r > 0}
        score = 0.0
        for rank, doc_id in enumerate(docs[:10], start=1):
            if doc_id in relevant:
                score = 1.0 / rank
                break
        rr.append(score)
    out["MRR@10"] = sum(rr) / len(rr) if rr else 0.0
    return out

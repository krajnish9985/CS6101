"""Ranking metrics, in two flavours.

**Pool-relative** (``evaluate_ranking``) mirrors Member 2's implementation
exactly: the ideal DCG is computed from the relevance labels present in
the candidate list. A gold document BM25 never retrieved therefore does
not penalise the score. These numbers are directly comparable to the
BM25 and cross-encoder baselines in Member 2's notebook.

**Standard** (``evaluate_ranking_standard``) is what BEIR and the RankGPT
paper report: the ideal DCG comes from the full qrels for the query, so
recall failures in stage 1 do count against the final score.

Both are reported for every run. The distinction barely matters on
SciFact (about 1.1 relevant documents per query) but it matters a great
deal on TREC-COVID (over 1300 judged documents per query): there, almost
every candidate in a top-20 window is relevant, so pool-relative nDCG@10
is close to 1.0 whatever the ordering, and differences between methods
are compressed into invisibility.

Gains are ``2**rel - 1``. SciFact labels are binary; NFCorpus,
TREC-COVID and DBPedia are graded, so a grade-2 document contributes a
gain of 3 and the choice of flavour changes the numbers materially.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

Doc = Dict[str, Any]
METRIC_KEYS = ("nDCG@1", "nDCG@10", "MRR@10")


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """Discounted cumulative gain over the first ``k`` positions."""
    total = 0.0
    for index, relevance in enumerate(relevances[:k]):
        rank = index + 1
        total += (2 ** relevance - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(
    relevances: Sequence[int],
    k: int,
    ideal_pool: Optional[Sequence[int]] = None,
) -> float:
    """nDCG@k.

    ``ideal_pool`` supplies the relevance grades used to build the ideal
    ranking. Left as None it defaults to ``relevances`` itself, which is
    the pool-relative behaviour. Pass the full qrels grades for the
    standard BEIR definition.
    """
    actual = dcg_at_k(relevances, k)
    pool = relevances if ideal_pool is None else ideal_pool
    ideal = dcg_at_k(sorted(pool, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def reciprocal_rank_at_k(relevances: Sequence[int], k: int) -> float:
    """1 / rank of the first relevant document within the first ``k``.

    MRR does not depend on the ideal ranking, so it is identical under
    both flavours.
    """
    for index, relevance in enumerate(relevances[:k]):
        if relevance > 0:
            return 1.0 / (index + 1)
    return 0.0


def _grades(ranking: List[Doc]) -> List[int]:
    return [int(doc.get("relevance", 0)) for doc in ranking]


def evaluate_ranking(ranking: List[Doc]) -> Dict[str, float]:
    """Pool-relative metrics. Matches Member 2's ``evaluate_ranking``."""
    relevances = _grades(ranking)
    return {
        "nDCG@1": ndcg_at_k(relevances, 1),
        "nDCG@10": ndcg_at_k(relevances, 10),
        "MRR@10": reciprocal_rank_at_k(relevances, 10),
    }


def evaluate_ranking_standard(
    ranking: List[Doc],
    gold_grades: Sequence[int],
) -> Dict[str, float]:
    """Standard BEIR metrics: ideal DCG from the full qrels.

    ``gold_grades`` is every positive relevance grade judged for the
    query, including documents stage-1 retrieval missed entirely.
    """
    relevances = _grades(ranking)

    # The ideal pool is the judged qrels grades. Every retrieved relevant
    # document is itself judged, so for self-consistent data the judged
    # list dominates the observed one position by position and nDCG <= 1
    # follows automatically. Taking the position-wise max keeps that
    # guarantee even if relevant_docs is incomplete relative to the
    # relevance labels on retrieved_docs, which would otherwise produce
    # an ideal DCG smaller than the actual and an nDCG above 1.
    observed = sorted(relevances, reverse=True)
    judged = sorted(gold_grades, reverse=True)
    pool = [
        max(
            observed[index] if index < len(observed) else 0,
            judged[index] if index < len(judged) else 0,
        )
        for index in range(max(len(observed), len(judged)))
    ]

    return {
        "nDCG@1": ndcg_at_k(relevances, 1, ideal_pool=pool),
        "nDCG@10": ndcg_at_k(relevances, 10, ideal_pool=pool),
        "MRR@10": reciprocal_rank_at_k(relevances, 10),
    }


def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Macro-average over queries, which is what the baselines report."""
    if not per_query:
        return {key: 0.0 for key in METRIC_KEYS}
    return {
        key: sum(row[key] for row in per_query) / len(per_query)
        for key in METRIC_KEYS
    }
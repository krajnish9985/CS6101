"""Parsing and application of LLM-generated permutations.

A listwise re-ranker only helps if the model's free text can be turned
back into a permutation reliably. Open 7B/8B models break the requested
"[2] > [1] > [3]" format often enough that the recovery policy here is
part of the experiment, not boilerplate: every repair is counted and
reported alongside the ranking metrics.

Recovery policy, in order:
  1. Read identifiers from bracketed groups, ``[12]``.
  2. If that yields nothing, fall back to scraping bare integers, which
     rescues outputs like "2 > 1 > 3".
  3. Drop duplicates, keeping the first occurrence.
  4. Drop identifiers outside 1..num.
  5. Append any identifier the model never mentioned, in its original
     BM25 order.

Step 5 is what makes the output always a valid permutation: in the worst
case (empty or unparseable output) the BM25 order is returned unchanged,
so a parse failure degrades to the baseline rather than to a crash.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

Doc = Dict[str, Any]

# Bracketed identifier, e.g. "[12]". Tolerates "[ 12 ]".
_BRACKET_RE = re.compile(r"\[\s*(\d+)\s*\]")

# Bare integer, for the no-brackets fallback.
_BARE_INT_RE = re.compile(r"\d+")


@dataclass
class ParseResult:
    """A parsed permutation plus everything we want to report about it.

    ``order`` is always a complete 0-indexed permutation of ``range(num)``,
    suitable for passing straight to :func:`apply_permutation`.
    """

    order: List[int]
    num: int
    raw_text: str
    strategy: str = "bracket"
    n_identifiers_parsed: int = 0
    n_duplicates: int = 0
    n_out_of_range: int = 0
    n_missing: int = 0
    is_clean: bool = False
    is_total_failure: bool = False
    notes: List[str] = field(default_factory=list)

    def stats(self) -> Dict[str, Any]:
        """Flat dict for the per-query JSONL log and the summary table."""
        return {
            "strategy": self.strategy,
            "n_identifiers_parsed": self.n_identifiers_parsed,
            "n_duplicates": self.n_duplicates,
            "n_out_of_range": self.n_out_of_range,
            "n_missing": self.n_missing,
            "is_clean": self.is_clean,
            "is_total_failure": self.is_total_failure,
            "notes": self.notes,
        }


def _scrape_identifiers(raw_text: str) -> tuple[List[int], str]:
    """Pull 1-indexed identifiers out of the model's raw output.

    Brackets are tried first because they are what the prompt asks for
    and because they are robust to prose: in "Here are the top 10
    results: [3] > [1]" the bracket pass reads 3 and 1, whereas a bare
    integer scrape would also pick up the 10.
    """
    bracketed = [int(match) for match in _BRACKET_RE.findall(raw_text)]
    if bracketed:
        return bracketed, "bracket"

    bare = [int(match) for match in _BARE_INT_RE.findall(raw_text)]
    if bare:
        return bare, "bare_int"

    return [], "none"


def parse_permutation(raw_text: str, num: int) -> ParseResult:
    """Turn one raw LLM response into a valid permutation of ``num`` items.

    Parameters
    ----------
    raw_text:
        Exactly what the model generated, unmodified.
    num:
        How many passages were shown in the prompt. Identifiers are
        expected in ``1..num`` and the returned order is 0-indexed.
    """
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")

    identity = list(range(num))
    raw_text = raw_text or ""

    identifiers, strategy = _scrape_identifiers(raw_text)

    result = ParseResult(
        order=identity,
        num=num,
        raw_text=raw_text,
        strategy=strategy,
        n_identifiers_parsed=len(identifiers),
    )

    if not identifiers:
        result.n_missing = num
        result.is_total_failure = True
        result.notes.append("no identifiers found; falling back to BM25 order")
        return result

    if strategy == "bare_int":
        result.notes.append("no bracketed identifiers; scraped bare integers")

    # Convert to 0-indexed, then repair. Order of operations matters:
    # dedupe before the range filter so a repeated out-of-range id is
    # counted once in each bucket rather than twice.
    zero_indexed = [identifier - 1 for identifier in identifiers]

    deduped: List[int] = []
    for value in zero_indexed:
        if value not in deduped:
            deduped.append(value)
    result.n_duplicates = len(zero_indexed) - len(deduped)

    in_range = [value for value in deduped if 0 <= value < num]
    result.n_out_of_range = len(deduped) - len(in_range)

    missing = [value for value in identity if value not in in_range]
    result.n_missing = len(missing)

    result.order = in_range + missing
    result.is_clean = (
        result.n_duplicates == 0
        and result.n_out_of_range == 0
        and result.n_missing == 0
        and strategy == "bracket"
    )

    if result.n_duplicates:
        result.notes.append(f"{result.n_duplicates} duplicate identifier(s) dropped")
    if result.n_out_of_range:
        result.notes.append(f"{result.n_out_of_range} out-of-range identifier(s) dropped")
    if result.n_missing:
        result.notes.append(
            f"{result.n_missing} identifier(s) never mentioned; appended in BM25 order"
        )

    # Guard against a logic error in the repair steps above reaching the
    # evaluator as a silently wrong ranking.
    assert sorted(result.order) == identity, "repair did not yield a permutation"

    return result


def apply_permutation(docs: List[Doc], order: List[int]) -> List[Doc]:
    """Reorder ``docs`` by ``order`` and tag each with its new rank.

    The original ``rank`` and ``bm25_score`` fields are left untouched so
    that rank movement stays inspectable for the qualitative examples.
    (The reference implementation overwrites them, which loses that.)
    A new ``llm_rank`` field carries the post-re-ranking position.

    The returned list is in ranked order and every doc still carries its
    ``relevance`` field, which is the only thing Member 2's
    ``evaluate_ranking`` reads.
    """
    if sorted(order) != list(range(len(docs))):
        raise ValueError(
            f"order is not a permutation of {len(docs)} items: {order}"
        )

    reranked: List[Doc] = []
    for new_rank, source_index in enumerate(order, start=1):
        doc = copy.deepcopy(docs[source_index])
        doc["llm_rank"] = new_rank
        reranked.append(doc)

    return reranked


def rerank_with_permutation(docs: List[Doc], raw_text: str) -> tuple[List[Doc], ParseResult]:
    """Convenience wrapper: parse one response and apply it to ``docs``."""
    result = parse_permutation(raw_text, len(docs))
    return apply_permutation(docs, result.order), result
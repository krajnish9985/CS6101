"""Permutation-generation reranking.

Two entry points:

  rerank_single_window  - one LLM call per query over the top-k candidates
                          (the midterm's primary top-20 experiment)
  rerank_sliding_window - the paper's sliding window over a longer list
                          (reference implementation; Member 4 owns integration)

Both batch every LLM call across queries, which is what makes a full SciFact
run take minutes rather than an hour on one GPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .data import Candidate, QueryRecord
from .permutation import ParseResult, parse_permutation
from .prompts import build_messages


@dataclass
class WindowTrace:
    """What happened inside one LLM call."""

    start: int
    end: int
    raw_output: str
    parse: ParseResult
    prompt_tokens: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "raw_output": self.raw_output,
            "prompt_tokens": self.prompt_tokens,
            "parse": self.parse.to_dict(),
        }


@dataclass
class RerankResult:
    query_id: str
    query_text: str
    ranked_doc_ids: List[str]
    input_doc_ids: List[str]
    windows: List[WindowTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "input_doc_ids": self.input_doc_ids,
            "ranked_doc_ids": self.ranked_doc_ids,
            "windows": [w.to_dict() for w in self.windows],
        }


def window_schedule(n: int, window_size: int, step: int) -> List[Tuple[int, int]]:
    """Bottom-up window positions, exactly as in the RankGPT reference code.

    Windows are processed from the END of the list upward, so a strong passage
    sitting near rank 100 can bubble up across several overlapping windows.
    """
    if n <= window_size:
        return [(0, n)]

    schedule = []
    end_pos = n
    start_pos = n - window_size
    while start_pos >= 0:
        schedule.append((start_pos, end_pos))
        end_pos -= step
        start_pos -= step
    # Make sure the very top of the list is covered.
    if schedule and schedule[-1][0] > 0:
        schedule.append((0, min(window_size, schedule[-1][1])))
    return schedule


def _build_batch(
    queries: Sequence[Tuple[str, List[Candidate]]],
    prompt_style: str,
    max_passage_words: Optional[int],
    include_title: bool,
):
    return [
        build_messages(
            style=prompt_style,
            query=qtext,
            candidates=cands,
            max_words=max_passage_words,
            include_title=include_title,
        )
        for qtext, cands in queries
    ]


def rerank_single_window(
    ranker,
    records: Sequence[QueryRecord],
    prompt_style: str = "single_turn",
    max_passage_words: Optional[int] = 120,
    include_title: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 256,
    shuffle_candidates: bool = False,
    seed: int = 42,
    count_prompt_tokens: bool = True,
) -> List[RerankResult]:
    """One permutation-generation call per query."""
    rng = random.Random(seed)

    working: List[List[Candidate]] = []
    for rec in records:
        cands = list(rec.candidates)
        if shuffle_candidates:
            rng.shuffle(cands)
        working.append(cands)

    batch = _build_batch(
        [(rec.query_text, cands) for rec, cands in zip(records, working)],
        prompt_style,
        max_passage_words,
        include_title,
    )

    token_counts = None
    if count_prompt_tokens and hasattr(ranker, "count_tokens"):
        token_counts = [ranker.count_tokens(m) for m in batch]

    outputs = ranker.generate(batch, temperature=temperature, max_tokens=max_tokens)

    results = []
    for i, (rec, cands, raw) in enumerate(zip(records, working, outputs)):
        parse = parse_permutation(raw, len(cands))
        ranked = [cands[j].doc_id for j in parse.order]
        results.append(
            RerankResult(
                query_id=rec.query_id,
                query_text=rec.query_text,
                ranked_doc_ids=ranked,
                input_doc_ids=[c.doc_id for c in cands],
                windows=[
                    WindowTrace(
                        start=0,
                        end=len(cands),
                        raw_output=raw,
                        parse=parse,
                        prompt_tokens=token_counts[i] if token_counts else None,
                    )
                ],
            )
        )
    return results


def rerank_sliding_window(
    ranker,
    records: Sequence[QueryRecord],
    window_size: int = 20,
    step: int = 10,
    prompt_style: str = "single_turn",
    max_passage_words: Optional[int] = 120,
    include_title: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 256,
    count_prompt_tokens: bool = False,
    verbose: bool = True,
) -> List[RerankResult]:
    """Sliding-window PG over the full candidate list.

    Windows are processed bottom-up. All queries advance in lock-step so that
    every window index becomes one batched LLM call.
    """
    state: List[List[Candidate]] = [list(rec.candidates) for rec in records]
    schedules = [window_schedule(len(c), window_size, step) for c in state]
    traces: List[List[WindowTrace]] = [[] for _ in records]

    max_steps = max(len(s) for s in schedules)

    for step_idx in range(max_steps):
        active = [i for i, s in enumerate(schedules) if step_idx < len(s)]
        if not active:
            break

        batch_messages = []
        spans = []
        for i in active:
            start, end = schedules[i][step_idx]
            spans.append((start, end))
            batch_messages.append(
                build_messages(
                    style=prompt_style,
                    query=records[i].query_text,
                    candidates=state[i][start:end],
                    max_words=max_passage_words,
                    include_title=include_title,
                )
            )

        token_counts = None
        if count_prompt_tokens and hasattr(ranker, "count_tokens"):
            token_counts = [ranker.count_tokens(m) for m in batch_messages]

        if verbose:
            print(f"[sliding] window {step_idx + 1}/{max_steps}, {len(active)} queries")

        outputs = ranker.generate(
            batch_messages, temperature=temperature, max_tokens=max_tokens
        )

        for k, i in enumerate(active):
            start, end = spans[k]
            raw = outputs[k]
            chunk = state[i][start:end]
            parse = parse_permutation(raw, len(chunk))
            state[i][start:end] = [chunk[j] for j in parse.order]
            traces[i].append(
                WindowTrace(
                    start=start,
                    end=end,
                    raw_output=raw,
                    parse=parse,
                    prompt_tokens=token_counts[k] if token_counts else None,
                )
            )

    return [
        RerankResult(
            query_id=rec.query_id,
            query_text=rec.query_text,
            ranked_doc_ids=[c.doc_id for c in cands],
            input_doc_ids=[c.doc_id for c in rec.candidates],
            windows=tr,
        )
        for rec, cands, tr in zip(records, state, traces)
    ]

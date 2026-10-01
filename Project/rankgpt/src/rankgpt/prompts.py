"""Prompt templates for listwise permutation generation (PG).

Three styles are provided so we can run a prompt ablation:

  rankgpt_multiturn : faithful to the RankGPT paper (one chat turn per passage)
  single_turn       : all passages in one user message (shorter, usually more
                      reliable for small open models)
  short             : minimal instructions, used to measure how much the long
                      instruction block actually buys us
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .data import Candidate

SYSTEM_PROMPT = (
    "You are RankLLM, an intelligent assistant that can rank passages "
    "based on their relevance to a search query."
)

SYSTEM_PROMPT_SHORT = "You are a helpful assistant that ranks passages."


def _passage_block(
    candidates: Sequence[Candidate],
    max_words: Optional[int],
    include_title: bool,
) -> str:
    lines = []
    for i, cand in enumerate(candidates, start=1):
        body = cand.content(max_words=max_words, include_title=include_title)
        lines.append(f"[{i}] {body}")
    return "\n".join(lines)


def _example_output(num: int) -> str:
    """A format example that matches the window size, e.g. '[4] > [2] > ...'."""
    if num >= 3:
        return "[4] > [2] > [1] > ..." if num >= 4 else "[2] > [1] > [3]"
    return "[2] > [1]"


def build_single_turn(
    query: str,
    candidates: Sequence[Candidate],
    max_words: Optional[int] = 120,
    include_title: bool = True,
) -> List[Dict[str, str]]:
    num = len(candidates)
    user = (
        f"I will provide you with {num} passages, each indicated by a number "
        f"identifier in square brackets. Rank the passages based on their "
        f"relevance to the search query.\n\n"
        f"{_passage_block(candidates, max_words, include_title)}\n\n"
        f"Search Query: {query}\n\n"
        f"Rank the {num} passages above based on their relevance to the search "
        f"query. All passages should be included and listed using identifiers, "
        f"in descending order of relevance. The output format should be "
        f"[] > [] > [], e.g. {_example_output(num)}. Only respond with the "
        f"ranking results, do not say any word or explain."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_short(
    query: str,
    candidates: Sequence[Candidate],
    max_words: Optional[int] = 120,
    include_title: bool = True,
) -> List[Dict[str, str]]:
    num = len(candidates)
    user = (
        f"Query: {query}\n\n"
        f"{_passage_block(candidates, max_words, include_title)}\n\n"
        f"Rank all {num} passages from most to least relevant. "
        f"Answer only with identifiers, e.g. {_example_output(num)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SHORT},
        {"role": "user", "content": user},
    ]


def build_rankgpt_multiturn(
    query: str,
    candidates: Sequence[Candidate],
    max_words: Optional[int] = 120,
    include_title: bool = True,
) -> List[Dict[str, str]]:
    """The paper's original multi-turn construction."""
    num = len(candidates)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"I will provide you with {num} passages, each indicated by "
                f"number identifier []. Rank the passages based on their "
                f"relevance to query: {query}."
            ),
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]

    for i, cand in enumerate(candidates, start=1):
        body = cand.content(max_words=max_words, include_title=include_title)
        messages.append({"role": "user", "content": f"[{i}] {body}"})
        messages.append({"role": "assistant", "content": f"Received passage [{i}]."})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Search Query: {query}.\n"
                f"Rank the {num} passages above based on their relevance to the "
                f"search query. The passages should be listed in descending order "
                f"using identifiers. The most relevant passages should be listed "
                f"first. The output format should be [] > [], e.g. "
                f"{_example_output(num)}. Only respond with the ranking results, "
                f"do not say any word or explain."
            ),
        }
    )
    return messages


BUILDERS = {
    "single_turn": build_single_turn,
    "short": build_short,
    "rankgpt_multiturn": build_rankgpt_multiturn,
}


def build_messages(
    style: str,
    query: str,
    candidates: Sequence[Candidate],
    max_words: Optional[int] = 120,
    include_title: bool = True,
) -> List[Dict[str, str]]:
    if style not in BUILDERS:
        raise ValueError(f"Unknown prompt style {style!r}. Options: {sorted(BUILDERS)}")
    return BUILDERS[style](
        query=query,
        candidates=candidates,
        max_words=max_words,
        include_title=include_title,
    )

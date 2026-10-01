"""Prompt construction for RankGPT-style listwise permutation generation.

The ``paper`` builder reproduces the prompt from Sun et al. (2023),
"Is ChatGPT Good at Search? Investigating Large Language Models as
Re-Ranking Agents" (EMNLP 2023), as published in the paper and in the
authors' reference implementation at https://github.com/sunnweiwei/RankGPT.

The ``compact`` builder is our own single-turn variant, used as a prompt
ablation: it carries the same instructions and the same passage text but
drops the interleaved "Received passage [i]." acknowledgement turns.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

# A chat message in the usual {"role": ..., "content": ...} form.
Message = Dict[str, str]

# One candidate passage, as produced by Member 1's reranking_data.jsonl.
Doc = Dict[str, Any]

# RankGPT truncates every passage to its first 300 whitespace tokens.
MAX_PASSAGE_WORDS = 300

SYSTEM_MESSAGE = (
    "You are RankGPT, an intelligent assistant that can rank passages "
    "based on their relevancy to the query."
)


def truncate_words(text: str, max_words: int = MAX_PASSAGE_WORDS) -> str:
    """Keep the first ``max_words`` whitespace-separated tokens of ``text``.

    Whitespace is collapsed at the same time, which matters because BEIR
    abstracts contain newlines that would otherwise break the [i] layout
    of the prompt.
    """
    if not text:
        return ""
    return " ".join(str(text).split()[:max_words])


def format_passage(doc: Doc, max_words: int = MAX_PASSAGE_WORDS) -> str:
    """Render one candidate as the passage text the LLM sees.

    Title and body are concatenated, matching how BM25 indexed them in
    Member 1's pipeline, so the LLM reads the same text that was
    retrieved. Truncation is applied after concatenation, exactly as in
    the reference implementation.
    """
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    merged = f"{title} {text}".strip()
    return truncate_words(merged, max_words)


def post_prompt(query: str, num: int) -> str:
    """The final instruction turn that asks for the permutation.

    Verbatim from the paper. The literal grammar of "Only response the
    ranking results" is theirs; we keep it so our prompt is the paper's
    prompt and not a paraphrase of it.
    """
    return (
        f"Search Query: {query}. \n"
        f"Rank the {num} passages above based on their relevance to the "
        f"search query. The passages should be listed in descending order "
        f"using identifiers. The most relevant passages should be listed "
        f"first. The output format should be [] > [], e.g., [1] > [2]. "
        f"Only response the ranking results, do not say any word or explain."
    )


def build_paper_messages(
    query: str,
    docs: List[Doc],
    max_words: int = MAX_PASSAGE_WORDS,
) -> List[Message]:
    """Faithful RankGPT prompt: one user/assistant turn pair per passage."""
    num = len(docs)

    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                f"I will provide you with {num} passages, each indicated by "
                f"number identifier []. \nRank the passages based on their "
                f"relevance to query: {query}."
            ),
        },
        {"role": "assistant", "content": "Okay, please provide the passages."},
    ]

    for identifier, doc in enumerate(docs, start=1):
        messages.append(
            {"role": "user", "content": f"[{identifier}] {format_passage(doc, max_words)}"}
        )
        messages.append(
            {"role": "assistant", "content": f"Received passage [{identifier}]."}
        )

    messages.append({"role": "user", "content": post_prompt(query, num)})
    return messages


def build_compact_messages(
    query: str,
    docs: List[Doc],
    max_words: int = MAX_PASSAGE_WORDS,
) -> List[Message]:
    """Ablation prompt: all passages in a single user turn."""
    num = len(docs)

    blocks = "\n\n".join(
        f"[{identifier}] {format_passage(doc, max_words)}"
        for identifier, doc in enumerate(docs, start=1)
    )

    user_content = (
        f"I will provide you with {num} passages, each indicated by number "
        f"identifier [].\nRank the passages based on their relevance to "
        f"query: {query}.\n\n"
        f"{blocks}\n\n"
        f"{post_prompt(query, num)}"
    )

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


PromptBuilder = Callable[[str, List[Doc], int], List[Message]]

# Named so a YAML config can select a variant with a plain string.
PROMPT_BUILDERS: Dict[str, PromptBuilder] = {
    "paper": build_paper_messages,
    "compact": build_compact_messages,
}


def get_prompt_builder(name: str) -> PromptBuilder:
    """Look up a builder by config name, failing loudly on a typo."""
    try:
        return PROMPT_BUILDERS[name]
    except KeyError:
        raise KeyError(
            f"Unknown prompt variant {name!r}. "
            f"Available: {sorted(PROMPT_BUILDERS)}"
        ) from None
"""Unit tests for permutation parsing and prompt construction.

These run on the login node with no GPU: they are what lets us trust the
parsing layer before spending a Slurm allocation on it.
"""

from __future__ import annotations

import pytest

from rankgpt_pg.permutation import (
    apply_permutation,
    parse_permutation,
    rerank_with_permutation,
)
from rankgpt_pg.prompts import (
    build_compact_messages,
    build_paper_messages,
    format_passage,
    get_prompt_builder,
    truncate_words,
)


def make_docs(n: int):
    """Minimal stand-ins shaped like Member 1's retrieved_docs entries."""
    return [
        {
            "rank": i + 1,
            "doc_id": f"d{i + 1}",
            "bm25_score": 10.0 - i,
            "title": f"Title {i + 1}",
            "text": f"Body text for passage {i + 1}.",
            "relevance": 1 if i == 2 else 0,
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Well-formed output
# --------------------------------------------------------------------------

def test_clean_permutation():
    result = parse_permutation("[3] > [1] > [2]", num=3)
    assert result.order == [2, 0, 1]
    assert result.is_clean
    assert result.strategy == "bracket"
    assert result.n_missing == 0


def test_identity_permutation_is_clean():
    result = parse_permutation("[1] > [2] > [3]", num=3)
    assert result.order == [0, 1, 2]
    assert result.is_clean


def test_two_digit_identifiers():
    raw = " > ".join(f"[{i}]" for i in range(20, 0, -1))
    result = parse_permutation(raw, num=20)
    assert result.order == list(range(19, -1, -1))
    assert result.is_clean


# --------------------------------------------------------------------------
# Malformed output: the cases 7B/8B models actually produce
# --------------------------------------------------------------------------

def test_duplicates_keep_first_occurrence():
    result = parse_permutation("[1] > [1] > [2] > [3]", num=3)
    assert result.order == [0, 1, 2]
    assert result.n_duplicates == 1
    assert not result.is_clean


def test_out_of_range_dropped():
    result = parse_permutation("[5] > [1] > [2] > [3]", num=3)
    assert result.order == [0, 1, 2]
    assert result.n_out_of_range == 1


def test_zero_is_out_of_range():
    # Identifiers are 1-indexed, so [0] is invalid, not passage 1.
    result = parse_permutation("[0] > [2] > [1] > [3]", num=3)
    assert result.n_out_of_range == 1
    assert result.order == [1, 0, 2]


def test_missing_identifiers_appended_in_bm25_order():
    result = parse_permutation("[2]", num=3)
    # Mentioned first, then the untouched passages in their original order.
    assert result.order == [1, 0, 2]
    assert result.n_missing == 2


def test_surrounding_prose_is_ignored():
    raw = "Sure! Here is the ranking: [2] > [1] > [3]. Hope this helps."
    result = parse_permutation(raw, num=3)
    assert result.order == [1, 0, 2]
    assert result.strategy == "bracket"


def test_prose_integers_do_not_leak_in():
    # A bare-integer scrape would pick up the "10" here; the bracket pass
    # must win so it does not become a phantom identifier.
    raw = "Here are the top 10 passages: [3] > [1] > [2]"
    result = parse_permutation(raw, num=3)
    assert result.order == [2, 0, 1]
    assert result.n_out_of_range == 0


def test_no_brackets_falls_back_to_bare_integers():
    result = parse_permutation("2 > 1 > 3", num=3)
    assert result.order == [1, 0, 2]
    assert result.strategy == "bare_int"
    assert not result.is_clean


def test_newline_separated_identifiers():
    result = parse_permutation("[3]\n[1]\n[2]", num=3)
    assert result.order == [2, 0, 1]


def test_spaces_inside_brackets():
    result = parse_permutation("[ 3 ] > [ 1 ] > [ 2 ]", num=3)
    assert result.order == [2, 0, 1]


# --------------------------------------------------------------------------
# Total failure must degrade to the BM25 baseline, never crash
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["", "   ", "I cannot help with that.", None])
def test_total_failure_returns_bm25_order(raw):
    result = parse_permutation(raw, num=5)
    assert result.order == [0, 1, 2, 3, 4]
    assert result.is_total_failure
    assert result.n_missing == 5


def test_num_must_be_positive():
    with pytest.raises(ValueError):
        parse_permutation("[1]", num=0)


# --------------------------------------------------------------------------
# Applying a permutation
# --------------------------------------------------------------------------

def test_apply_permutation_reorders_and_tags():
    docs = make_docs(3)
    reranked = apply_permutation(docs, [2, 0, 1])

    assert [d["doc_id"] for d in reranked] == ["d3", "d1", "d2"]
    assert [d["llm_rank"] for d in reranked] == [1, 2, 3]


def test_apply_permutation_preserves_bm25_fields():
    docs = make_docs(3)
    reranked = apply_permutation(docs, [2, 0, 1])

    # Original BM25 rank must survive so rank movement stays inspectable.
    assert reranked[0]["rank"] == 3
    assert reranked[0]["bm25_score"] == 8.0
    assert reranked[0]["relevance"] == 1


def test_apply_permutation_does_not_mutate_input():
    docs = make_docs(3)
    before = [d["doc_id"] for d in docs]
    apply_permutation(docs, [2, 0, 1])
    assert [d["doc_id"] for d in docs] == before
    assert "llm_rank" not in docs[0]


def test_apply_permutation_rejects_non_permutation():
    with pytest.raises(ValueError):
        apply_permutation(make_docs(3), [0, 0, 1])


def test_rerank_wrapper_end_to_end():
    docs = make_docs(4)
    reranked, result = rerank_with_permutation(docs, "[4] > [3] > [2] > [1]")
    assert [d["doc_id"] for d in reranked] == ["d4", "d3", "d2", "d1"]
    assert result.is_clean
    assert len(reranked) == 4


def test_every_candidate_survives_a_garbage_response():
    # The invariant the evaluator depends on: no candidate is ever lost.
    docs = make_docs(20)
    reranked, result = rerank_with_permutation(docs, "[1] > [1] > [99] > banana")
    assert len(reranked) == 20
    assert {d["doc_id"] for d in reranked} == {d["doc_id"] for d in docs}
    assert not result.is_clean


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def test_truncate_words_collapses_whitespace():
    assert truncate_words("a\n\nb   c", max_words=10) == "a b c"
    assert truncate_words("one two three", max_words=2) == "one two"
    assert truncate_words("", max_words=5) == ""


def test_format_passage_merges_title_and_text():
    doc = {"title": "Vitamin D", "text": "A study of outcomes."}
    assert format_passage(doc) == "Vitamin D A study of outcomes."


def test_format_passage_handles_missing_title():
    assert format_passage({"title": None, "text": "Body."}) == "Body."
    assert format_passage({"text": "Body."}) == "Body."


def test_format_passage_truncates_to_word_budget():
    doc = {"title": "T", "text": " ".join(str(i) for i in range(500))}
    assert len(format_passage(doc, max_words=50).split()) == 50


def test_paper_prompt_turn_structure():
    docs = make_docs(3)
    messages = build_paper_messages("my query", docs)

    # system + intro + ack + 2 turns per passage + final instruction
    assert len(messages) == 3 + 2 * 3 + 1
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[3]["content"].startswith("[1] ")
    assert messages[4]["content"] == "Received passage [1]."
    assert "3 passages" in messages[1]["content"]
    assert "my query" in messages[1]["content"]
    assert "[] > []" in messages[-1]["content"]


def test_paper_prompt_alternates_roles():
    messages = build_paper_messages("q", make_docs(5))
    roles = [m["role"] for m in messages[1:]]
    # Chat templates require strict user/assistant alternation after system.
    assert roles == ["user", "assistant"] * 6 + ["user"]


def test_compact_prompt_is_two_messages():
    messages = build_compact_messages("my query", make_docs(4))
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    for identifier in range(1, 5):
        assert f"[{identifier}] " in messages[1]["content"]
    assert "Received passage" not in messages[1]["content"]


def test_both_builders_show_the_same_passage_text():
    docs = make_docs(3)
    paper = build_paper_messages("q", docs)
    compact = build_compact_messages("q", docs)
    for doc in docs:
        body = format_passage(doc)
        assert any(body in m["content"] for m in paper)
        assert body in compact[1]["content"]


def test_builder_registry():
    assert get_prompt_builder("paper") is build_paper_messages
    assert get_prompt_builder("compact") is build_compact_messages
    with pytest.raises(KeyError):
        get_prompt_builder("nope")
"""Unit tests for permutation parsing. Run: PYTHONPATH=src pytest -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rankgpt.permutation import extract_identifiers, parse_permutation
from rankgpt.rerank import window_schedule


def test_clean_permutation():
    r = parse_permutation("[3] > [1] > [2]", 3)
    assert r.order == [2, 0, 1]
    assert r.is_clean


def test_duplicates_keep_first():
    r = parse_permutation("[2] > [2] > [1] > [3]", 3)
    assert r.order == [1, 0, 2]
    assert r.num_duplicates == 1
    assert not r.is_clean


def test_missing_ids_appended_in_input_order():
    r = parse_permutation("[3] > [1]", 4)
    # 2 and 4 are missing, appended in original order
    assert r.order == [2, 0, 1, 3]
    assert r.num_missing == 2


def test_out_of_range_dropped():
    r = parse_permutation("[9] > [1] > [0] > [2]", 2)
    assert r.order == [0, 1]
    assert r.num_out_of_range == 2


def test_prose_around_ranking_is_tolerated():
    text = "Sure! Here is the ranking:\n[2] > [1] > [3]\nLet me know if you need more."
    r = parse_permutation(text, 3)
    assert r.order == [1, 0, 2]
    assert r.is_clean


def test_empty_output_falls_back_to_input_order():
    r = parse_permutation("", 4)
    assert r.order == [0, 1, 2, 3]
    assert r.is_empty
    assert r.num_missing == 4


def test_refusal_text_falls_back_to_input_order():
    r = parse_permutation("I cannot rank these passages.", 3)
    assert r.order == [0, 1, 2]
    assert r.is_empty


def test_bare_number_chain_fallback():
    ids, fallback = extract_identifiers("3 > 1 > 2")
    assert ids == [3, 1, 2]
    assert fallback
    r = parse_permutation("3 > 1 > 2", 3)
    assert r.order == [2, 0, 1]
    assert r.used_fallback_regex


def test_brackets_take_priority_over_bare_numbers():
    ids, fallback = extract_identifiers("Ranking of 20 passages: [2] > [1]")
    assert ids == [2, 1]
    assert not fallback


def test_newline_separated_ids():
    r = parse_permutation("[4]\n[2]\n[1]\n[3]", 4)
    assert r.order == [3, 1, 0, 2]
    assert r.is_clean


def test_whitespace_inside_brackets():
    r = parse_permutation("[ 2 ] > [ 1 ]", 2)
    assert r.order == [1, 0]


@pytest.mark.parametrize("n", [1, 2, 5, 20, 100])
def test_always_returns_valid_permutation(n):
    for text in ["", "garbage", "[1] [1] [1]", "[999]", "[1] > [2] > [3]"]:
        r = parse_permutation(text, n)
        assert sorted(r.order) == list(range(n))


def test_single_candidate():
    r = parse_permutation("[1]", 1)
    assert r.order == [0]
    assert r.is_clean


# --- sliding window schedule ---

def test_schedule_short_list_is_one_window():
    assert window_schedule(15, 20, 10) == [(0, 15)]
    assert window_schedule(20, 20, 10) == [(0, 20)]


def test_schedule_100_20_10_is_bottom_up():
    sched = window_schedule(100, 20, 10)
    assert sched[0] == (80, 100)
    assert sched[-1] == (0, 20)
    assert len(sched) == 9
    # every position must be covered
    covered = set()
    for s, e in sched:
        covered.update(range(s, e))
    assert covered == set(range(100))


def test_schedule_covers_top_when_not_divisible():
    sched = window_schedule(66, 20, 10)
    assert sched[0] == (46, 66)
    assert sched[-1][0] == 0
    covered = set()
    for s, e in sched:
        covered.update(range(s, e))
    assert covered == set(range(66))

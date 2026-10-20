"""Tests for the classification scores, follow-up ranking and correct-vs-wrong summary."""

import numpy as np
import pandas as pd
import pytest

from exoplanet_uq.models import evaluate as ev

# 4 planets, 4 non-planets. The model finds 3 planets, misses 1, and raises 1 false alarm.
LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
PROBS = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.1, 0.05])


# ---------- FR4 ----------


def test_confusion_matrix_counts_the_four_outcomes():
    assert ev.confusion_matrix(LABELS, PROBS) == {"tp": 3, "fp": 1, "fn": 1, "tn": 3}


def test_scores_match_hand_worked_values():
    s = ev.classification_scores(LABELS, PROBS)
    assert s["accuracy"] == pytest.approx(6 / 8)
    assert s["precision"] == pytest.approx(3 / 4)  # 3 found, 1 false alarm
    assert s["recall"] == pytest.approx(3 / 4)  # 3 of 4 planets found
    assert s["f1"] == pytest.approx(0.75)
    assert 0.0 <= s["roc_auc"] <= 1.0


def test_threshold_exactly_one_half_counts_as_planet():
    m = ev.confusion_matrix(np.array([1.0, 0.0]), np.array([0.5, 0.49]))
    assert m == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}


def test_a_model_that_never_says_planet_gives_zero_not_a_crash():
    probs = np.full(8, 0.1)
    s = ev.classification_scores(LABELS, probs)
    assert s["precision"] == 0.0 and s["recall"] == 0.0 and s["f1"] == 0.0
    assert s["confusion_matrix"]["tp"] == 0 and s["confusion_matrix"]["fp"] == 0


def test_perfect_predictions():
    s = ev.classification_scores(LABELS, LABELS * 0.9 + 0.05)
    assert s["accuracy"] == s["precision"] == s["recall"] == s["f1"] == 1.0


@pytest.mark.parametrize(
    "labels, probs",
    [([1, 0], [0.5]), ([1, 3], [0.5, 0.5]), ([1, 0], [1.2, 0.5]), ([], [])],
)
def test_bad_inputs_are_refused(labels, probs):
    with pytest.raises(ValueError):
        ev.confusion_matrix(np.array(labels, dtype=float), np.array(probs, dtype=float))


# ---------- FR7 ----------


def test_follow_up_list_puts_the_most_uncertain_first_with_ranks():
    table = pd.DataFrame(
        {"kepoi_name": ["A", "B", "C", "D"], "total_entropy": [0.1, 0.6, 0.3, 0.6]}
    )
    ranked = ev.rank_for_follow_up(table)
    assert ranked["kepoi_name"].tolist() == ["B", "D", "C", "A"]  # tie B/D by name
    assert ranked["follow_up_rank"].tolist() == [1, 2, 3, 4]
    assert table["kepoi_name"].tolist() == ["A", "B", "C", "D"]  # input unchanged


def test_follow_up_list_needs_the_columns():
    with pytest.raises(ValueError):
        ev.rank_for_follow_up(pd.DataFrame({"kepoi_name": ["A"]}))


# ---------- FR8 ----------


def test_uncertainty_by_correctness_splits_right_and_wrong():
    uncertainty = np.array([0.1, 0.1, 0.1, 0.9, 0.8, 0.1, 0.1, 0.1])  # the 2 mistakes are unsure
    result = ev.uncertainty_by_correctness(LABELS, PROBS, uncertainty)
    assert result["correct"]["count"] == 6 and result["wrong"]["count"] == 2
    assert result["wrong"]["mean"] == pytest.approx(0.85)
    assert result["correct"]["mean"] == pytest.approx(0.1)
    assert result["comparison_possible"]


def test_with_no_wrong_answers_the_comparison_is_not_possible():
    result = ev.uncertainty_by_correctness(LABELS, LABELS * 0.9 + 0.05, np.full(8, 0.2))
    assert result["wrong"] == {"count": 0, "mean": None, "median": None}
    assert not result["comparison_possible"]


def test_uncertainty_must_have_one_value_per_signal():
    with pytest.raises(ValueError):
        ev.uncertainty_by_correctness(LABELS, PROBS, np.ones(3))

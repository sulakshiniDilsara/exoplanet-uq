"""Tests for calibration measures and temperature scaling.

Many tests use the worked examples from the calibration lesson, where the
right answers were already worked out by hand.
"""

import numpy as np
import pytest

from exoplanet_uq.models import calibration as cal


def lesson_example():
    """20 signals in 5 bins (from the lesson): ECE must be 0.12."""
    probs = [0.10] * 6 + [0.50] * 2 + [0.70] * 4 + [0.90] * 8
    labels = [1] + [0] * 5 + [1, 0] + [1, 1, 0, 0] + [1] * 6 + [0] * 2
    return np.array(labels, dtype=float), np.array(probs)


# ---------- reliability table and ECE ----------


def test_ece_matches_the_lesson_example():
    labels, probs = lesson_example()
    assert cal.ece(labels, probs, n_bins=5) == pytest.approx(0.12, abs=1e-9)


def test_reliability_table_skips_empty_bins_and_counts_right():
    labels, probs = lesson_example()
    rows = cal.reliability_table(labels, probs, n_bins=5)
    assert [row["count"] for row in rows] == [6, 2, 4, 8]  # the 0.2-0.4 bin is empty
    assert rows[-1]["mean_predicted"] == pytest.approx(0.90)
    assert rows[-1]["fraction_planets"] == pytest.approx(0.75)


def test_a_probability_of_exactly_one_goes_in_the_last_bin():
    assert cal.bin_index(np.array([0.0, 0.95, 1.0]), n_bins=10).tolist() == [0, 9, 9]


def test_a_perfectly_calibrated_model_has_ece_zero():
    labels = np.array([1, 0, 1, 0], dtype=float)
    assert cal.ece(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.0)


# ---------- Brier and NLL ----------


def test_brier_and_nll_match_the_lesson_examples():
    labels = np.array([1, 0, 0], dtype=float)
    probs = np.array([0.9, 0.7, 0.2])
    assert cal.brier(labels, probs) == pytest.approx(0.18)
    assert cal.nll(labels, probs) == pytest.approx(0.51, abs=0.005)


def test_always_saying_one_half_gives_brier_quarter():
    labels = np.array([1, 0, 1, 1], dtype=float)
    assert cal.brier(labels, np.full(4, 0.5)) == pytest.approx(0.25)


def test_nll_survives_a_confident_mistake_without_crashing():
    value = cal.nll(np.array([0.0]), np.array([1.0]))  # "100 % planet", but it is not
    assert np.isfinite(value) and value > 10


@pytest.mark.parametrize(
    "labels, probs",
    [
        ([1, 0], [0.5]),  # different lengths
        ([2, 0], [0.5, 0.5]),  # label not 0/1
        ([1, 0], [1.5, 0.5]),  # probability above 1
        ([1, 0], [np.nan, 0.5]),  # missing probability
        ([], []),  # nothing at all
    ],
)
def test_bad_inputs_are_refused(labels, probs):
    with pytest.raises(ValueError):
        cal.brier(np.array(labels, dtype=float), np.array(probs, dtype=float))


# ---------- temperature scaling ----------


def test_temperature_matches_the_lesson_table():
    logit = np.array([3.0])
    assert cal.apply_temperature(logit, 0.5)[0] == pytest.approx(0.998, abs=1e-3)
    assert cal.apply_temperature(logit, 1.0)[0] == pytest.approx(0.953, abs=1e-3)
    assert cal.apply_temperature(logit, 2.0)[0] == pytest.approx(0.818, abs=1e-3)


def test_temperature_never_changes_the_predicted_class():
    logits = np.array([-4.0, -0.1, 0.2, 7.0])
    for t in (0.1, 1.0, 5.0):
        assert ((cal.apply_temperature(logits, t) >= 0.5) == (logits > 0)).all()


def test_mc_temperature_divides_each_run_then_averages():
    runs = np.array([[2.0], [-2.0]])  # two runs, one signal
    expected = (cal.sigmoid(np.array([1.0])) + cal.sigmoid(np.array([-1.0]))) / 2  # T = 2
    assert cal.apply_temperature(runs, 2.0)[0] == pytest.approx(expected[0])


def test_sigmoid_does_not_overflow():
    out = cal.sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        cal.apply_temperature(np.array([1.0]), 0.0)


def test_fit_finds_the_temperature_of_an_overconfident_model():
    # Make honest logits, then multiply by 2: an overconfident model whose right T is 2.
    rng = np.random.default_rng(0)
    honest = rng.normal(0, 2, 5000)
    labels = (rng.uniform(size=5000) < cal.sigmoid(honest)).astype(float)
    fit = cal.fit_temperature(labels, honest * 2.0)
    assert fit.trustworthy and fit.reason == ""
    assert fit.wrong_predictions > 0
    assert fit.temperature == pytest.approx(2.0, rel=0.1)


def test_fit_flags_a_set_with_no_mistakes():
    # Every prediction is right: NLL keeps falling as T -> 0, so there is no real best T.
    logits = np.array([3.0, 4.0, -3.0, -5.0])
    labels = np.array([1, 1, 0, 0], dtype=float)
    fit = cal.fit_temperature(labels, logits)
    assert not fit.trustworthy
    assert "no wrong predictions" in fit.reason


def test_fit_flags_no_mistakes_even_when_the_best_t_stops_just_before_the_edge():
    # Like our val set: all MC means are right, but one signal's runs disagree,
    # so the NLL minimum sits a little inside the grid. Still not trustworthy.
    runs = np.array([[6.0, -6.0, 1.5], [6.0, -6.0, -0.5], [6.0, -6.0, 1.0]])
    labels = np.array([1, 0, 1], dtype=float)
    fit = cal.fit_temperature(labels, runs)
    assert fit.wrong_predictions == 0
    assert not fit.trustworthy


def test_fit_flags_a_best_t_at_the_grid_edge():
    rng = np.random.default_rng(1)
    honest = rng.normal(0, 2, 2000)
    labels = (rng.uniform(size=2000) < cal.sigmoid(honest)).astype(float)
    fit = cal.fit_temperature(labels, honest * 2.0, grid=np.array([0.5, 0.8, 1.0]))  # true T=2
    assert not fit.trustworthy
    assert "edge" in fit.reason


# ---------- bootstrap ----------


def test_bootstrap_range_contains_the_ece_and_is_repeatable():
    labels, probs = lesson_example()
    low, high = cal.bootstrap_ece(labels, probs, n_samples=300, n_bins=5)
    assert low <= cal.ece(labels, probs, n_bins=5) <= high
    assert (low, high) == cal.bootstrap_ece(labels, probs, n_samples=300, n_bins=5)

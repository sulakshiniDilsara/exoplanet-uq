"""Tests for the train / val / test split.

They use fake stars, some with several signals, so they need no real data.
"""

import pandas as pd
import pytest

from exoplanet_uq.data import split as sp


def make_signals(n_planet_stars=40, n_other_stars=60, signals_per_star=(1, 2, 3)):
    """Fake signals. Star i gets 1, 2 or 3 signals (in turn), all with the star's label."""
    rows = []
    for star in range(n_planet_stars + n_other_stars):
        label = 1 if star < n_planet_stars else 0
        for number in range(signals_per_star[star % len(signals_per_star)]):
            rows.append(
                {
                    "kepoi_name": f"K{star:05d}.{number + 1:02d}",
                    "kepid": 1000 + star,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


# ---------- star_table ----------


def test_star_table_marks_a_star_with_any_planet_signal():
    signals = pd.DataFrame({"kepoi_name": ["A", "B", "C"], "kepid": [1, 1, 2], "label": [0, 1, 0]})
    assert sp.star_table(signals)["has_planet"].tolist() == [1, 0]


@pytest.mark.parametrize(
    "bad, message",
    [
        (pd.DataFrame({"kepid": [1], "label": [1]}), "missing"),
        (pd.DataFrame(columns=["kepoi_name", "kepid", "label"]), "empty"),
        (
            pd.DataFrame({"kepoi_name": ["A", "A"], "kepid": [1, 2], "label": [1, 0]}),
            "more than once",
        ),
        (pd.DataFrame({"kepoi_name": ["A"], "kepid": [1], "label": [2]}), "0 or 1"),
    ],
)
def test_star_table_rejects_bad_input(bad, message):
    with pytest.raises(sp.SplitError, match=message):
        sp.star_table(bad)


# ---------- split_stars ----------


def test_each_class_is_split_70_15_15():
    stars = sp.star_table(make_signals(n_planet_stars=40, n_other_stars=60))
    result = stars.merge(sp.split_stars(stars), on="kepid")
    counts = result.groupby(["has_planet", "split"]).size().to_dict()
    assert counts[(1, "test")] == 6 and counts[(1, "val")] == 6 and counts[(1, "train")] == 28
    assert counts[(0, "test")] == 9 and counts[(0, "val")] == 9 and counts[(0, "train")] == 42


def test_same_seed_gives_the_same_split_and_another_seed_does_not():
    stars = sp.star_table(make_signals())
    first = sp.split_stars(stars, seed=42)
    pd.testing.assert_frame_equal(first, sp.split_stars(stars, seed=42))
    assert not first.equals(sp.split_stars(stars, seed=7))


def test_row_order_does_not_change_the_split():
    stars = sp.star_table(make_signals())
    shuffled = stars.sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(sp.split_stars(stars), sp.split_stars(shuffled))


@pytest.mark.parametrize("val, test", [(0.0, 0.15), (0.15, 0.0), (0.5, 0.5), (-0.1, 0.2)])
def test_impossible_fractions_are_rejected(val, test):
    stars = sp.star_table(make_signals())
    with pytest.raises(sp.SplitError, match="fractions"):
        sp.split_stars(stars, val_fraction=val, test_fraction=test)


# ---------- assign_splits ----------


def test_every_signal_of_a_star_goes_to_the_same_split():
    result = sp.assign_splits(make_signals())
    assert (result.groupby("kepid")["split"].nunique() == 1).all()


def test_every_signal_is_in_exactly_one_split_and_none_is_lost():
    signals = make_signals()
    result = sp.assign_splits(signals)
    assert len(result) == len(signals)
    assert sorted(result["kepoi_name"]) == sorted(signals["kepoi_name"])
    assert result["split"].isin(sp.SPLITS).all()


def test_planet_share_is_about_the_same_in_every_split():
    result = sp.assign_splits(make_signals(n_planet_stars=40, n_other_stars=60))
    stars = result.drop_duplicates("kepid")
    shares = stars.groupby("split")["label"].mean()
    assert shares.max() - shares.min() < 0.05


def test_assign_does_not_change_the_input():
    signals = make_signals()
    before = signals.copy()
    sp.assign_splits(signals)
    pd.testing.assert_frame_equal(signals, before)


# ---------- check_split (the safety net) ----------


def test_check_split_catches_a_star_in_two_splits():
    table = pd.DataFrame(
        {"kepoi_name": ["A", "B", "C"], "kepid": [1, 1, 2], "split": ["train", "test", "val"]}
    )
    with pytest.raises(sp.SplitError, match="leakage"):
        sp.check_split(table)


@pytest.mark.parametrize("bad_split", [None, "holdout"])
def test_check_split_catches_a_missing_or_unknown_split(bad_split):
    table = pd.DataFrame({"kepoi_name": ["A", "B"], "kepid": [1, 2], "split": ["train", bad_split]})
    with pytest.raises(sp.SplitError, match="exactly one"):
        sp.check_split(table)

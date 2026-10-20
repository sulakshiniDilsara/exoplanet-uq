"""Split the signals into train, validation and test sets, by star.

All signals of one star share the star's light, noise and wobble. If one
signal of a star were in train and another in test, the model could
recognise the star instead of the signal (data leakage). So we split the
STARS, and every signal follows its star.

The split is stratified: stars with a planet and stars without one are
split separately, so all three sets keep about the same planet share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

TRAIN, VAL, TEST = "train", "val", "test"
SPLITS = (TRAIN, VAL, TEST)


class SplitError(ValueError):
    """Raised when the input is wrong or a split would leak."""


def star_table(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per star. A star "has a planet" if any of its signals has label 1."""
    missing = {"kepoi_name", "kepid", "label"} - set(signals.columns)
    if missing:
        raise SplitError(f"The signal table is missing columns: {sorted(missing)}")
    if signals.empty:
        raise SplitError("The signal table is empty.")
    if signals["kepoi_name"].duplicated().any():
        raise SplitError("A signal appears more than once.")
    if not signals["label"].isin([0, 1]).all():
        raise SplitError("Labels must be 0 or 1.")

    stars = signals.groupby("kepid").agg(has_planet=("label", "max")).reset_index()
    stars["has_planet"] = stars["has_planet"].astype(int)
    return stars


def split_stars(
    stars: pd.DataFrame,
    seed: int = SPLIT_SEED,
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
) -> pd.DataFrame:
    """Give every star a split. Each class (planet / no planet) is split on its own."""
    if not (0 < val_fraction and 0 < test_fraction and val_fraction + test_fraction < 1):
        raise SplitError("val and test fractions must be above 0 and add up to less than 1.")

    rng = np.random.default_rng(seed)
    parts = []
    for has_planet in (1, 0):
        # Sort first, so the result depends only on the seed, not on row order.
        ids = np.sort(stars.loc[stars["has_planet"] == has_planet, "kepid"].to_numpy())
        ids = rng.permutation(ids)
        n_test = round(len(ids) * test_fraction)
        n_val = round(len(ids) * val_fraction)
        labels = [TEST] * n_test + [VAL] * n_val + [TRAIN] * (len(ids) - n_test - n_val)
        parts.append(pd.DataFrame({"kepid": ids, "split": labels}))
    return pd.concat(parts).sort_values("kepid").reset_index(drop=True)


def assign_splits(signals: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Return a copy of the signal table with a "split" column, from each signal's star."""
    star_splits = split_stars(star_table(signals), seed=seed)
    result = signals.merge(star_splits, on="kepid", how="left", validate="many_to_one")
    check_split(result)
    return result


def check_split(table: pd.DataFrame) -> None:
    """Stop if any star is in two splits or any signal has no split. Raise SplitError."""
    if table["split"].isna().any() or not table["split"].isin(SPLITS).all():
        raise SplitError("Every signal must be in exactly one of train / val / test.")
    splits_per_star = table.groupby("kepid")["split"].nunique()
    leaking = splits_per_star[splits_per_star > 1].index.tolist()
    if leaking:
        raise SplitError(f"Data leakage: these stars are in more than one split: {leaking[:10]}")

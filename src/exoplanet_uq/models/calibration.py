"""Calibration: is the model's "90 % sure" really right 90 % of the time?

Measures (lower is better for all three):
    ECE   - sort predictions into bins, compare "average predicted" with
            "real fraction of planets" in each bin, weighted by bin size
    Brier - mean of (probability - true answer)^2, no bins
    NLL   - mean of -ln(probability given to the true answer); punishes
            confident mistakes hard

Fix: temperature scaling. Divide every logit by one number T, chosen on the
validation set (never on test). T > 1 softens an overconfident model.
It never changes which class is predicted, so accuracy and ROC-AUC stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BINS = 10
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42

# Keep probabilities away from exactly 0 and 1, so log(0) never happens.
EPSILON = 1e-7


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    trustworthy: bool  # False = do not use this T (see reason)
    reason: str
    wrong_predictions: int  # at T = 1, on the set used for fitting
    grid: np.ndarray
    nll_curve: np.ndarray


def _check(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if labels.shape != probs.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels and probabilities must be two 1-D arrays of the same length.")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1.")
    if ((probs < 0) | (probs > 1)).any() or not np.isfinite(probs).all():
        raise ValueError("probabilities must be between 0 and 1.")
    return labels, probs


def bin_index(probs: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Which equal-width bin each probability falls in. 1.0 goes in the last bin."""
    return np.minimum((np.asarray(probs) * n_bins).astype(int), n_bins - 1)


def reliability_table(labels, probs, n_bins: int = N_BINS) -> list[dict]:
    """One row per NON-empty bin: its range, count, average predicted, real fraction."""
    labels, probs = _check(labels, probs)
    which = bin_index(probs, n_bins)
    rows = []
    for number in range(n_bins):
        inside = which == number
        if inside.any():
            rows.append(
                {
                    "low": number / n_bins,
                    "high": (number + 1) / n_bins,
                    "count": int(inside.sum()),
                    "mean_predicted": float(probs[inside].mean()),
                    "fraction_planets": float(labels[inside].mean()),
                }
            )
    return rows


def ece(labels, probs, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error: bin gaps, weighted by how many signals each bin has."""
    rows = reliability_table(labels, probs, n_bins)
    total = sum(row["count"] for row in rows)
    return float(
        sum(
            row["count"] / total * abs(row["mean_predicted"] - row["fraction_planets"])
            for row in rows
        )
    )


def brier(labels, probs) -> float:
    labels, probs = _check(labels, probs)
    return float(np.mean((probs - labels) ** 2))


def nll(labels, probs) -> float:
    labels, probs = _check(labels, probs)
    probs = np.clip(probs, EPSILON, 1 - EPSILON)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Written this way so very large or very small numbers do not overflow.
    x = np.asarray(x, dtype=float)
    return np.where(
        x >= 0, 1 / (1 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x)))
    )


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Probabilities after dividing the logits by T.

    1-D logits: one prediction per signal.
    2-D logits (runs x signals, MC Dropout): divide EACH run, then average (Option A).
    """
    if not temperature > 0:
        raise ValueError("The temperature must be above 0.")
    probs = sigmoid(np.asarray(logits, dtype=float) / temperature)
    return probs.mean(axis=0) if probs.ndim == 2 else probs


def fit_temperature(labels, logits, grid: np.ndarray = TEMPERATURE_GRID) -> TemperatureFit:
    """Try every T in the grid on these (validation!) signals; keep the one with the lowest NLL.

    The fit is NOT trustworthy when:
    - every prediction is already right: then a sharper model always has a
      lower NLL, so there is no real best T (it runs towards 0), or
    - the best T is at either end of the grid.
    """
    labels = np.asarray(labels, dtype=float)
    curve = np.array([nll(labels, apply_temperature(logits, t)) for t in grid])
    best = int(np.argmin(curve))
    wrong = int(((apply_temperature(logits, 1.0) >= 0.5) != (labels == 1)).sum())

    if wrong == 0:
        reason = "no wrong predictions: the temperature cannot be estimated from this set"
    elif best in (0, len(grid) - 1):
        reason = "best temperature is at the edge of the grid"
    else:
        reason = ""
    return TemperatureFit(float(grid[best]), reason == "", reason, wrong, grid, curve)


def bootstrap_ece(
    labels,
    probs,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    n_bins: int = N_BINS,
) -> tuple[float, float]:
    """95 % range of ECE when the signals are resampled with replacement.

    A wide range means the ECE number is not stable (too few signals).
    """
    labels, probs = _check(labels, probs)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_samples):
        pick = rng.integers(0, len(labels), len(labels))
        values.append(ece(labels[pick], probs[pick], n_bins))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)

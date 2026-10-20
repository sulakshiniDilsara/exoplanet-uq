"""Classification scores, the follow-up ranking, and correct-vs-wrong uncertainty.

These close three requirements from the proposal:
    FR4  accuracy, precision, recall, F1, ROC-AUC and a confusion matrix
    FR7  rank and export the most uncertain candidates for follow-up
    FR8  compare uncertainty for correct and wrong predictions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exoplanet_uq.models.train import roc_auc

THRESHOLD = 0.5


def _check(labels, probs) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if labels.shape != probs.shape or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels and probabilities must be two 1-D arrays of the same length.")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1.")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("probabilities must be between 0 and 1.")
    return labels, probs


def confusion_matrix(labels, probs, threshold: float = THRESHOLD) -> dict[str, int]:
    """Counts of the four outcomes. "planet" means probability >= threshold."""
    labels, probs = _check(labels, probs)
    said_planet = probs >= threshold
    is_planet = labels == 1
    return {
        "tp": int(np.sum(said_planet & is_planet)),  # planet found
        "fp": int(np.sum(said_planet & ~is_planet)),  # false alarm
        "fn": int(np.sum(~said_planet & is_planet)),  # planet missed
        "tn": int(np.sum(~said_planet & ~is_planet)),  # correctly rejected
    }


def _safe_divide(top: float, bottom: float) -> float:
    # 0 / 0 happens, for example, when the model never says "planet".
    # We return 0 instead of crashing; the confusion matrix shows why.
    return float(top / bottom) if bottom > 0 else 0.0


def classification_scores(labels, probs, threshold: float = THRESHOLD) -> dict:
    """FR4: accuracy, precision, recall, F1, ROC-AUC and the confusion matrix."""
    counts = confusion_matrix(labels, probs, threshold)
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "accuracy": _safe_divide(tp + tn, tp + fp + fn + tn),
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "roc_auc": roc_auc(labels, probs),
        "confusion_matrix": counts,
        "threshold": threshold,
    }


def rank_for_follow_up(table: pd.DataFrame, by: str = "total_entropy") -> pd.DataFrame:
    """FR7: the same signals, most uncertain first, with rank 1 = check this first.

    Ties keep a fixed order (by kepoi_name), so the list is the same every run.
    """
    if by not in table.columns or "kepoi_name" not in table.columns:
        raise ValueError(f"The table needs the columns 'kepoi_name' and {by!r}.")
    ranked = table.sort_values([by, "kepoi_name"], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "follow_up_rank", np.arange(1, len(ranked) + 1))
    return ranked


def uncertainty_by_correctness(labels, probs, uncertainty, threshold: float = THRESHOLD) -> dict:
    """FR8: how uncertain the model was on its right answers and on its wrong answers.

    With no wrong answers the "wrong" group is empty: its mean is None, and the
    comparison cannot be made yet (it is not "zero uncertainty").
    """
    labels, probs = _check(labels, probs)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if uncertainty.shape != labels.shape:
        raise ValueError("uncertainty must have one value per signal.")
    correct = (probs >= threshold) == (labels == 1)

    def summary(values: np.ndarray) -> dict:
        if len(values) == 0:
            return {"count": 0, "mean": None, "median": None}
        return {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
        }

    result = {"correct": summary(uncertainty[correct]), "wrong": summary(uncertainty[~correct])}
    result["comparison_possible"] = bool(correct.any() and (~correct).any())
    return result

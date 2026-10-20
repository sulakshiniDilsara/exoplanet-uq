"""Monte Carlo (MC) Dropout: predict many times with dropout ON, and measure the spread.

    one signal -> run 1 (dropout ON) -> 0.97
               -> run 2 (dropout ON) -> 0.99
               -> ...
               -> run T (dropout ON) -> 0.95
    mean = the answer, spread = how unsure the model is

From the same runs we also split the total uncertainty (entropy) into
two parts. This split is an APPROXIMATION (see Depeweg et al., 2018):
    total (entropy of the mean) = aleatoric (mean of each run's entropy)
                                + epistemic (the rest, "mutual information")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from exoplanet_uq.models.data import SplitData
from exoplanet_uq.models.train import as_tensors

REFERENCE_RUNS = 1000
CANDIDATE_RUNS = (10, 20, 50, 100)
TOLERANCE = 0.01
MC_SEED = 42

# Keep probabilities away from exactly 0 and 1, so log(0) never happens.
EPSILON = 1e-7


@dataclass(frozen=True)
class RunsChoice:
    n_runs: int
    converged: bool  # False = no candidate was close enough; the largest one is used
    table: list[dict]  # per candidate: n_runs, max_mean_diff, max_std_diff


def enable_mc_dropout(model: nn.Module) -> int:
    """Prediction mode for everything, then switch ONLY the dropout layers back on.

    Returns how many dropout layers were switched on (0 would make MC useless).
    """
    model.eval()
    switched = 0
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()
            switched += 1
    return switched


def mc_predict(
    model: nn.Module,
    data: SplitData,
    n_runs: int,
    seed: int = MC_SEED,
    as_logits: bool = False,
) -> np.ndarray:
    """Planet probabilities (or logits) from n_runs dropout-ON runs. Shape: (n_runs, n_signals).

    as_logits=True returns the raw outputs, for temperature scaling (Step 7).
    """
    if n_runs < 2:
        raise ValueError("MC Dropout needs at least 2 runs to measure a spread.")
    global_view, local_view, _ = as_tensors(data)
    torch.manual_seed(seed)  # the same seed gives the same dropout choices
    try:
        if enable_mc_dropout(model) == 0:
            raise ValueError("The model has no dropout layer, so MC Dropout cannot work.")
        with torch.no_grad():
            runs = [model(global_view, local_view) for _ in range(n_runs)]
    finally:
        model.eval()  # always leave the model in normal prediction mode
    logits = torch.stack(runs)
    return logits.numpy() if as_logits else torch.sigmoid(logits).numpy()


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """How unsure one probability is, in nats: 0.69 at p = 0.5, near 0 at p = 0 or 1."""
    p = np.clip(np.asarray(p, dtype=float), EPSILON, 1 - EPSILON)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def uncertainty(runs: np.ndarray) -> dict[str, np.ndarray]:
    """Per signal: mean, std, total entropy, and its aleatoric / epistemic parts."""
    runs = np.asarray(runs, dtype=float)
    mean = runs.mean(axis=0)
    total = binary_entropy(mean)
    aleatoric = binary_entropy(runs).mean(axis=0)
    # total >= aleatoric always holds in exact maths; tiny negative values are rounding.
    epistemic = np.clip(total - aleatoric, 0.0, None)
    return {
        "prob_mc_mean": mean,
        "mc_std": runs.std(axis=0),
        "total_entropy": total,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
    }


def choose_n_runs(
    reference_runs: np.ndarray,
    candidates: Sequence[int] = CANDIDATE_RUNS,
    tolerance: float = TOLERANCE,
) -> RunsChoice:
    """Smallest T whose mean AND std are within `tolerance` of the reference, for EVERY signal.

    The first T rows of the reference runs are used as "a run with T samples".
    """
    reference_runs = np.asarray(reference_runs, dtype=float)
    if max(candidates) > len(reference_runs):
        raise ValueError("Every candidate T must be <= the number of reference runs.")
    ref_mean = reference_runs.mean(axis=0)
    ref_std = reference_runs.std(axis=0)

    table = []
    for n_runs in sorted(candidates):
        part = reference_runs[:n_runs]
        table.append(
            {
                "n_runs": n_runs,
                "max_mean_diff": float(np.abs(part.mean(axis=0) - ref_mean).max()),
                "max_std_diff": float(np.abs(part.std(axis=0) - ref_std).max()),
            }
        )
    for row in table:
        if row["max_mean_diff"] <= tolerance and row["max_std_diff"] <= tolerance:
            return RunsChoice(row["n_runs"], True, table)
    return RunsChoice(table[-1]["n_runs"], False, table)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation: +1 = when x goes up, y always goes up; 0 = no link; -1 = opposite."""
    ranks_x = pd.Series(x, dtype=float).rank()
    ranks_y = pd.Series(y, dtype=float).rank()
    return float(ranks_x.corr(ranks_y))

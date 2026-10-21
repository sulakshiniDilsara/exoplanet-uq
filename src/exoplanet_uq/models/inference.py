"""Prediction helpers for the research dashboard (Step 9).

The dashboard only calls these functions, so no science happens inside the
web page. Everything reuses the same code as the pipeline scripts:
    load_split  -> the views and labels of one split
    mc_predict  -> T dropout-ON runs (Step 6)
    uncertainty -> mean, spread and entropies (Step 6)

The test set stays locked here too: only train and val can be loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from exoplanet_uq.data import split as sp
from exoplanet_uq.models import cnn, mc_dropout, train
from exoplanet_uq.models.data import SplitData, load_split

# Deny first: the dashboard may only show these splits. The test set is
# opened once, at the final evaluation, never in a demo.
DASHBOARD_SPLITS = (sp.VAL, sp.TRAIN)


class InferenceError(ValueError):
    """Raised when the dashboard asks for something it may not see or that does not exist."""


def load_model(model_path: Path) -> cnn.TwoBranchCNN:
    """Load the trained CNN in prediction mode.

    weights_only=True: load numbers only, never run code hidden in the file.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise InferenceError(f"Model file not found: {model_path}")
    model = cnn.TwoBranchCNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def load_dashboard_split(
    split_name: str, views_path: Path, split_path: Path, labelled_path: Path
) -> SplitData:
    """Load train or val for the dashboard. Any other name (including test) is refused."""
    if split_name not in DASHBOARD_SPLITS:
        raise InferenceError(
            f"The dashboard can only show {DASHBOARD_SPLITS}; {split_name!r} is locked."
        )
    # allow_test stays False, so load_split refuses the test set a second time.
    return load_split(views_path, split_path, labelled_path, split_name)


def predict_split(
    model: torch.nn.Module, data: SplitData, n_runs: int, seed: int = mc_dropout.MC_SEED
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run MC Dropout on a whole split at once.

    The whole split is run together, with the same seed as Step 6, so a val
    signal gets exactly the numbers saved in metadata/mc_val_predictions.csv
    (one source of truth). Returns one row per signal and the raw runs,
    shape (n_runs, n_signals).
    """
    runs = mc_dropout.mc_predict(model, data, n_runs, seed=seed)
    table = pd.DataFrame({"kepoi_name": data.names, "label": data.labels.astype(int)})
    table["prob_single"] = train.predict_proba(model, data)
    for column, values in mc_dropout.uncertainty(runs).items():
        table[column] = values
    return table, runs


def signal_index(data: SplitData, kepoi_name: str) -> int:
    """Row number of one signal in this split. Unknown names are refused, not guessed."""
    matches = np.flatnonzero(data.names == kepoi_name)
    if len(matches) != 1:
        raise InferenceError(f"{kepoi_name!r} is not a signal of this split.")
    return int(matches[0])


def confidence_words(mc_std: float) -> str:
    """Plain words for the spread of the MC runs, for people who are not ML experts.

    The limits are for display only; the numbers are always shown next to them.
    """
    if mc_std < 0.02:
        return "very sure"
    if mc_std < 0.10:
        return "fairly sure"
    return "unsure: worth a human look"

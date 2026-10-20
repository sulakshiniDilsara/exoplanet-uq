"""Load the views of one split, with their labels, ready for the model.

The test set is locked: asking for it raises an error unless the caller
says allow_test=True on purpose. Only the final evaluation may do that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet_uq.data import split as sp


class DataError(ValueError):
    """Raised when the views, split and labels do not fit together."""


@dataclass(frozen=True)
class SplitData:
    """Everything the model needs from one split, in the same row order."""

    names: np.ndarray  # kepoi_name
    global_view: np.ndarray  # (n, 2001), scaled
    local_view: np.ndarray  # (n, 201), scaled
    labels: np.ndarray  # (n,), 0 or 1

    def __len__(self) -> int:
        return len(self.labels)


def scale_view(view: np.ndarray) -> np.ndarray:
    """Move the median to 0 and the deepest point to -1 (one view at a time).

    A real view always has a dip, so median > minimum. If a view is flat,
    we only move it to 0 instead of dividing by zero.
    """
    view = np.asarray(view, dtype=np.float32)
    median = float(np.median(view))
    depth = median - float(np.min(view))
    if depth <= 0:
        return view - median
    return ((view - median) / depth).astype(np.float32)


def load_split(
    views_path: Path,
    split_path: Path,
    labelled_path: Path,
    split_name: str,
    allow_test: bool = False,
) -> SplitData:
    """Load and scale the views of one split. Labels come from koi_labelled.csv."""
    if split_name not in sp.SPLITS:
        raise DataError(f"Unknown split {split_name!r}; use one of {sp.SPLITS}.")
    if split_name == sp.TEST and not allow_test:
        raise DataError("The test set is locked. Only the final evaluation may open it.")

    split = pd.read_csv(split_path)
    names = split.loc[split["split"] == split_name, "kepoi_name"].to_numpy()
    if len(names) == 0:
        raise DataError(f"The {split_name} split is empty.")

    labelled = pd.read_csv(labelled_path).set_index("kepoi_name")
    missing_labels = set(names) - set(labelled.index)
    if missing_labels:
        raise DataError(f"No label for: {sorted(missing_labels)[:5]}")
    labels = labelled.loc[names, "label"].to_numpy().astype(np.float32)

    # allow_pickle=False: the file may only hold plain numbers and text.
    with np.load(views_path, allow_pickle=False) as views:
        row_of = {name: row for row, name in enumerate(views["kepoi_name"])}
        missing_views = [name for name in names if name not in row_of]
        if missing_views:
            raise DataError(f"No views for: {missing_views[:5]}")
        rows = [row_of[name] for name in names]
        global_view = np.stack([scale_view(v) for v in views["global_view"][rows]])
        local_view = np.stack([scale_view(v) for v in views["local_view"][rows]])

    return SplitData(names, global_view, local_view, labels)

"""Step 8: classification scores (FR4), follow-up ranking (FR7), correct vs wrong (FR8).

Run from the project root, after Step 6:
    python scripts/08_evaluate.py

It reads the Step 6 predictions (metadata/mc_val_predictions.csv), so the
model is not run again and the numbers match Step 6 exactly.
Validation set only. The test set is NOT used here.

Outputs:
    metadata/evaluation_val.json      FR4 scores for A (single) and B (MC mean), FR8 summary
    metadata/follow_up_val.csv        FR7: most uncertain signal first, with a rank
    data/models/confusion_val.png     the two confusion matrices
"""

import json

import matplotlib

matplotlib.use("Agg")  # draw to a file only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exoplanet_uq.config import DATA_DIR, METADATA_DIR  # noqa: E402
from exoplanet_uq.models import evaluate as ev  # noqa: E402

MODEL_DIR = DATA_DIR / "models"
FOLLOW_UP_COLUMNS = [
    "follow_up_rank",
    "kepoi_name",
    "kepid",
    "label",
    "prob_mc_mean",
    "mc_std",
    "total_entropy",
    "aleatoric",
    "epistemic",
    "koi_model_snr",
    "koi_depth",
    "koi_fpflag_ss",
]


def plot_confusion(scores: dict, path) -> None:
    fig, axes = plt.subplots(1, len(scores), figsize=(4.5 * len(scores), 4))
    for ax, (name, result) in zip(np.atleast_1d(axes), scores.items(), strict=True):
        m = result["confusion_matrix"]
        grid = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
        ax.imshow(grid, cmap="Blues")
        for (row, col), value in np.ndenumerate(grid):
            # White text on dark cells, black on light ones, so every number is readable.
            color = "white" if value > grid.max() / 2 else "black"
            ax.text(col, row, str(value), ha="center", va="center", fontsize=16, color=color)
        ax.set_xticks([0, 1], ["says planet", "says not planet"])
        ax.set_yticks([0, 1], ["really planet", "really not planet"])
        ax.set_title(f"{name} (val)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    table = pd.read_csv(METADATA_DIR / "mc_val_predictions.csv")
    labels = table["label"].to_numpy()

    scores = {
        "A_single": ev.classification_scores(labels, table["prob_single"]),
        "B_mc_mean": ev.classification_scores(labels, table["prob_mc_mean"]),
    }
    by_correctness = {
        measure: ev.uncertainty_by_correctness(labels, table["prob_mc_mean"], table[measure])
        for measure in ("mc_std", "total_entropy")
    }

    ranked = ev.rank_for_follow_up(table, by="total_entropy")
    ranked[FOLLOW_UP_COLUMNS].to_csv(METADATA_DIR / "follow_up_val.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(scores, MODEL_DIR / "confusion_val.png")

    info = {
        "split": "val",
        "signals": len(table),
        "source": "metadata/mc_val_predictions.csv (Step 6)",
        "scores": scores,
        "uncertainty_by_correctness": by_correctness,
        "follow_up_top5": ranked["kepoi_name"].head(5).tolist(),
    }
    (METADATA_DIR / "evaluation_val.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

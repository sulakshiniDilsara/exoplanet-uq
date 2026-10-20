"""Step 5: train the two-branch 1D-CNN on the train split, check it on the val split.

Run from the project root, after Step 4:
    python scripts/05_train.py

The test split is NOT loaded here. It is opened once, at the final evaluation.

Outputs:
    data/models/cnn_pilot.pt        the model's knobs (weights only)
    data/models/loss_curve.png      train and val loss per epoch
    metadata/train_history.csv      the same numbers as the picture
    metadata/train_info.json        settings, best epoch, val scores
"""

import json
import time

import matplotlib

matplotlib.use("Agg")  # draw to a file only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from exoplanet_uq.config import DATA_DIR, METADATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from exoplanet_uq.data import split as sp  # noqa: E402
from exoplanet_uq.models import cnn, train  # noqa: E402
from exoplanet_uq.models.data import load_split  # noqa: E402

MODEL_DIR = DATA_DIR / "models"


def main() -> None:
    paths = (
        PROCESSED_DATA_DIR / "views.npz",
        METADATA_DIR / "split.csv",
        METADATA_DIR / "koi_labelled.csv",
    )
    train_data = load_split(*paths, sp.TRAIN)
    val_data = load_split(*paths, sp.VAL)
    pos_weight = train.class_weight(train_data.labels)
    print(f"train {len(train_data)}, val {len(val_data)}, planet weight {pos_weight:.2f}")

    train.set_seed()
    model = cnn.TwoBranchCNN()
    print(f"model knobs: {cnn.count_parameters(model):,}")

    started = time.time()
    result = train.train_model(model, train_data, val_data, pos_weight)
    seconds = round(time.time() - started, 1)
    print(f"stopped after {len(result.history)} epochs, best epoch {result.best_epoch}")

    val_prob = train.predict_proba(model, val_data)
    train_prob = train.predict_proba(model, train_data)
    scores = {
        "val_loss": round(result.best_val_loss, 4),
        "val_accuracy": round(train.accuracy(val_data.labels, val_prob), 3),
        "val_roc_auc": round(train.roc_auc(val_data.labels, val_prob), 3),
        "train_accuracy": round(train.accuracy(train_data.labels, train_prob), 3),
        "train_roc_auc": round(train.roc_auc(train_data.labels, train_prob), 3),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Save only the knobs (state_dict), never the whole Python object.
    torch.save(model.state_dict(), MODEL_DIR / "cnn_pilot.pt")

    history = pd.DataFrame(result.history)
    history.to_csv(METADATA_DIR / "train_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_loss"], label="val loss")
    ax.axvline(result.best_epoch, color="grey", linestyle="--", label="best epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (class-weighted BCE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "loss_curve.png", dpi=80)
    plt.close(fig)

    info = {
        "model": "TwoBranchCNN",
        "parameters": cnn.count_parameters(model),
        "dropout_rate": cnn.DROPOUT_RATE,
        "kernel_size": cnn.KERNEL_SIZE,
        "pool_size": cnn.POOL_SIZE,
        "hidden_units": cnn.HIDDEN_UNITS,
        "learning_rate": train.LEARNING_RATE,
        "batch_size": train.BATCH_SIZE,
        "max_epochs": train.MAX_EPOCHS,
        "patience": train.PATIENCE,
        "seed": train.SEED,
        "pos_weight": round(pos_weight, 4),
        "train_signals": len(train_data),
        "val_signals": len(val_data),
        "epochs_run": len(result.history),
        "best_epoch": result.best_epoch,
        "train_seconds": seconds,
        "torch_version": torch.__version__,
        "scores": scores,
    }
    (METADATA_DIR / "train_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

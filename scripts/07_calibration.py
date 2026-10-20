"""Step 7: calibration on the validation set (ECE, Brier, NLL, temperature scaling).

Run from the project root, after Step 6:
    python scripts/07_calibration.py

Three versions are compared:
    A  single prediction (dropout OFF)
    B  MC Dropout mean (T runs from Step 6)
    C  MC Dropout + temperature scaling (temperature fitted on val)

The test split is NOT loaded here.

Outputs:
    metadata/calibration_val.json        scores, reliability tables, temperature fit
    data/models/reliability_val.png      reliability diagram + NLL vs temperature
"""

import json

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from exoplanet_uq.config import DATA_DIR, METADATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from exoplanet_uq.data import split as sp  # noqa: E402
from exoplanet_uq.models import calibration as cal  # noqa: E402
from exoplanet_uq.models import cnn, mc_dropout, train  # noqa: E402
from exoplanet_uq.models.data import load_split  # noqa: E402

MODEL_DIR = DATA_DIR / "models"


def scores(labels, probs) -> dict:
    low, high = cal.bootstrap_ece(labels, probs)
    return {
        "accuracy": round(train.accuracy(labels, probs), 4),
        "roc_auc": round(train.roc_auc(labels, probs), 4),
        "ece": round(cal.ece(labels, probs), 4),
        "ece_bootstrap_95": [round(low, 4), round(high, 4)],
        "brier": round(cal.brier(labels, probs), 4),
        "nll": round(cal.nll(labels, probs), 4),
        "reliability": cal.reliability_table(labels, probs),
    }


def plot(results: dict, fit: cal.TemperatureFit, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    for name, marker in (("A_single", "o"), ("B_mc_mean", "s")):
        rows = results[name]["reliability"]
        x = [row["mean_predicted"] for row in rows]
        y = [row["fraction_planets"] for row in rows]
        axes[0].plot(x, y, marker=marker, label=name)
        for xi, yi, row in zip(x, y, rows, strict=True):
            axes[0].annotate(str(row["count"]), (xi, yi), textcoords="offset points", xytext=(5, 5))
    axes[0].set_xlabel("average predicted P(planet) in the bin")
    axes[0].set_ylabel("real fraction of planets in the bin")
    axes[0].set_title("reliability diagram (val; numbers = signals per bin)")
    axes[0].legend()

    axes[1].plot(fit.grid, fit.nll_curve)
    axes[1].axvline(
        fit.temperature, color="red", linestyle=":", label=f"best T = {fit.temperature:.3g}"
    )
    axes[1].axvline(1.0, color="grey", linestyle="--", label="T = 1 (no change)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("temperature T")
    axes[1].set_ylabel("val NLL (MC mean, Option A)")
    axes[1].set_title("temperature search" + ("" if fit.trustworthy else "  [NOT trustworthy]"))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    paths = (
        PROCESSED_DATA_DIR / "views.npz",
        METADATA_DIR / "split.csv",
        METADATA_DIR / "koi_labelled.csv",
    )
    val = load_split(*paths, sp.VAL)
    labels = val.labels

    model = cnn.TwoBranchCNN()
    # weights_only=True: load numbers only, never run code hidden in the file.
    model.load_state_dict(torch.load(MODEL_DIR / "cnn_pilot.pt", weights_only=True))

    n_runs = json.loads((METADATA_DIR / "mc_info.json").read_text(encoding="utf-8"))["chosen_runs"]
    single_logits = train.predict_logits(model, val)
    mc_logits = mc_dropout.mc_predict(model, val, mc_dropout.REFERENCE_RUNS, as_logits=True)[
        :n_runs
    ]  # the same first n_runs runs (same seed) that Step 6 used

    fit = cal.fit_temperature(labels, mc_logits)
    results = {
        "A_single": scores(labels, cal.apply_temperature(single_logits, 1.0)),
        "B_mc_mean": scores(labels, cal.apply_temperature(mc_logits, 1.0)),
    }
    if fit.trustworthy:
        results["C_mc_temperature"] = scores(
            labels, cal.apply_temperature(mc_logits, fit.temperature)
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot(results, fit, MODEL_DIR / "reliability_val.png")

    info = {
        "split": "val",
        "signals": int(len(labels)),
        "wrong_predictions_single": int(((single_logits > 0) != (labels == 1)).sum()),
        "n_bins": cal.N_BINS,
        "mc_runs": n_runs,
        "temperature_fit": {
            "best_temperature": round(fit.temperature, 4),
            "trustworthy": fit.trustworthy,
            "reason": fit.reason,
            "wrong_predictions_mc": fit.wrong_predictions,
            "grid": [round(float(fit.grid[0]), 3), round(float(fit.grid[-1]), 3)],
            "nll_at_T1": round(float(cal.nll(labels, cal.apply_temperature(mc_logits, 1.0))), 4),
            "note": (
                "fitted on val; apply the same T to test at the final evaluation"
                if fit.trustworthy
                else "temperature NOT used; version C is not reported for this set"
            ),
        },
        "results": results,
    }
    (METADATA_DIR / "calibration_val.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        k: {m: v[m] for m in ("ece", "ece_bootstrap_95", "brier", "nll")}
        for k, v in results.items()
    }
    print(json.dumps({"temperature_fit": info["temperature_fit"], "scores": summary}, indent=2))


if __name__ == "__main__":
    main()

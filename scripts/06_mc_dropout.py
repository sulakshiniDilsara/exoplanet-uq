"""Step 6: MC Dropout on the validation set. Choose T, measure the uncertainty.

Run from the project root, after Step 5:
    python scripts/06_mc_dropout.py

The test split is NOT loaded here. The error analysis on test happens once,
at the final evaluation.

Outputs:
    metadata/mc_val_predictions.csv    one row per val signal
    metadata/mc_info.json              chosen T, convergence table, checks
    data/models/mc_convergence.png     how the answer settles as T grows
    data/models/mc_uncertainty.png     spread vs answer, uncertainty vs SNR
"""

import json

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from exoplanet_uq.config import DATA_DIR, METADATA_DIR, PROCESSED_DATA_DIR  # noqa: E402
from exoplanet_uq.data import split as sp  # noqa: E402
from exoplanet_uq.models import cnn, mc_dropout, train  # noqa: E402
from exoplanet_uq.models.data import load_split  # noqa: E402

MODEL_DIR = DATA_DIR / "models"
HARD_CASE_COLUMNS = ["koi_model_snr", "koi_depth", "koi_fpflag_ss"]


def plot_convergence(choice: mc_dropout.RunsChoice, path) -> None:
    table = pd.DataFrame(choice.table)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["n_runs"], table["max_mean_diff"], "o-", label="mean: biggest difference")
    ax.plot(table["n_runs"], table["max_std_diff"], "s-", label="std: biggest difference")
    ax.axhline(mc_dropout.TOLERANCE, color="grey", linestyle="--", label="tolerance")
    ax.axvline(choice.n_runs, color="green", linestyle=":", label=f"chosen T = {choice.n_runs}")
    ax.set_xscale("log")
    ax.set_xlabel("T (number of MC runs)")
    ax.set_ylabel(f"difference from T = {mc_dropout.REFERENCE_RUNS}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def plot_uncertainty(table: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, marker, name in ((1, "o", "planet"), (0, "x", "false positive")):
        part = table[table["label"] == label]
        axes[0].scatter(part["prob_mc_mean"], part["mc_std"], marker=marker, label=name)
        axes[1].scatter(part["koi_model_snr"], part["total_entropy"], marker=marker, label=name)
    axes[0].set_xlabel("MC mean (planet probability)")
    axes[0].set_ylabel("MC std (spread)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("signal-to-noise ratio (koi_model_snr)")
    axes[1].set_ylabel("total entropy")
    for ax in axes:
        ax.legend()
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

    model = cnn.TwoBranchCNN()
    # weights_only=True: load numbers only, never run code hidden in the file.
    model.load_state_dict(torch.load(MODEL_DIR / "cnn_pilot.pt", weights_only=True))

    prob_single = train.predict_proba(model, val)
    reference = mc_dropout.mc_predict(model, val, mc_dropout.REFERENCE_RUNS)
    choice = mc_dropout.choose_n_runs(reference)
    print(f"chosen T = {choice.n_runs} (converged: {choice.converged})")

    measures = mc_dropout.uncertainty(reference[: choice.n_runs])
    table = pd.DataFrame({"kepoi_name": val.names, "label": val.labels.astype(int)})
    table["prob_single"] = prob_single
    for column, values in measures.items():
        table[column] = values

    labelled = pd.read_csv(METADATA_DIR / "koi_labelled.csv")
    table = table.merge(
        labelled[["kepoi_name", "kepid", *HARD_CASE_COLUMNS]], on="kepoi_name", how="left"
    )
    table.to_csv(METADATA_DIR / "mc_val_predictions.csv", index=False)

    wrong = (table["prob_mc_mean"] >= 0.5) != (table["label"] == 1)
    false_positives = table[table["label"] == 0]
    checks = {
        "val_signals": len(table),
        "wrong_predictions": int(wrong.sum()),
        "max_abs_mc_minus_single": round(
            float((table["prob_mc_mean"] - table["prob_single"]).abs().max()), 4
        ),
        "mc_std_median": round(float(table["mc_std"].median()), 5),
        "mc_std_max": round(float(table["mc_std"].max()), 4),
        "spearman_entropy_vs_snr": round(
            mc_dropout.spearman(table["total_entropy"], table["koi_model_snr"]), 3
        ),
        "spearman_entropy_vs_depth": round(
            mc_dropout.spearman(table["total_entropy"], table["koi_depth"]), 3
        ),
        "fp_mean_entropy_ss_flag_1": round(
            float(
                false_positives.loc[false_positives["koi_fpflag_ss"] == 1, "total_entropy"].mean()
            ),
            5,
        ),
        "fp_mean_entropy_ss_flag_0": round(
            float(
                false_positives.loc[false_positives["koi_fpflag_ss"] == 0, "total_entropy"].mean()
            ),
            5,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot_convergence(choice, MODEL_DIR / "mc_convergence.png")
    plot_uncertainty(table, MODEL_DIR / "mc_uncertainty.png")

    info = {
        "reference_runs": mc_dropout.REFERENCE_RUNS,
        "candidate_runs": list(mc_dropout.CANDIDATE_RUNS),
        "tolerance": mc_dropout.TOLERANCE,
        "seed": mc_dropout.MC_SEED,
        "chosen_runs": choice.n_runs,
        "converged": choice.converged,
        "convergence": choice.table,
        "dropout_rate": cnn.DROPOUT_RATE,
        "decomposition_note": (
            "aleatoric/epistemic split is an approximation (Depeweg et al., 2018)"
        ),
        "checks": checks,
    }
    (METADATA_DIR / "mc_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

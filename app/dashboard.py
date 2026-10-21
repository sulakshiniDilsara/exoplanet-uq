"""Exoplanet UQ research dashboard (Step 9).

Run from the project root:
    streamlit run app/dashboard.py

This file is the web page only. The science comes from the exoplanet_uq
package (models/inference.py) and from the saved results in metadata/,
so the dashboard always shows the same numbers as the pipeline.

Safety: runs on this computer only (see .streamlit/config.toml), has no
file upload, and never loads the test set.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from exoplanet_uq.config import DATA_DIR, METADATA_DIR, PROCESSED_DATA_DIR
from exoplanet_uq.data import preprocess
from exoplanet_uq.data import split as sp
from exoplanet_uq.models import inference

MODEL_PATH = DATA_DIR / "models" / "cnn_pilot.pt"
VIEWS_PATH = PROCESSED_DATA_DIR / "views.npz"
SPLIT_PATH = METADATA_DIR / "split.csv"
LABELLED_PATH = METADATA_DIR / "koi_labelled.csv"

# Probabilities run from 0 to 1; the small margin keeps dots at the edges visible.
PROBABILITY_AXIS = [-0.03, 1.03]
TABLE_ROW_PIXELS = 35  # Streamlit's default table row height
PLANET_COLOR = "#E8A33D"
NOT_PLANET_COLOR = "#6C8EBF"
LABEL_NAMES = {1: "planet (CONFIRMED)", 0: "not a planet (FALSE POSITIVE)"}

PAGES = (
    "Overview",
    "Signal Explorer",
    "Uncertainty",
    "Follow-up List",
    "Calibration",
    "Research Evidence",
)


# ---------- loading (cached, so each file is read once) ----------


@st.cache_data
def read_json(name: str) -> dict:
    return json.loads((METADATA_DIR / name).read_text(encoding="utf-8"))


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METADATA_DIR / name)


@st.cache_resource
def get_model():
    return inference.load_model(MODEL_PATH)


@st.cache_resource
def get_split(split_name: str):
    return inference.load_dashboard_split(split_name, VIEWS_PATH, SPLIT_PATH, LABELLED_PATH)


@st.cache_resource
def get_predictions(split_name: str):
    """MC Dropout for a whole split, with the T chosen in Step 6 (one source of truth)."""
    n_runs = read_json("mc_info.json")["chosen_runs"]
    return inference.predict_split(get_model(), get_split(split_name), n_runs)


def check_files() -> None:
    """Stop with a clear message if a needed file is missing, instead of a crash."""
    needed = [MODEL_PATH, VIEWS_PATH, SPLIT_PATH, LABELLED_PATH]
    missing = [path for path in needed if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(DATA_DIR.parent)) for path in missing)
        st.error(f"Missing file(s): {names}. Run `git pull` or the pipeline steps first.")
        st.stop()


# ---------- small chart helpers ----------


def view_chart(x: np.ndarray, y: np.ndarray, x_title: str, title: str) -> alt.Chart:
    """One light-curve view as a line. The dip should sit in the middle at -1."""
    data = pd.DataFrame({"x": x, "brightness": y})
    return (
        alt.Chart(data, title=title)
        .mark_line(color=PLANET_COLOR, strokeWidth=1.2)
        .encode(
            x=alt.X("x:Q", title=x_title),
            y=alt.Y("brightness:Q", title="scaled brightness (dip = -1)"),
            tooltip=[alt.Tooltip("x:Q", format=".3f"), alt.Tooltip("brightness:Q", format=".3f")],
        )
        .interactive()
    )


def runs_histogram(runs: np.ndarray) -> alt.Chart:
    """Where the T dropout-ON answers fall between 0 (not a planet) and 1 (planet)."""
    data = pd.DataFrame({"planet probability": runs})
    return (
        alt.Chart(data, title=f"The {len(runs)} MC Dropout answers")
        .mark_bar(color=PLANET_COLOR)
        .encode(
            x=alt.X(
                "planet probability:Q",
                bin=alt.Bin(extent=[0, 1], step=0.05),
                scale=alt.Scale(domain=PROBABILITY_AXIS),
            ),
            y=alt.Y("count():Q", title="number of runs"),
        )
    )


def label_color() -> alt.Color:
    return alt.Color(
        "truth:N",
        scale=alt.Scale(domain=["planet", "not a planet"], range=[PLANET_COLOR, NOT_PLANET_COLOR]),
        title="NASA answer",
    )


# ---------- pages ----------


def page_overview() -> None:
    processed = read_json("processed_info.json")
    split_info = read_json("split_info.json")["counts"]
    scores = read_json("evaluation_val.json")["scores"]["B_mc_mean"]
    mc = read_json("mc_info.json")
    calibration = read_json("calibration_val.json")["results"]["A_single"]

    st.header("Research overview")
    cards = st.columns(5)
    cards[0].metric("Kepler stars", processed["stars"], border=True)
    cards[1].metric(
        "Signals used", processed["kept"], help=f"{processed['signals']} found", border=True
    )
    cards[2].metric(
        "Val accuracy", f"{scores['accuracy']:.2f}", help="31 validation signals", border=True
    )
    cards[3].metric("MC Dropout runs (T)", mc["chosen_runs"], border=True)
    cards[4].metric("ECE (val)", f"{calibration['ece']:.3f}", help="lower is better", border=True)

    st.subheader("Research workflow")
    st.markdown(
        "NASA KOI catalogue (labels) → Kepler light curves → cleaning and phase folding → "
        "split **by star** (train / val / test) → two-branch **1D-CNN** → "
        "**Monte Carlo Dropout** (uncertainty) → **calibration** (is the confidence honest?) → "
        "ranked **follow-up list**"
    )

    st.subheader("The question this research answers")
    st.markdown(
        'A normal classifier says *"planet: 0.92"* and stops. This component also asks '
        "**how sure is the model?** (MC Dropout) and **is that sureness honest?** "
        "(reliability diagram, ECE, Brier, NLL). Signals the model is unsure about are ranked "
        "first, so astronomers can spend telescope time where it matters."
    )

    st.subheader("Current status")
    counts = ", ".join(
        f"{name} {split_info[name]['signals']} signals ({split_info[name]['stars']} stars)"
        for name in sp.SPLITS
    )
    st.markdown(
        f"Pilot dataset: {processed['stars']} Kepler stars, {processed['kept']} usable signals. "
        f"Split by star: {counts}. Steps 1-8 (data, model, MC Dropout, calibration, evaluation) "
        "run on the validation set. The test set is **locked** until the final evaluation."
    )
    st.info(
        "Validation scored 31 / 31, so the model is very confident. That is exactly why this "
        "research checks whether the confidence is honest, and why a larger dataset is next."
    )


def page_signal_explorer() -> None:
    st.header("Signal explorer")
    st.caption(
        "Pick one signal. The model runs live with dropout ON, the same number of times as "
        "in Step 6, and shows its answer and how sure it is."
    )

    choice = st.radio(
        "Signals from",
        ("Validation (not seen in training)", "Training (seen in training)"),
        horizontal=True,
        key="signal_set",
    )
    split_name = sp.VAL if choice.startswith("Validation") else sp.TRAIN
    data = get_split(split_name)
    table, runs = get_predictions(split_name)
    labelled = read_csv("koi_labelled.csv").set_index("kepoi_name")

    def describe(name: str) -> str:
        return f"{name} · NASA: {LABEL_NAMES[int(labelled.loc[name, 'label'])]}"

    # Most uncertain first, so the interesting cases are at the top of the list.
    ordered = table.sort_values("total_entropy", ascending=False)["kepoi_name"].tolist()
    name = st.selectbox(
        "Signal (most uncertain first)", ordered, format_func=describe, key="signal"
    )
    row = inference.signal_index(data, name)
    result = table.iloc[row]
    facts = labelled.loc[name]

    if split_name == sp.TRAIN:
        st.warning(
            "Seen in training: the model learned from this signal, so its answer here is not "
            "a fair test. Use validation signals to judge the model."
        )

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Model answer")
        probability = float(result["prob_mc_mean"])
        verdict = "Planet" if probability >= 0.5 else "Not a planet"
        st.metric(
            "Planet probability",
            f"{probability:.1%}",
            help="mean of the MC Dropout runs",
        )
        st.metric("Uncertainty (spread)", f"± {float(result['mc_std']):.3f}")
        st.markdown(
            f"**Verdict:** {verdict} · the model is "
            f"**{inference.confidence_words(float(result['mc_std']))}**"
        )
        correct = (probability >= 0.5) == (int(result["label"]) == 1)
        st.markdown(f"**NASA answer:** {LABEL_NAMES[int(result['label'])]}")
        st.markdown("**Match:** " + ("yes ✅" if correct else "no ❌"))
        with st.expander("More numbers"):
            st.markdown(
                f"- Single prediction (dropout off): {float(result['prob_single']):.4f}\n"
                f"- Total uncertainty (entropy): {float(result['total_entropy']):.4f}\n"
                f"- Aleatoric part (noise in the data): {float(result['aleatoric']):.4f}\n"
                f"- Epistemic part (model does not know): {float(result['epistemic']):.4f}\n"
                "\nThe aleatoric / epistemic split is an approximation."
            )
        st.subheader("Signal facts (NASA)")
        st.markdown(
            f"- Star (kepid): {int(facts['kepid'])}\n"
            f"- Period: {facts['koi_period']:.3f} days\n"
            f"- Dip length: {facts['koi_duration']:.2f} hours\n"
            f"- Dip depth: {facts['koi_depth']:.0f} ppm\n"
            f"- Signal-to-noise: {facts['koi_model_snr']:.1f}\n"
            f"- Eclipsing-binary flag: {'yes' if facts['koi_fpflag_ss'] == 1 else 'no'}"
        )

    with right:
        n_global, n_local = len(data.global_view[row]), len(data.local_view[row])
        global_x = (np.arange(n_global) + 0.5) / n_global - 0.5
        half = preprocess.LOCAL_WIDTH_DURATIONS / 2
        local_x = np.linspace(-half, half, n_local)
        st.altair_chart(
            view_chart(
                global_x,
                data.global_view[row],
                "orbit phase (0 = dip)",
                "Global view: the whole orbit",
            ),
            width="stretch",
        )
        st.altair_chart(
            view_chart(
                local_x,
                data.local_view[row],
                "time from dip centre (dip lengths)",
                "Local view: zoom on the dip",
            ),
            width="stretch",
        )
        st.altair_chart(runs_histogram(runs[:, row]), width="stretch")


def page_uncertainty() -> None:
    st.header("Uncertainty (MC Dropout, validation set)")
    mc = read_json("mc_info.json")
    table = read_csv("mc_val_predictions.csv")
    table["truth"] = np.where(table["label"] == 1, "planet", "not a planet")

    cards = st.columns(3)
    cards[0].metric("Runs per signal (T)", mc["chosen_runs"], border=True)
    cards[1].metric("Median spread", f"{mc['checks']['mc_std_median']:.4f}", border=True)
    cards[2].metric("Largest spread", f"{mc['checks']['mc_std_max']:.3f}", border=True)

    scatter = (
        alt.Chart(table, title="Answer vs uncertainty for each validation signal")
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X(
                "prob_mc_mean:Q",
                title="planet probability (MC mean)",
                scale=alt.Scale(domain=PROBABILITY_AXIS),
            ),
            y=alt.Y("mc_std:Q", title="uncertainty (spread of the runs)"),
            color=label_color(),
            tooltip=[
                "kepoi_name",
                "truth",
                alt.Tooltip("prob_mc_mean:Q", format=".3f"),
                alt.Tooltip("mc_std:Q", format=".4f"),
            ],
        )
        .interactive()
    )
    st.altair_chart(scatter, width="stretch")
    st.caption(
        "Most signals sit at 0 or 1 with almost no spread: the model is very confident. "
        "The few points higher up are the cases it is unsure about."
    )

    st.subheader("How many runs are enough? (choosing T)")
    convergence = (
        pd.DataFrame(mc["convergence"])
        .rename(columns={"max_mean_diff": "answer (mean)", "max_std_diff": "uncertainty (spread)"})
        .melt(id_vars="n_runs", var_name="measure", value_name="largest difference")
    )
    line = (
        alt.Chart(convergence)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_runs:Q", title="number of runs T", scale=alt.Scale(type="log")),
            y=alt.Y("largest difference:Q", title=f"difference from {mc['reference_runs']} runs"),
            color=alt.Color("measure:N", title=None),
        )
    )
    limit = (
        alt.Chart(pd.DataFrame({"y": [mc["tolerance"]]}))
        .mark_rule(strokeDash=[4, 4], color="gray")
        .encode(y="y:Q")
    )
    st.altair_chart(line + limit, width="stretch")
    st.caption(
        f"T = {mc['chosen_runs']} is the smallest tested T within {mc['tolerance']} "
        "of the 1,000-run reference."
    )

    st.subheader("Where does the uncertainty come from? (top 10)")
    top = table.nlargest(10, "total_entropy")[["kepoi_name", "aleatoric", "epistemic"]]
    parts = top.melt(id_vars="kepoi_name", var_name="part", value_name="entropy")
    bars = (
        alt.Chart(parts)
        .mark_bar()
        .encode(
            y=alt.Y("kepoi_name:N", sort="-x", title=None),
            x=alt.X("entropy:Q", stack=True, title="uncertainty (entropy)"),
            color=alt.Color("part:N", title=None),
        )
    )
    st.altair_chart(bars, width="stretch")
    st.caption(
        "Aleatoric = noise in the data. Epistemic = the model does not know enough. "
        "This split is an approximation."
    )


def page_follow_up() -> None:
    st.header("Follow-up list (most uncertain first)")
    st.caption(
        "Signals are ranked by total uncertainty. In real use, astronomers would check the "
        "top of this list first with other telescopes."
    )
    table = read_csv("follow_up_val.csv")
    table["NASA answer"] = table["label"].map({1: "planet", 0: "not a planet"})
    shown = table[
        [
            "follow_up_rank",
            "kepoi_name",
            "NASA answer",
            "prob_mc_mean",
            "mc_std",
            "total_entropy",
            "koi_model_snr",
            "koi_depth",
            "koi_fpflag_ss",
        ]
    ].rename(
        columns={
            "follow_up_rank": "rank",
            "kepoi_name": "signal",
            "prob_mc_mean": "planet probability",
            "mc_std": "spread",
            "total_entropy": "uncertainty",
            "koi_model_snr": "SNR",
            "koi_depth": "depth (ppm)",
            "koi_fpflag_ss": "EB flag",
        }
    )
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        # Tall enough to show every validation signal without scrolling.
        height=TABLE_ROW_PIXELS * (len(shown) + 1) + 3,
        column_config={
            "planet probability": st.column_config.ProgressColumn(
                format="%.3f", min_value=0.0, max_value=1.0
            ),
            "spread": st.column_config.NumberColumn(format="%.4f"),
            "uncertainty": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    top = table.iloc[0]
    st.info(
        f"Rank 1 is {top['kepoi_name']}: NASA says it is not a planet, and it has the "
        "eclipsing-binary flag. The model gave it only "
        f"{top['prob_mc_mean']:.0%} with a large spread, so it was rightly unsure."
    )
    st.download_button(
        "Download the list (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="follow_up_val.csv",
        mime="text/csv",
    )


def page_calibration() -> None:
    st.header("Calibration: is the confidence honest?")
    info = read_json("calibration_val.json")
    versions = {"A: single prediction": "A_single", "B: MC Dropout mean": "B_mc_mean"}

    cards = st.columns(2)
    for column, (title, key) in zip(cards, versions.items(), strict=True):
        result = info["results"][key]
        low, high = result["ece_bootstrap_95"]
        with column.container(border=True):
            st.markdown(f"**{title}**")
            inner = st.columns(3)
            inner[0].metric("ECE", f"{result['ece']:.4f}", help=f"95% range {low}-{high}")
            inner[1].metric("Brier", f"{result['brier']:.4f}")
            inner[2].metric("NLL", f"{result['nll']:.4f}")
    st.caption("Lower is better for all three. ECE uses 10 bins.")

    picked = st.radio("Reliability diagram for", list(versions), horizontal=True)
    bins = pd.DataFrame(info["results"][versions[picked]]["reliability"])
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", y="y:Q")
    )
    points = (
        alt.Chart(bins)
        .mark_circle(color=PLANET_COLOR, opacity=0.9)
        .encode(
            x=alt.X(
                "mean_predicted:Q",
                title="average predicted probability",
                scale=alt.Scale(domain=PROBABILITY_AXIS),
            ),
            y=alt.Y(
                "fraction_planets:Q",
                title="real fraction of planets",
                scale=alt.Scale(domain=PROBABILITY_AXIS),
            ),
            size=alt.Size("count:Q", title="signals in bin", scale=alt.Scale(range=[80, 900])),
            tooltip=[
                "low",
                "high",
                "count",
                alt.Tooltip("mean_predicted:Q", format=".3f"),
                "fraction_planets",
            ],
        )
    )
    st.altair_chart(
        (diagonal + points).properties(title="Reliability diagram (dashed line = perfect)"),
        width="stretch",
    )

    fit = info["temperature_fit"]
    st.warning(
        "Honest reading: all predictions are near 0 or 1 and all are right, so the bins "
        "between 0.2 and 0.9 are empty. A small ECE here is not yet proof of good calibration."
    )
    st.warning(
        f"Temperature scaling was refused on purpose: {fit['reason']}. "
        "It will be fitted when a larger validation set contains wrong predictions."
    )


def page_evidence() -> None:
    st.header("Research evidence")
    train_info = read_json("train_info.json")
    evaluation = read_json("evaluation_val.json")
    processed = read_json("processed_info.json")

    st.subheader("Training (two-branch 1D-CNN)")
    cards = st.columns(4)
    cards[0].metric("Model parameters", f"{train_info['parameters']:,}", border=True)
    cards[1].metric(
        "Best epoch", f"{train_info['best_epoch']} / {train_info['epochs_run']}", border=True
    )
    cards[2].metric("Training time (CPU)", f"{train_info['train_seconds']} s", border=True)
    cards[3].metric("Seed", train_info["seed"], border=True)
    history = (
        read_csv("train_history.csv")
        .rename(columns={"train_loss": "training", "val_loss": "validation"})
        .melt(id_vars="epoch", var_name="set", value_name="loss")
    )
    curve = (
        alt.Chart(history, title="Loss curve (early stopping on validation loss)")
        .mark_line()
        .encode(x="epoch:Q", y="loss:Q", color=alt.Color("set:N", title=None))
        .interactive()
    )
    st.altair_chart(curve, width="stretch")

    st.subheader("Validation scores")
    scores = pd.DataFrame(
        {
            name: {key: evaluation["scores"][key][name] for key in ("A_single", "B_mc_mean")}
            for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
        }
    ).T.rename(columns={"A_single": "single prediction", "B_mc_mean": "MC Dropout mean"})
    left, right = st.columns(2)
    left.dataframe(scores.style.format("{:.2f}"), width="stretch")
    matrix = evaluation["scores"]["B_mc_mean"]["confusion_matrix"]
    right.dataframe(
        pd.DataFrame(
            {
                "model: planet": [matrix["tp"], matrix["fp"]],
                "model: not a planet": [matrix["fn"], matrix["tn"]],
            },
            index=["NASA: planet", "NASA: not a planet"],
        ),
        width="stretch",
    )

    st.subheader("Data")
    split_info = read_json("split_info.json")["counts"]
    st.dataframe(
        pd.DataFrame(split_info)
        .T.rename_axis("split")
        .rename(columns={"planet_share": "planet share"}),
        width="stretch",
    )
    st.caption(
        f"{processed['signals']} signals found, {processed['excluded']} excluded with a recorded "
        "reason. Split by star, so no star is in two sets (no data leakage)."
    )

    st.subheader("Quality and honesty checks")
    st.markdown(
        "- Automated tests (pytest) for every step, with fake data and no internet\n"
        "- Same seed (42) gives byte-identical results; exact package versions in a lock file\n"
        "- Test set locked in code; this dashboard cannot open it\n"
        '- "Too good to be true" checks: near-duplicates, shuffled labels, '
        "single-number shortcut, sample size"
    )
    st.subheader("Limitations")
    st.markdown(
        "- Pilot only: 150 stars, 31 validation signals, so the true error rate could be up "
        "to about 10%\n"
        "- Kepler only (TESS planned); CNN only (LSTM planned)\n"
        "- Classifies known signals; it does not search for new ones\n"
        "- Final numbers will come from the locked test set at the end"
    )


# ---------- layout ----------


def sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🪐 Exoplanet UQ")
        st.caption("J26-IT-449 · Component 3 · K.H.S.D. Sewwandi")
        st.divider()
        page = st.radio("Navigation", PAGES, key="nav")
        st.divider()
        st.caption(
            "PP1 research demo · Kepler pilot (150 stars) · runs locally · "
            "test set locked · not a deployed service"
        )
    return page


def main() -> None:
    st.set_page_config(page_title="Exoplanet UQ", page_icon="🪐", layout="wide")
    check_files()
    page = sidebar()
    st.title("Bayesian Uncertainty Quantification for Exoplanet Detection")
    st.caption("1D-CNN classification · Monte Carlo Dropout · Calibration · Kepler light curves")
    handlers = {
        "Overview": page_overview,
        "Signal Explorer": page_signal_explorer,
        "Uncertainty": page_uncertainty,
        "Follow-up List": page_follow_up,
        "Calibration": page_calibration,
        "Research Evidence": page_evidence,
    }
    handlers[page]()


main()

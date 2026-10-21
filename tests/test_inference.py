"""Tests for the dashboard's prediction helpers (models/inference.py).

Most tests use the real model class with random knobs and tiny fake inputs.
One test uses the committed model and views to check that the dashboard
shows exactly the numbers saved by Step 6; it is skipped if they are missing.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from exoplanet_uq.config import DATA_DIR, METADATA_DIR, PROCESSED_DATA_DIR
from exoplanet_uq.models import cnn, inference
from exoplanet_uq.models.data import SplitData

MODEL_PATH = DATA_DIR / "models" / "cnn_pilot.pt"
VIEWS_PATH = PROCESSED_DATA_DIR / "views.npz"
SPLIT_PATH = METADATA_DIR / "split.csv"
LABELLED_PATH = METADATA_DIR / "koi_labelled.csv"


def fake_data(n: int = 4, seed: int = 0) -> SplitData:
    rng = np.random.default_rng(seed)
    return SplitData(
        names=np.array([f"K{i:05d}.01" for i in range(n)]),
        global_view=rng.normal(0, 0.3, (n, cnn.GLOBAL_LENGTH)).astype(np.float32),
        local_view=rng.normal(0, 0.3, (n, cnn.LOCAL_LENGTH)).astype(np.float32),
        labels=np.array([i % 2 for i in range(n)], dtype=np.float32),
    )


def fresh_model() -> cnn.TwoBranchCNN:
    torch.manual_seed(0)
    return cnn.TwoBranchCNN()


# ---------- the test lock ----------


@pytest.mark.parametrize("split_name", ["test", "TEST", "", "../test", "everything"])
def test_dashboard_refuses_the_test_set_and_unknown_splits(split_name):
    # The paths are never read: the name is refused before any file is opened.
    with pytest.raises(inference.InferenceError, match="locked"):
        inference.load_dashboard_split(split_name, VIEWS_PATH, SPLIT_PATH, LABELLED_PATH)


def test_only_train_and_val_are_allowed():
    assert set(inference.DASHBOARD_SPLITS) == {"train", "val"}


# ---------- loading the model ----------


def test_missing_model_file_gives_a_clear_error(tmp_path):
    with pytest.raises(inference.InferenceError, match="not found"):
        inference.load_model(tmp_path / "no_model.pt")


def test_saved_knobs_load_back_in_prediction_mode(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(fresh_model().state_dict(), path)
    model = inference.load_model(path)
    assert not model.training  # dropout off until MC Dropout switches it on


# ---------- predictions ----------


def test_predict_split_gives_one_row_per_signal_with_valid_numbers():
    data = fake_data(5)
    table, runs = inference.predict_split(fresh_model(), data, n_runs=10)
    assert runs.shape == (10, 5)
    assert list(table["kepoi_name"]) == list(data.names)
    for column in ("prob_single", "prob_mc_mean"):
        assert table[column].between(0, 1).all()
    assert (table["mc_std"] >= 0).all()
    assert (table["epistemic"] >= 0).all()


def test_same_seed_gives_the_same_answers():
    data = fake_data(3)
    first, _ = inference.predict_split(fresh_model(), data, n_runs=8)
    second, _ = inference.predict_split(fresh_model(), data, n_runs=8)
    pd.testing.assert_frame_equal(first, second)


def test_signal_index_finds_known_names_and_refuses_unknown_ones():
    data = fake_data(3)
    assert inference.signal_index(data, "K00002.01") == 2
    with pytest.raises(inference.InferenceError):
        inference.signal_index(data, "K99999.01")


@pytest.mark.parametrize(
    ("spread", "words"),
    [(0.0, "very sure"), (0.019, "very sure"), (0.05, "fairly sure"), (0.13, "unsure")],
)
def test_confidence_words(spread, words):
    assert inference.confidence_words(spread).startswith(words)


# ---------- one source of truth with Step 6 ----------


@pytest.mark.skipif(
    not (MODEL_PATH.is_file() and VIEWS_PATH.is_file()),
    reason="needs the committed model and views",
)
def test_live_val_predictions_match_the_saved_step_6_table():
    model = inference.load_model(MODEL_PATH)
    val = inference.load_dashboard_split("val", VIEWS_PATH, SPLIT_PATH, LABELLED_PATH)
    table, _ = inference.predict_split(model, val, n_runs=100)
    saved = pd.read_csv(METADATA_DIR / "mc_val_predictions.csv").set_index("kepoi_name")
    saved = saved.loc[table["kepoi_name"]]
    for column in ("prob_single", "prob_mc_mean", "mc_std", "total_entropy"):
        # Tiny float differences between CPUs are fine; real differences are not.
        np.testing.assert_allclose(table[column], saved[column], atol=1e-6)

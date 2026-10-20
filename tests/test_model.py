"""Tests for the model, its data loading and its training.

They use small fake data, so they need no real light curves and run in seconds.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from exoplanet_uq.data import split as sp
from exoplanet_uq.models import cnn, train
from exoplanet_uq.models.data import DataError, SplitData, load_split, scale_view

# ---------- fake data ----------


def fake_views(label: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A planet: one dip in the middle. A non-planet: the same dip plus a second
    dip at phase 0.5 (like an eclipsing binary's secondary eclipse)."""
    phase = np.linspace(-0.5, 0.5, cnn.GLOBAL_LENGTH)
    global_view = 1.0 - 0.01 * (np.abs(phase) < 0.01)
    if label == 0:
        global_view -= 0.006 * (np.abs(phase) > 0.49)
    local_view = 1.0 - 0.01 * (np.abs(np.linspace(-2, 2, cnn.LOCAL_LENGTH)) < 0.5)
    global_view = global_view + rng.normal(0, 0.0005, cnn.GLOBAL_LENGTH)
    local_view = local_view + rng.normal(0, 0.0005, cnn.LOCAL_LENGTH)
    return global_view.astype(np.float32), local_view.astype(np.float32)


def fake_split(n: int, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)], dtype=np.float32)
    pairs = [fake_views(int(label), rng) for label in labels]
    return SplitData(
        names=np.array([f"K{i:05d}.01" for i in range(n)]),
        global_view=np.stack([scale_view(g) for g, _ in pairs]),
        local_view=np.stack([scale_view(loc) for _, loc in pairs]),
        labels=labels,
    )


@pytest.fixture
def files(tmp_path):
    """Fake views.npz, split.csv and koi_labelled.csv on disk."""
    rng = np.random.default_rng(0)
    names = [f"K{i:05d}.01" for i in range(6)]
    labels = [1, 0, 1, 0, 1, 0]
    splits = ["train", "train", "val", "val", "test", "test"]
    pairs = [fake_views(label, rng) for label in labels]
    np.savez(
        tmp_path / "views.npz",
        kepoi_name=np.array(names),
        global_view=np.stack([g for g, _ in pairs]),
        local_view=np.stack([loc for _, loc in pairs]),
    )
    table = pd.DataFrame({"kepoi_name": names, "kepid": range(6), "label": labels, "split": splits})
    table.to_csv(tmp_path / "split.csv", index=False)
    table[["kepoi_name", "kepid", "label"]].to_csv(tmp_path / "labelled.csv", index=False)
    return tmp_path / "views.npz", tmp_path / "split.csv", tmp_path / "labelled.csv"


# ---------- scale_view ----------


def test_scale_view_puts_median_at_0_and_dip_at_minus_1():
    view = np.array([1.0, 1.0, 0.998, 1.0, 1.0])
    scaled = scale_view(view)
    assert np.median(scaled) == pytest.approx(0.0)
    assert scaled.min() == pytest.approx(-1.0)
    assert scaled.dtype == np.float32


def test_scale_view_handles_a_flat_view_without_dividing_by_zero():
    scaled = scale_view(np.ones(5))
    assert np.all(scaled == 0.0)


# ---------- load_split ----------


def test_load_split_returns_the_right_rows_and_labels(files):
    data = load_split(*files, sp.TRAIN)
    assert data.names.tolist() == ["K00000.01", "K00001.01"]
    assert data.labels.tolist() == [1.0, 0.0]
    assert data.global_view.shape == (2, cnn.GLOBAL_LENGTH)
    assert data.local_view.shape == (2, cnn.LOCAL_LENGTH)
    assert data.global_view.min() == pytest.approx(-1.0)  # scaled


def test_the_test_set_is_locked(files):
    with pytest.raises(DataError, match="locked"):
        load_split(*files, sp.TEST)


def test_the_test_set_opens_only_on_purpose(files):
    assert len(load_split(*files, sp.TEST, allow_test=True)) == 2


def test_unknown_split_name_is_rejected(files):
    with pytest.raises(DataError, match="Unknown"):
        load_split(*files, "holdout")


def test_a_signal_without_views_is_reported(files):
    views, split, labelled = files
    table = pd.read_csv(split)
    table.loc[0, "kepoi_name"] = "K99999.01"
    table.to_csv(split, index=False)
    labels = pd.read_csv(labelled)
    labels.loc[0, "kepoi_name"] = "K99999.01"
    labels.to_csv(labelled, index=False)
    with pytest.raises(DataError, match="No views"):
        load_split(views, split, labelled, sp.TRAIN)


# ---------- the model ----------


def test_model_gives_one_finite_logit_per_signal():
    model = cnn.TwoBranchCNN()
    data = fake_split(3, seed=0)
    out = model(torch.from_numpy(data.global_view), torch.from_numpy(data.local_view))
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


def test_model_has_exactly_one_dropout_layer_with_the_planned_rate():
    dropouts = [m for m in cnn.TwoBranchCNN().modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) == 1
    assert dropouts[0].p == cnn.DROPOUT_RATE


def test_model_is_small():
    assert cnn.count_parameters(cnn.TwoBranchCNN()) < 300_000


def test_weights_save_and_load_safely(tmp_path):
    model = cnn.TwoBranchCNN()
    torch.save(model.state_dict(), tmp_path / "m.pt")
    again = cnn.TwoBranchCNN()
    again.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
    for a, b in zip(model.state_dict().values(), again.state_dict().values(), strict=True):
        assert torch.equal(a, b)


# ---------- scores ----------


def test_roc_auc_known_values():
    labels = np.array([1, 1, 0, 0])
    assert train.roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert train.roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert train.roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert train.roc_auc(labels, np.array([0.9, 0.3, 0.5, 0.1])) == 0.75


def test_roc_auc_needs_both_classes():
    with pytest.raises(ValueError):
        train.roc_auc(np.array([1, 1]), np.array([0.2, 0.9]))


def test_accuracy_and_class_weight():
    assert train.accuracy(np.array([1, 0, 1]), np.array([0.9, 0.4, 0.2])) == pytest.approx(2 / 3)
    assert train.class_weight(np.array([1, 0, 0, 0])) == 3.0
    with pytest.raises(ValueError):
        train.class_weight(np.array([0, 0]))


# ---------- training ----------


def test_the_model_learns_an_easy_fake_task():
    train.set_seed(0)
    model = cnn.TwoBranchCNN()
    train_data, val_data = fake_split(60, seed=1), fake_split(20, seed=2)
    train.train_model(model, train_data, val_data, pos_weight=1.0, max_epochs=40, patience=15)
    assert train.accuracy(val_data.labels, train.predict_proba(model, val_data)) >= 0.9


def test_the_same_seed_gives_the_same_training():
    data, val = fake_split(16, seed=1), fake_split(8, seed=2)
    runs = []
    for _ in range(2):
        train.set_seed(5)
        model = cnn.TwoBranchCNN()
        result = train.train_model(model, data, val, pos_weight=1.0, max_epochs=3, patience=5)
        runs.append((result.history, train.predict_proba(model, val)))
    assert runs[0][0] == runs[1][0]
    np.testing.assert_array_equal(runs[0][1], runs[1][1])


def test_early_stopping_keeps_the_best_epoch():
    train.set_seed(0)
    model = cnn.TwoBranchCNN()
    data = fake_split(16, seed=1)
    # Random labels: nothing real to learn, so val loss soon stops improving.
    noisy = SplitData(data.names, data.global_view, data.local_view, data.labels[::-1].copy())
    result = train.train_model(model, data, noisy, pos_weight=1.0, max_epochs=200, patience=5)
    best = min(row["val_loss"] for row in result.history)
    assert len(result.history) < 200  # it stopped early
    assert result.best_val_loss == pytest.approx(best)
    assert len(result.history) - result.best_epoch == 5  # stopped `patience` epochs later
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    assert train.mean_loss(model, noisy, loss_fn) == pytest.approx(best, rel=1e-5)

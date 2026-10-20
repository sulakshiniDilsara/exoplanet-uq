"""Tests for MC Dropout and the uncertainty numbers.

They use the real model class with random knobs and tiny fake inputs,
so they need no trained model and no real data.
"""

import numpy as np
import pytest
import torch
from torch import nn

from exoplanet_uq.models import cnn, mc_dropout
from exoplanet_uq.models.data import SplitData


def fake_data(n: int = 4, seed: int = 0) -> SplitData:
    rng = np.random.default_rng(seed)
    return SplitData(
        names=np.array([f"K{i:05d}.01" for i in range(n)]),
        global_view=rng.normal(0, 0.3, (n, cnn.GLOBAL_LENGTH)).astype(np.float32),
        local_view=rng.normal(0, 0.3, (n, cnn.LOCAL_LENGTH)).astype(np.float32),
        labels=np.array([i % 2 for i in range(n)], dtype=np.float32),
    )


def fresh_model(dropout_rate: float = cnn.DROPOUT_RATE) -> cnn.TwoBranchCNN:
    torch.manual_seed(0)
    return cnn.TwoBranchCNN(dropout_rate=dropout_rate)


# ---------- binary_entropy and uncertainty (lesson 4, section 5.3) ----------


def test_entropy_is_highest_at_one_half_and_near_zero_when_sure():
    assert mc_dropout.binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))  # 0.693
    assert mc_dropout.binary_entropy(np.array([0.99]))[0] == pytest.approx(0.056, abs=1e-3)
    assert mc_dropout.binary_entropy(np.array([0.0, 1.0])).max() < 1e-5  # no log(0) crash


def test_lesson_case_1_every_run_says_one_half_is_all_aleatoric():
    runs = np.array([[0.5], [0.5], [0.5], [0.5]])
    u = mc_dropout.uncertainty(runs)
    assert u["prob_mc_mean"][0] == pytest.approx(0.5)
    assert u["total_entropy"][0] == pytest.approx(0.693, abs=1e-3)
    assert u["aleatoric"][0] == pytest.approx(0.693, abs=1e-3)
    assert u["epistemic"][0] == pytest.approx(0.0, abs=1e-9)
    assert u["mc_std"][0] == pytest.approx(0.0)


def test_lesson_case_2_runs_disagree_is_mostly_epistemic():
    runs = np.array([[0.99], [0.01], [0.99], [0.01]])
    u = mc_dropout.uncertainty(runs)
    assert u["prob_mc_mean"][0] == pytest.approx(0.5)
    assert u["total_entropy"][0] == pytest.approx(0.693, abs=1e-3)
    assert u["aleatoric"][0] == pytest.approx(0.056, abs=1e-3)
    assert u["epistemic"][0] == pytest.approx(0.637, abs=1e-3)
    assert u["mc_std"][0] == pytest.approx(0.49)


def test_the_two_parts_always_add_up_to_the_total_and_are_never_negative():
    runs = np.random.default_rng(3).uniform(0, 1, (50, 20))
    u = mc_dropout.uncertainty(runs)
    np.testing.assert_allclose(u["aleatoric"] + u["epistemic"], u["total_entropy"], atol=1e-9)
    assert (u["epistemic"] >= 0).all() and (u["aleatoric"] >= 0).all()


# ---------- enable_mc_dropout and mc_predict ----------


def test_only_dropout_layers_are_switched_on():
    model = fresh_model()
    assert mc_dropout.enable_mc_dropout(model) == 1
    for layer in model.modules():
        if layer is model:
            continue
        expected = isinstance(layer, nn.Dropout)
        assert layer.training == expected, type(layer).__name__


def test_mc_runs_differ_but_the_model_ends_in_prediction_mode():
    model = fresh_model()
    runs = mc_dropout.mc_predict(model, fake_data(), n_runs=20)
    assert runs.shape == (20, 4)
    assert ((runs >= 0) & (runs <= 1)).all()
    assert runs.std(axis=0).max() > 0  # dropout really changes the answers
    assert not any(layer.training for layer in model.modules())  # back to eval()


def test_with_dropout_rate_zero_every_run_is_the_same():
    runs = mc_dropout.mc_predict(fresh_model(dropout_rate=0.0), fake_data(), n_runs=10)
    assert runs.std(axis=0).max() == pytest.approx(0.0, abs=1e-7)


def test_the_same_seed_gives_the_same_runs():
    model, data = fresh_model(), fake_data()
    first = mc_dropout.mc_predict(model, data, n_runs=10, seed=7)
    second = mc_dropout.mc_predict(model, data, n_runs=10, seed=7)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, mc_dropout.mc_predict(model, data, n_runs=10, seed=8))


def test_fewer_than_two_runs_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        mc_dropout.mc_predict(fresh_model(), fake_data(), n_runs=1)


def test_a_model_without_dropout_is_refused_and_left_in_prediction_mode():
    class NoDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(cnn.GLOBAL_LENGTH + cnn.LOCAL_LENGTH, 1)

        def forward(self, g, loc):
            return self.layer(torch.cat([g, loc], dim=1)).squeeze(1)

    model = NoDropout()
    model.train()
    with pytest.raises(ValueError, match="no dropout"):
        mc_dropout.mc_predict(model, fake_data(), n_runs=5)
    assert not model.training  # the finally block still ran


# ---------- choose_n_runs ----------


def test_choose_picks_the_smallest_t_that_is_close_enough():
    rng = np.random.default_rng(0)
    reference = np.clip(rng.normal(0.8, 0.05, (1000, 30)), 0, 1)
    choice = mc_dropout.choose_n_runs(reference, candidates=(10, 20, 50, 100), tolerance=0.02)
    assert choice.converged
    assert choice.n_runs in (10, 20, 50, 100)
    rows = {row["n_runs"]: row for row in choice.table}
    picked = rows[choice.n_runs]
    assert picked["max_mean_diff"] <= 0.02 and picked["max_std_diff"] <= 0.02
    for smaller in (t for t in rows if t < choice.n_runs):
        row = rows[smaller]
        assert row["max_mean_diff"] > 0.02 or row["max_std_diff"] > 0.02


def test_choose_says_so_when_no_candidate_is_close_enough():
    rng = np.random.default_rng(0)
    reference = rng.uniform(0, 1, (1000, 30))  # very noisy runs
    choice = mc_dropout.choose_n_runs(reference, candidates=(10, 20), tolerance=0.001)
    assert not choice.converged
    assert choice.n_runs == 20  # falls back to the largest candidate


def test_choose_refuses_a_candidate_bigger_than_the_reference():
    with pytest.raises(ValueError):
        mc_dropout.choose_n_runs(np.zeros((50, 3)), candidates=(10, 100))


# ---------- spearman ----------


def test_spearman_known_values():
    assert mc_dropout.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert mc_dropout.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert mc_dropout.spearman([1, 2, 3, 4], [1, 4, 9, 100]) == pytest.approx(1.0)  # order only

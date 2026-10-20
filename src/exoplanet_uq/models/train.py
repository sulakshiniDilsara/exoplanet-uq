"""Train the CNN with early stopping, and score it.

Training loop (one epoch):
    shuffle the training signals -> for each batch of 32:
        model guesses -> loss (how wrong) -> Adam turns the knobs a little
    then score the validation set. If the validation loss has not improved
    for PATIENCE epochs, stop, and go back to the best epoch's knobs.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from exoplanet_uq.models.data import SplitData

SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20


@dataclass
class TrainResult:
    history: list[dict]  # one row per epoch: epoch, train_loss, val_loss
    best_epoch: int
    best_val_loss: float


def set_seed(seed: int = SEED) -> None:
    """Make every random choice repeatable: Python, numpy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def as_tensors(data: SplitData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(data.global_view),
        torch.from_numpy(data.local_view),
        torch.from_numpy(data.labels),
    )


def class_weight(labels: np.ndarray) -> float:
    """How much more a missed planet costs: (number of non-planets) / (number of planets).

    Always computed from the TRAINING labels only.
    """
    n_planets = float(np.sum(labels == 1))
    if n_planets == 0:
        raise ValueError("The training set has no planets.")
    return float(np.sum(labels == 0)) / n_planets


def mean_loss(model: nn.Module, data: SplitData, loss_fn: nn.Module) -> float:
    """Average loss over a whole split, in prediction mode (dropout off), no learning."""
    model.eval()
    global_view, local_view, labels = as_tensors(data)
    with torch.no_grad():
        return loss_fn(model(global_view, local_view), labels).item()


def train_model(
    model: nn.Module,
    train: SplitData,
    val: SplitData,
    pos_weight: float,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
) -> TrainResult:
    """Train in place. At the end the model holds the knobs of the best epoch."""
    # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one, more stable step.
    # pos_weight makes a missed planet cost more (class weighting).
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    shuffler = torch.Generator().manual_seed(seed)
    global_view, local_view, labels = as_tensors(train)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, max_epochs + 1):
        model.train()  # dropout ON while learning
        order = torch.randperm(len(train), generator=shuffler)
        total = 0.0
        for start in range(0, len(train), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(global_view[batch], local_view[batch]), labels[batch])
            loss.backward()  # work out which way to turn each knob
            optimizer.step()  # turn the knobs a little
            total += loss.item() * len(batch)  # .item(): a plain number, no gradient link

        val_loss = mean_loss(model, val, loss_fn)
        history.append({"epoch": epoch, "train_loss": total / len(train), "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break  # early stopping: no improvement for `patience` epochs

    model.load_state_dict(best_state)
    return TrainResult(history, best_epoch, best_val_loss)


def predict_proba(model: nn.Module, data: SplitData) -> np.ndarray:
    """Planet probability for every signal, with dropout OFF (one normal prediction)."""
    model.eval()
    global_view, local_view, _ = as_tensors(data)
    with torch.no_grad():
        return torch.sigmoid(model(global_view, local_view)).numpy()


def accuracy(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((probabilities >= threshold) == (labels == 1)))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Chance that a random planet gets a higher score than a random non-planet.

    1.0 = perfect ranking, 0.5 = coin toss. Ties count as half.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("ROC-AUC needs at least one planet and one non-planet.")
    # Compare every planet with every non-planet (small sets, so this is fine).
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positives) * len(negatives)))

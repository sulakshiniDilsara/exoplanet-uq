"""A small two-branch 1D-CNN: one branch for the global view, one for the local view.

    global (2001) -> 3 conv blocks --\
                                      >-- join -> FC 64 -> ReLU -> Dropout -> FC 1
    local  (201)  -> 2 conv blocks --/

The model returns a logit (any number). sigmoid(logit) is the planet
probability. Training uses the logit directly, which is more stable.

The model is small on purpose: the pilot has only 123 training signals,
and a big model would memorise them (overfitting).
"""

from __future__ import annotations

import torch
from torch import nn

GLOBAL_LENGTH = 2001
LOCAL_LENGTH = 201
KERNEL_SIZE = 5
POOL_SIZE = 4
HIDDEN_UNITS = 64
DROPOUT_RATE = 0.3


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Convolution -> ReLU -> max pooling. "padding" keeps the length before pooling."""
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, KERNEL_SIZE, padding=KERNEL_SIZE // 2),
        nn.ReLU(),
        nn.MaxPool1d(POOL_SIZE),
    )


class TwoBranchCNN(nn.Module):
    def __init__(self, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        self.global_branch = nn.Sequential(
            conv_block(1, 16), conv_block(16, 32), conv_block(32, 64), nn.Flatten()
        )
        self.local_branch = nn.Sequential(conv_block(1, 16), conv_block(16, 32), nn.Flatten())

        # Work out how many numbers each branch gives, by running a dummy input once.
        with torch.no_grad():
            n_global = self.global_branch(torch.zeros(1, 1, GLOBAL_LENGTH)).shape[1]
            n_local = self.local_branch(torch.zeros(1, 1, LOCAL_LENGTH)).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_global + n_local, HIDDEN_UNITS),
            nn.ReLU(),
            # The only dropout layer. Step 6 (MC Dropout) switches it on at prediction time.
            nn.Dropout(dropout_rate),
            nn.Linear(HIDDEN_UNITS, 1),
        )

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        """global_view: (batch, 2001), local_view: (batch, 201) -> logits: (batch,)"""
        # Conv1d wants (batch, channels, length); we have one channel.
        g = self.global_branch(global_view.unsqueeze(1))
        loc = self.local_branch(local_view.unsqueeze(1))
        return self.head(torch.cat([g, loc], dim=1)).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """How many numbers ("knobs") the model learns."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

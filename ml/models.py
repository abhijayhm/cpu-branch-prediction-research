"""Tiny ANBA networks. PyTorch is train/export only."""

from __future__ import annotations

import torch
from torch import nn


class Dense16(nn.Module):
    name = "nn_a_dense16"
    arch_id = 0

    def __init__(self, n_in: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Dense32_8(nn.Module):
    name = "nn_b_dense32_8"
    arch_id = 1

    def __init__(self, n_in: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.Tanh(), nn.Linear(32, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PerceptronLike(nn.Module):
    name = "nn_c_perceptron"
    arch_id = 2

    def __init__(self, n_in: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


REGISTRY = {
    "nn_a": Dense16,
    "nn_b": Dense32_8,
    "nn_c": PerceptronLike,
    "dense16": Dense16,
    "dense32_8": Dense32_8,
    "perceptron": PerceptronLike,
}

'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
N_ACTIONS = 12 # fixed for Rubik's Cube (6 faces * 3 turns)

class MLPQNet(nn.Module):
    """
    Q-network for one-hot cubie encoding (dim=256).
    Plug this after CubieEncoder (one-hot).
    """
    def __init__(self, in_dim: int = 256, hidden: int = 512, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, int(0.5*hidden)), nn.ReLU(inplace=True),
            nn.Linear(int(0.5*hidden), int(0.1*hidden)), nn.ReLU(inplace=True),
            nn.Linear(int(0.1*hidden), n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 256] float32
        returns: [B, n_actions] Q-values
        """
        return self.net(x)
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
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 256] float32
        returns: [B, n_actions] Q-values
        """
        return self.net(x)

    @torch.no_grad()
    def act(self, x: torch.Tensor, epsilon: float = 0.0, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ε-greedy action selection.
        x: [B, 256]
        action_mask: optional [B, n_actions] with 0 for allowed, -inf for banned (or bool: True=allow, False=ban).
        """
        if epsilon > 0.0 and torch.rand(()) < epsilon:
            # random allowed action
            if action_mask is None:
                return torch.randint(0, N_ACTIONS, (x.size(0),), device=x.device)
            else:
                if action_mask.dtype == torch.bool:
                    probs = action_mask.float()
                else:
                    probs = torch.isfinite(action_mask).float()
                probs = probs / probs.sum(dim=-1, keepdim=True)
                return torch.multinomial(probs, 1).squeeze(1)
        q = self.forward(x)
        if action_mask is not None:
            if action_mask.dtype == torch.bool:
                # mask False (disallowed) with -inf
                mask = torch.where(action_mask, torch.zeros_like(q), torch.full_like(q, float("-inf")))
                q = q + mask
            else:
                q = q + action_mask  # expect 0 for allowed, -inf for banned
        return q.argmax(dim=-1)
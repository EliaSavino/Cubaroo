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
    Feed-forward Q-network for one-hot cubie encoding (dim = 256).

    This network is intended for use with a one-hot cubie encoder
    (e.g., :class:`CubieEncoder`) that produces a 256-dimensional float32 vector.
    It outputs per-action Q-values suitable for ε-greedy or greedy action
    selection.

    Parameters
    ----------
    in_dim : int, default=256
        Input feature dimension (256 for the one-hot cubie encoder).
    hidden : int, default=512
        Width of the first hidden layer. Subsequent layers scale down to
        ``0.5 * hidden`` and ``0.1 * hidden``.
    n_actions : int, default=N_ACTIONS (12)
        Number of discrete actions (Rubik's Cube here uses 12 moves: each of
        the 6 faces rotated clockwise or counter-clockwise).

    Notes
    -----
    - Activation: ReLU (in-place) after each hidden linear layer.
    - Output: raw Q-values (no activation) of shape ``[B, n_actions]``.
    - Minimal checks are performed to keep behavior predictable; callers
      should ensure inputs are already ``float32`` tensors on the correct device.

    Examples
    --------
    >>> net = MLPQNet()
    >>> x = torch.zeros(32, 256)  # batch of 32 observations
    >>> q = net(x)
    >>> q.shape
    torch.Size([32, 12])
    """

    def __init__(self, in_dim: int = 256, hidden: int = 512, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, int(0.5 * hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(0.5 * hidden), int(0.1 * hidden)),
            nn.ReLU(inplace=True),
            nn.Linear(int(0.1 * hidden), n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-action Q-values.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, in_dim]`` (float32). ``B`` is batch size.

        Returns
        -------
        torch.Tensor
            Q-values of shape ``[B, n_actions]`` (same device/dtype as input).

        Raises
        ------
        ValueError
            If the input's last dimension does not match ``in_dim``.
        """
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected input with last dim = {self.in_dim}, got {x.shape[-1]}")
        return self.net(x)
'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

from __future__ import annotations
from typing import Optional, Any, Protocol, runtime_checkable
import numpy as np
import torch

from src.models.adaptors import QValueProvider

@runtime_checkable
class BasePolicy(Protocol):
    """
    Minimal interface for action-selection policies.

    Implementations must provide an :meth:`act` method returning one action index
    per batch element.

    Methods
    -------
    act(X, epsilon=0.0, action_mask=None) -> np.ndarray
        Select actions for a batch of inputs. See :meth:`act` for details.
    """

    def act(
        self,
        X: Any,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Select actions for a batch of inputs.

        Parameters
        ----------
        X : Any
            Model-specific input batch (e.g., np.ndarray or torch.Tensor).
        epsilon : float, default=0.0
            Exploration probability in [0, 1]. With probability ``epsilon``,
            choose a random allowed action; otherwise choose the greedy action.
        action_mask : Optional[np.ndarray], default=None
            Optional mask of shape ``[B, A]``. Two forms are supported:
              • **Boolean mask**: True = allowed, False = disallowed.
              • **Additive mask**: 0 for allowed, very negative or ``-np.inf``
                for disallowed (added to Q before argmax).

        Returns
        -------
        np.ndarray
            Array of shape ``[B]`` with selected action indices (dtype ``int64``).
        """
        ...


# ───────────────────────── Framework-agnostic ε-greedy policy ─────────────────────────
class EpsGreedyPolicy(BasePolicy):
    """
    ε-greedy action selector for any :class:`QValueProvider`.

    Supports optional action masks:
      - **Boolean mask**: shape ``[B, A]`` with True = allowed, False = disallowed.
      - **Additive mask**: shape ``[B, A]`` with 0 for allowed and a very negative
        value (e.g., ``-np.inf``) for disallowed; added to Q-values prior to argmax.

    Parameters
    ----------
    model : QValueProvider
        Source of Q-values. Must implement ``q_values(X) -> np.ndarray[B, A]``
        and the attribute ``n_actions``.
    rng : Optional[np.random.Generator], default=None
        Random number generator for exploration. If None, uses ``np.random.default_rng()``.

    Notes
    -----
    - If a row's mask disallows **all** actions, exploration falls back to uniform
      over all actions; exploitation falls back to an unmasked argmax.
    - Inputs are not copied unless necessary; outputs are always ``int64``.
    """

    def __init__(self, model: QValueProvider, rng: Optional[np.random.Generator] = None) -> None:
        self.model = model
        self.rng = rng or np.random.default_rng()
    @torch.no_grad()
    def act(
        self,
        X: Any,
        epsilon: float = 0.0,
        action_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        ε-greedy action selection.

        Parameters
        ----------
        X : Any
            Batch input for the underlying model (model-specific structure).
        epsilon : float, default=0.0
            Probability of selecting a random allowed action. Values outside
            [0, 1] are clipped to the range.
        action_mask : Optional[np.ndarray], default=None
            Either:
              • Boolean mask ``[B, A]`` (True=allowed), or
              • Additive mask ``[B, A]`` (0 for allowed, very negative for banned).

        Returns
        -------
        np.ndarray
            Chosen action indices of shape ``[B]`` with dtype ``int64``.

        Raises
        ------
        ValueError
            If the mask shape does not match the returned Q-values shape.
        """
        # Get Q-values
        device = next(self.model.module.parameters()).device  # type: ignore[attr-defined]
        q = self.model.q_values(X)  # [B, A] on device
        B, A = q.shape

        # normalize mask to additive form
        if action_mask is not None:
            if action_mask.shape != (B, A):
                raise ValueError(f"action_mask must be [B, A]=({B}, {A}), got {tuple(action_mask.shape)}")
            if action_mask.dtype == torch.bool:
                q = q.masked_fill(~action_mask, float("-inf"))
            else:
                q = q + action_mask  # expected 0 or -inf (or very negative)
        # greedy
        greedy = q.argmax(dim=-1)  # [B]

        # exploration flags
        eps = float(max(0.0, min(1.0, epsilon)))
        if eps <= 0.0:
            return greedy

        explore_flags = (torch.rand(B, device=device) < eps)  # [B] bool
        if not explore_flags.any():
            return greedy

        # sample random valid actions uniformly
        if action_mask is None:
            rand_act = torch.randint(A, (B,), device=device)
        else:
            # valid actions per row
            valid = (action_mask.isfinite() if action_mask.dtype.is_floating_point else action_mask).to(q.dtype)
            # fallback: if a row has no valid actions, allow all
            row_sums = valid.sum(dim=1, keepdim=True)
            safe_valid = torch.where(row_sums > 0, valid, torch.ones_like(valid))
            probs = safe_valid / safe_valid.sum(dim=1, keepdim=True)
            rand_act = torch.multinomial(probs, 1).squeeze(1)

        actions = torch.where(explore_flags, rand_act, greedy)
        return actions.long()



'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

from __future__ import annotations
from typing import Optional, Any, Protocol, runtime_checkable
import numpy as np
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
        q = self.model.q_values(X)  # [B, A]
        if q.ndim != 2:
            raise ValueError(f"q_values must return a 2D array [B, A], got shape {q.shape}")
        B, A = q.shape
        if A != self.model.n_actions:
            raise ValueError(
                f"q_values second dimension must equal model.n_actions={self.model.n_actions}, got {A}"
            )

        # Clip epsilon to sane bounds
        eps = float(np.clip(epsilon, 0.0, 1.0))

        # Normalize mask → boolean "allowed" mask
        if action_mask is None:
            allowed = np.ones_like(q, dtype=bool)
            add_mask = None
        else:
            if action_mask.shape != (B, A):
                raise ValueError(
                    f"action_mask must have shape [B, A] = {(B, A)}, got {action_mask.shape}"
                )
            if action_mask.dtype == bool:
                allowed = action_mask
                add_mask = None
            else:
                # Finite entries are allowed; non-finite (e.g., -inf) disallowed.
                allowed = np.isfinite(action_mask)
                add_mask = action_mask  # to be added during exploitation

        # Decide which rows explore
        explore_flags = self.rng.random(B) < eps
        actions = np.empty(B, dtype=np.int64)

        # Exploration: row-wise uniform over allowed; if none allowed → uniform over all actions
        if explore_flags.any():
            allowed_rows = allowed[explore_flags]
            has_any = allowed_rows.any(axis=1, keepdims=True)
            # If a row has no allowed actions, allow all as a fallback
            safe_allowed = np.where(has_any, allowed_rows, True)

            # Convert to per-row categorical distributions
            probs = safe_allowed.astype(float)
            probs /= probs.sum(axis=1, keepdims=True)

            # Sample: inverse-CDF per row
            cum = probs.cumsum(axis=1)
            r = self.rng.random(size=(probs.shape[0], 1))
            sampled = (cum < r).sum(axis=1)
            actions[explore_flags] = sampled

        # Exploitation: argmax respecting mask
        if (~explore_flags).any():
            q_greedy = q[~explore_flags].copy()

            if add_mask is not None:
                # Add additive mask directly (e.g., 0 or -inf)
                q_greedy += add_mask[~explore_flags]
            else:
                # Convert boolean allowed mask → -inf for disallowed
                disallowed = ~allowed[~explore_flags]
                # Avoid modifying in-place where all actions are disallowed; in that
                # pathological case, argmax will pick 0 (consistent fallback).
                q_greedy[disallowed] = -np.inf

            actions[~explore_flags] = np.argmax(q_greedy, axis=1)

        return actions



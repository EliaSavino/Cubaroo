'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

from __future__ import annotations
from typing import Protocol, Optional, Any, Callable
import numpy as np
import torch

N_ACTIONS: int = 12

class QValueProvider(Protocol):
    """
    Minimal interface for Q-value models.

    Implementations must provide:
      - n_actions: int (number of discrete actions)
      - q_values(X) -> np.ndarray of shape [B, n_actions]
    """
    n_actions: int

    def q_values(self, X: Any) -> np.ndarray:
        """
        Compute Q-values for a batch of inputs.

        Parameters
        ----------
        X : Any
            Model-specific input batch.

        Returns
        -------
        np.ndarray
            Array of shape [B, n_actions] with Q-values.
        """
        ...


# -------- Framework-agnostic ε-greedy policy ----------------------------------
class EpsGreedyPolicy:
    """
    ε-greedy action selector for any QValueProvider.

    Supports optional action masks:
      - Bool mask: shape [B, n_actions], True = allowed, False = disallowed
      - Additive mask: shape [B, n_actions], 0 for allowed, -inf (or very negative) for banned
    """

    def __init__(self, model: QValueProvider, rng: Optional[np.random.Generator] = None):
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
            Batch input for the model (model-specific structure).
        epsilon : float
            Probability of taking a random allowed action.
        action_mask : Optional[np.ndarray]
            Either bool mask [B, A] (True=allowed) or additive mask [B, A] (0 or -inf).

        Returns
        -------
        np.ndarray
            Chosen action indices [B], dtype int64.
        """
        q = self.model.q_values(X)  # [B, A]
        B, A = q.shape
        assert A == self.model.n_actions, "q_values must return [B, n_actions]"

        # Normalize mask to bool "allowed" mask
        if action_mask is None:
            allowed = np.ones_like(q, dtype=bool)
        else:
            if action_mask.dtype == bool:
                allowed = action_mask
            else:
                allowed = np.isfinite(action_mask)

        # Sample exploration flags per row
        explore_flags = self.rng.random(B) < epsilon
        actions = np.empty(B, dtype=np.int64)

        # Exploration: uniform over allowed; fallback to all if none allowed
        if explore_flags.any():
            allowed_rows = allowed[explore_flags]
            row_has_any = allowed_rows.any(axis=1, keepdims=True)
            safe_allowed = np.where(row_has_any, allowed_rows, True)  # if none, all allowed
            probs = safe_allowed.astype(float)
            probs /= probs.sum(axis=1, keepdims=True)
            # Multinomial row-wise sampling
            cum = probs.cumsum(axis=1)
            r = self.rng.random(size=(probs.shape[0], 1))
            sampled = (cum < r).sum(axis=1)
            actions[explore_flags] = sampled

        # Exploitation: argmax respecting mask via -inf
        if (~explore_flags).any():
            q_greedy = q[~explore_flags].copy()
            if action_mask is not None and action_mask.dtype != bool:
                q_greedy += action_mask[~explore_flags]
            else:
                disallowed = ~allowed[~explore_flags]
                q_greedy[disallowed] = -np.inf
            actions[~explore_flags] = np.argmax(q_greedy, axis=1)

        return actions


# -------- Adapters -------------------------------------------------------------
# PyTorch adapter
class TorchQAdapter(QValueProvider):
    """
    Adapter to use a torch.nn.Module inside EpsGreedyPolicy.

    Parameters
    ----------
    module : Any
        A PyTorch module with `forward` returning [B, n_actions] tensor.
    n_actions : int
        Number of actions.
    device : Optional[str]
        Torch device (e.g., "cuda", "cpu"). If None, use module's device.
    pre : Optional[Callable[[Any], "torch.Tensor"]]
        Optional preprocessor turning X into a torch.Tensor.
    post : Optional[Callable[[np.ndarray], np.ndarray]]
        Optional postprocessor applied to the resulting Q-values ndarray.
    no_grad : bool
        If True (default), run in torch.no_grad().
    """
    def __init__(
        self,
        module: Any,
        n_actions: int,
        device: Optional[str] = None,
        pre: Optional[Callable[[Any], "torch.Tensor"]] = None,
        post: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        no_grad: bool = True,
    ):
        self.module = module
        self.n_actions = int(n_actions)
        self.device = device
        self.pre = pre
        self.post = post
        self.no_grad = no_grad
        self._torch = torch

    def q_values(self, X: Any) -> np.ndarray:
        torch = self._torch
        if self.pre is not None:
            tX = self.pre(X)
        else:
            # try best-effort conversion
            tX = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)

        if self.device is not None:
            tX = tX.to(self.device)
            self.module = self.module.to(self.device)

        if self.no_grad:
            with torch.no_grad():
                q = self.module(tX)
        else:
            q = self.module(tX)

        out = q.detach().cpu().numpy()
        if self.post is not None:
            out = self.post(out)
        return out


# scikit-learn adapter
class SklearnQAdapter(QValueProvider):
    """
    Adapter to use an sklearn regressor/classifier that outputs [B, n_actions].

    Notes
    -----
    - For multi-output regressors: ensure model.predict(X) -> [B, n_actions].
    - For classifiers with `predict_proba`: you may want to convert probabilities
      to Q-values externally or pass a `post` function that maps logits/proba to Q.
    """
    def __init__(
        self,
        model: Any,
        n_actions: int,
        post: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.model = model
        self.n_actions = int(n_actions)
        self.post = post

    def q_values(self, X: Any) -> np.ndarray:
        q = self.model.predict(X)  # expect [B, n_actions]
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            # If model returns [B], broadcast to [B, 1] and check n_actions==1
            q = q[:, None]
        assert q.shape[1] == self.n_actions, (
            f"Expected n_actions={self.n_actions}, got {q.shape[1]}"
        )
        if self.post is not None:
            q = self.post(q)
        return q
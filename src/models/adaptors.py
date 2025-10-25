'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from __future__ import annotations

from typing import Protocol, Any, Optional, Callable, runtime_checkable

import numpy as np
import torch

@runtime_checkable
class QValueProvider(Protocol):
    """
    Minimal interface for Q-value models.

    Implementations must provide:
    - ``n_actions: int`` — number of discrete actions.
    - ``q_values(X) -> np.ndarray`` — batch Q-values of shape ``[B, n_actions]``.

    Examples
    --------
    >>> class Dummy(QValueProvider):
    ...     n_actions = 3
    ...     def q_values(self, X):
    ...         X = np.asarray(X)
    ...         return np.tile(np.arange(self.n_actions), (len(X), 1))
    >>> prov: QValueProvider = Dummy()
    >>> prov.q_values(np.zeros((4, 2))).shape
    (4, 3)
    """
    n_actions: int

    def q_values(self, X: Any) -> np.ndarray:
        """
        Compute Q-values for a batch of inputs.

        Parameters
        ----------
        X : Any
            Model-specific input batch (e.g., np.ndarray or torch.Tensor).

        Returns
        -------
        np.ndarray
            Array of shape ``[B, n_actions]`` with Q-values (float dtype).
        """
        ...


class TorchQAdapter(QValueProvider):
    """
    Adapter to use a ``torch.nn.Module`` within an ε-greedy policy or other logic
    expecting a ``QValueProvider``.

    Parameters
    ----------
    module : Any
        A PyTorch module whose forward returns a tensor of shape ``[B, n_actions]``.
    n_actions : int
        Number of discrete actions.
    device : Optional[str], default=None
        Torch device string (e.g., ``"cuda"``, ``"cpu"``, ``"mps"``). If ``None``,
        the adapter leaves tensors on their current device and does not move the module.
    pre : Optional[Callable[[Any], torch.Tensor]], default=None
        Optional preprocessor to convert inputs ``X`` to a ``torch.Tensor``.
        If ``None``, a best-effort ``torch.as_tensor`` is applied when needed.
    post : Optional[Callable[[np.ndarray], np.ndarray]], default=None
        Optional postprocessor applied to the resulting Q-values ndarray.
    no_grad : bool, default=True
        If True, wrap the forward pass in ``torch.no_grad()`` for inference.

    Notes
    -----
    - Output is always converted to a CPU NumPy array (float dtype).
    - This class does **not** change the module's training mode.
    """

    def __init__(
        self,
        module: Any,
        n_actions: int,
        device: Optional[str] = None,
        pre: Optional[Callable[[Any], torch.Tensor]] = None,
        post: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        no_grad: bool = True,
    ) -> None:
        self.module = module
        self.n_actions = int(n_actions)
        self.device = device
        self.pre = pre
        self.post = post
        self.no_grad = no_grad
        self._torch = torch

    def q_values(self, X: Any) -> np.ndarray:
        """
        Compute Q-values via the wrapped PyTorch module.

        Parameters
        ----------
        X : Any
            Input batch. Will be passed through ``pre`` if provided; otherwise
            converted using ``torch.as_tensor`` when not already a tensor.

        Returns
        -------
        np.ndarray
            Q-values of shape ``[B, n_actions]`` (float).
        """
        torch_ = self._torch
        if self.pre is not None:
            tX = self.pre(X)
        else:
            tX = X if isinstance(X, torch_.Tensor) else torch_.as_tensor(X)
        if self.device is not None:
            tX = tX.to(self.device)
        if self.no_grad:
            with torch_.no_grad():
                q = self.module(tX)  # type: ignore[misc]
        else:
            q = self.module(tX)      # type: ignore[misc]

        if self.post is not None:
            q = self.post(q)

        # Shape check on tensor
        if q.dim() != 2 or q.size(1) != self.n_actions:
            raise ValueError(
                f"TorchQAdapter expected output shape [B, {self.n_actions}], got {tuple(q.shape)}"
            )
        return q


class SklearnQAdapter(QValueProvider):
    """
    Adapter to use an sklearn-like estimator as a ``QValueProvider``.

    Parameters
    ----------
    model : Any
        An sklearn regressor/classifier. Must produce an array of shape
        ``[B, n_actions]`` via ``model.predict(X)`` (or compatible).
    n_actions : int
        Number of discrete actions.
    post : Optional[Callable[[np.ndarray], np.ndarray]], default=None
        Optional mapping applied to the raw predictions (e.g., logits/proba → Q).

    Notes
    -----
    - For multi-output regressors, ensure ``predict(X)`` returns shape ``[B, n_actions]``.
    - For classifiers using ``predict_proba``, you may wrap the model in a lightweight
      shim whose ``predict`` returns concatenated class probabilities or pass a
      ``post`` function to transform outputs into Q-values.
    """

    def __init__(
        self,
        model: Any,
        n_actions: int,
        post: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.model = model
        self.n_actions = int(n_actions)
        self.post = post

    def q_values(self, X: Any) -> np.ndarray:
        """
        Compute Q-values via the wrapped sklearn-like model.

        Parameters
        ----------
        X : Any
            Input batch appropriate for the underlying model.

        Returns
        -------
        np.ndarray
            Q-values of shape ``[B, n_actions]`` (float).
        """
        q = self.model.predict(X)  # expect [B, n_actions]
        q = np.asarray(q, dtype=float)

        # Handle 1-D outputs (rare): broadcast to [B, 1]
        if q.ndim == 1:
            q = q[:, None]

        if q.ndim != 2 or q.shape[1] != self.n_actions:
            raise ValueError(
                f"SklearnQAdapter expected output shape [B, {self.n_actions}], "
                f"got {tuple(q.shape)}"
            )

        if self.post is not None:
            q = self.post(q)

        return q
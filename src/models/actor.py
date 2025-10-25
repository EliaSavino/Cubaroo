'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch import nn
from src.models.adaptors import TorchQAdapter, SklearnQAdapter
from src.models.policy import EpsGreedyPolicy

@dataclass
class ModelActor:
    """
    Unified ε-greedy actor facade.

    If ``use_model_act=True``, this actor delegates to ``model.act(x, epsilon, action_mask)``.
    Otherwise it uses a framework-agnostic ε-greedy policy (NumPy-facing) built on top
    of the provided adapter and converts the resulting actions back to a Torch tensor.

    Parameters
    ----------
    model : nn.Module
        The Q-network (or a module exposing ``.act``).
    device : torch.device
        Target device for returned action tensors.
    n_actions : int
        Number of discrete actions.
    use_model_act : bool
        Whether to call ``model.act`` directly.
    policy : EpsGreedyPolicy, optional
        ε-greedy policy used when ``use_model_act=False``. Must produce an
        ``np.ndarray`` of dtype int64, shape (B,).

    Notes
    -----
    - Returned tensor dtype is normalized to ``torch.long``; callers can assume
      indices suitable for ``gather`` or environment stepping.
    """

    model: nn.Module
    device: torch.device
    n_actions: int
    use_model_act: bool
    policy: Optional[EpsGreedyPolicy] = None  # provided when not using model.act

    @torch.no_grad()
    def act(
        self,
        x: torch.Tensor,
        epsilon: float = 0.0,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select actions for a batch of observations.

        Parameters
        ----------
        x : torch.Tensor
            Batched observations, shape (B, ...), already on any device.
        epsilon : float, default=0.0
            ε for ε-greedy selection (0 → greedy).
        action_mask : torch.Tensor or None, optional
            Optional mask of allowed actions with shape (B, A). Non-zero/True entries
            are allowed; zero/False entries are disallowed.

        Returns
        -------
        torch.Tensor
            Action indices of shape (B,), dtype ``torch.long`` on ``self.device``.
        """
        if self.use_model_act:
            # Normalize dtype/device for downstream code
            out = self.model.act(x, epsilon=epsilon, action_mask=action_mask)  # type: ignore[attr-defined]
            return out.to(self.device).long()

        if self.policy is None:
            raise RuntimeError(
                "ModelActor was constructed without a policy but use_model_act=False."
            )

        # Universal policy path (NumPy under the hood)
        mask_np: Optional[np.ndarray] = None
        if action_mask is not None:
            mask_np = action_mask.detach().to("cpu").numpy()

        a_np = self.policy.act(x, epsilon=epsilon, action_mask=mask_np)  # (B,), int64 expected
        if isinstance(a_np, np.ndarray):
            return torch.from_numpy(a_np).to(self.device).long()
        elif isinstance(a_np, torch.Tensor):
            return a_np.to(self.device).long()
        else:
            raise TypeError(f"Unexpected action type: {type(a_np)}")


def build_actor(
    model: nn.Module,
    device: torch.device,
    n_actions: int,
    prefer_model_act: bool = True,
) -> ModelActor:
    """
    Construct a unified actor for training/evaluation.

    Strategy
    --------
    1) If the model exposes a callable ``.act(x, epsilon, action_mask=None)`` AND
       ``prefer_model_act`` is True, return a ``ModelActor`` that directly calls it.
    2) Otherwise, fall back to a **framework-agnostic** ε-greedy policy built on top of
       a Q-adapter:
         - Torch modules → ``TorchQAdapter``
         - sklearn-like models (``.predict``) → ``SklearnQAdapter``

    Parameters
    ----------
    model : nn.Module
        Q-network or module used to compute Q-values (or that already exposes ``.act``).
    device : torch.device
        Device for returned action tensors.
    n_actions : int
        Number of discrete actions.
    prefer_model_act : bool, default=True
        Prefer the model's own ``.act`` method when available.

    Returns
    -------
    ModelActor
        An actor exposing ``.act(x, epsilon, action_mask=None) -> torch.LongTensor[B]``.

    Raises
    ------
    TypeError
        If no suitable policy/adapter can be inferred for the provided model.

    Notes
    -----
    - This function assumes concrete implementations of ``EpsGreedyPolicy``,
      ``TorchQAdapter``, and ``SklearnQAdapter`` exist elsewhere in your codebase.
    - The returned actor **always** outputs ``torch.long`` action indices.
    """
    has_model_act = prefer_model_act and callable(getattr(model, "act", None))
    if has_model_act:
        return ModelActor(
            model=model,
            device=device,
            n_actions=n_actions,
            use_model_act=True,
            policy=None,
        )

    # Fallback to framework-agnostic policy + adapter
    if isinstance(model, nn.Module):
        # Lazy imports or constructor calls for your concrete classes here:
        policy = EpsGreedyPolicy(  # type: ignore[call-arg, assignment]
            TorchQAdapter(          # type: ignore[call-arg]
                module=model,
                n_actions=n_actions,
                device=str(device),  # e.g., "cpu", "cuda:0", "mps"
                pre=None,            # pass tensors directly
                post=None,
                no_grad=True,
            )
        )
        return ModelActor(
            model=model,
            device=device,
            n_actions=n_actions,
            use_model_act=False,
            policy=policy,
        )

    # Optional: sklearn route (only if such usage exists in your trainer)
    if hasattr(model, "predict"):
        policy = EpsGreedyPolicy(  # type: ignore[call-arg, assignment]
            SklearnQAdapter(       # type: ignore[call-arg]
                model=model,       # type: ignore[arg-type]
                n_actions=n_actions,
                post=None,         # or a mapper from proba/logits -> Q-values
            )
        )
        return ModelActor(
            model=model,
            device=device,
            n_actions=n_actions,
            use_model_act=False,
            policy=policy,
        )

    raise TypeError(
        "Cannot build actor: model has no `.act` and is neither a torch.nn.Module "
        "nor sklearn-like with `.predict`."
    )
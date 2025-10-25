'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from __future__ import annotations

import numpy as np
import torch
from torch import nn as nn
from typing import Union




def polyak_update(target: nn.Module, online: nn.Module, tau: float = 0.005) -> None:
    """
    Perform a Polyak (soft) update of the target network parameters.

    The operation updates each parameter of the ``target`` network as:

        θ_target ← (1 - τ) * θ_target + τ * θ_online

    where ``τ`` (tau) is the soft update coefficient, typically a small value
    such as 0.005.

    Parameters
    ----------
    target : nn.Module
        The target network whose parameters will be *partially* updated.
    online : nn.Module
        The online (or main) network providing the new parameter values.
    tau : float, optional
        Soft update rate between 0 and 1 (default = 0.005).

    Returns
    -------
    None
        The function updates the target network in place.

    Raises
    ------
    ValueError
        If ``tau`` is outside the range [0, 1].

    Notes
    -----
    - Polyak averaging smooths parameter updates, improving stability in
      algorithms like DDPG, TD3, and DQN variants.
    - For ``tau = 1.0``, this is equivalent to a hard copy (target ← online).
    - For ``tau = 0.0``, the target remains unchanged.

    Examples
    --------
    >>> target = nn.Linear(4, 2)
    >>> online = nn.Linear(4, 2)
    >>> polyak_update(target, online, tau=0.01)
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be between 0 and 1, got {tau}")

    with torch.no_grad():
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_param.data.mul_(1 - tau).add_(online_param.data, alpha=tau)


def linear_epsilon(step: int, cfg: "DQNConfig") -> float:
    """
    Linearly anneal epsilon for ε-greedy exploration.

    The schedule interpolates from ``cfg.eps_start`` to ``cfg.eps_end`` over
    ``cfg.eps_decay_steps`` steps and then stays at ``cfg.eps_end``.

    Parameters
    ----------
    step : int
        The current global training step (non-negative).
    cfg : DQNConfig
        Configuration object providing:
        - ``eps_start`` (float): starting epsilon.
        - ``eps_end`` (float): final epsilon after decay.
        - ``eps_decay_steps`` (int): number of steps over which to decay.

    Returns
    -------
    float
        The epsilon value for the given step.

    Raises
    ------
    ValueError
        If ``step`` is negative.

    Notes
    -----
    If ``cfg.eps_decay_steps <= 0``, the function degrades to returning
    ``cfg.eps_end`` immediately (i.e., no decay).

    Examples
    --------
    >>> class DQNConfig: ...
    >>> cfg = DQNConfig()
    >>> cfg.eps_start, cfg.eps_end, cfg.eps_decay_steps = 1.0, 0.05, 10000
    >>> linear_epsilon(0, cfg)
    1.0
    >>> linear_epsilon(5000, cfg)
    0.525
    >>> linear_epsilon(20000, cfg)
    0.05
    """
    if step < 0:
        raise ValueError("step must be non-negative")

    # Guard against zero/negative decay to avoid division by zero.
    if getattr(cfg, "eps_decay_steps", 0) <= 0:
        return float(cfg.eps_end)

    t = min(1.0, step / float(cfg.eps_decay_steps))
    return float(cfg.eps_start + (cfg.eps_end - cfg.eps_start) * t)


def to_device(
    x: np.ndarray,
    device: Union[torch.device, str],
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert a NumPy array to a Torch tensor with the desired dtype and device.

    Parameters
    ----------
    x : np.ndarray
        Input array. Must be convertible via ``torch.from_numpy`` (i.e., numeric).
    device : torch.device or str
        Target device (e.g., ``"cpu"``, ``"cuda"`` or ``torch.device("mps")``).
    dtype : torch.dtype
        Desired tensor dtype, e.g. ``torch.long`` or ``torch.float32``.

    Returns
    -------
    torch.Tensor
        Tensor on the specified device with the requested dtype.

    Notes
    -----
    - Uses ``torch.from_numpy`` (zero-copy when possible). The resulting tensor
      shares memory with ``x`` until moved to another device or otherwise cloned
      by PyTorch internals.
    - Only integer vs floating dtypes are differentiated in the original code.
      This implementation respects the exact ``dtype`` you pass.

    Examples
    --------
    >>> arr = np.array([1, 2, 3], dtype=np.int64)
    >>> to_device(arr, "cpu", torch.long).dtype == torch.long
    True
    >>> to_device(arr.astype(np.float32), "cpu", torch.float32).dtype == torch.float32
    True
    """
    t = torch.from_numpy(x)
    # Keep behavior minimal but precise: cast to the requested dtype.
    if t.dtype is not dtype:
        t = t.to(dtype)
    return t.to(device)


def infer_obs_dtype(obs: np.ndarray) -> torch.dtype:
    """
    Infer the appropriate Torch dtype for observations.

    Heuristic:
    - If the NumPy dtype is any kind of integer, return ``torch.long`` (suitable
      for index-based encodings / embedding lookups).
    - Otherwise, return ``torch.float32`` (suitable for one-hot / dense floats).

    Parameters
    ----------
    obs : np.ndarray
        A representative observation array.

    Returns
    -------
    torch.dtype
        ``torch.long`` for integer arrays, else ``torch.float32``.

    Examples
    --------
    >>> infer_obs_dtype(np.array([0, 1, 2], dtype=np.int32)) is torch.long
    True
    >>> infer_obs_dtype(np.array([0., 1., 0.], dtype=np.float32)) is torch.float32
    True
    """
    return torch.long if np.issubdtype(obs.dtype, np.integer) else torch.float32

def _maybe_compile(module: nn.Module, device: torch.device) -> nn.Module:
    try:
        import torch._dynamo as dynamo
        dynamo.config.assume_static_by_default = True  # help static scheduling

        if device.type == "mps":
            # Avoid Inductor kernels on Metal; aot_eager builds the graph but executes eagerly
            return torch.compile(module)
        else:
            # CUDA/CPU: inductor is fine
            return torch.compile(module, mode="reduce-overhead", dynamic=False)
    except Exception as e:
        print(f"  ! torch.compile skipped: {e}")
        return module
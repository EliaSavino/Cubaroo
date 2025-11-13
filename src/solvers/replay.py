'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from __future__ import annotations

from typing import Tuple, List

import numpy as np


class Replay:
    """
    Experience replay buffer for off-policy RL agents.

    Stores transitions of the form ``(state, action, reward, next_state, done)``
    up to a fixed capacity and allows random mini-batch sampling for training.

    Two optional behaviors:
    - ``drop_after_sample=True``: remove sampled transitions (each is used at most once).
    - ``prioritized=True``: sample according to TD-error priorities (PER-lite) with
      exponent ``alpha``. Importance weights are returned for loss reweighting.

    Parameters
    ----------
    cap : int, default=300_000
        Maximum number of transitions to store (capacity > 0).
    prioritized : bool, default=False
        If True, enabled prioritized sampling by TD-error.
    alpha : float, default=0.6
        Prioritization exponent. 0 → uniform, 1 → fully proportional.
    drop_after_sample : bool, default=False
        If True, remove transitions after sampling. (For PER, keep this False.)

    Notes
    -----
    - Transition layout is: ``(s, a, r, ns, d)`` where
      ``s`` and ``ns`` are arrays (any shape), ``a`` is an int, and ``r`` and ``d`` are floats.
    - When ``prioritized=True``, the sampler returns normalized importance weights
      ``w_i ∝ (N * p_i)^(-1)`` scaled so that ``max_i w_i = 1``.
    - If you enable ``drop_after_sample=True`` together with ``prioritized=True``,
      priority storage will also shrink; this class does not implement PER's
      canonical tree structure—keep ``drop_after_sample=False`` for PER-like usage.

    Examples
    --------
    >>> rb = Replay(cap=1000, prioritized=True, alpha=0.6)
    >>> rb.push(s=np.zeros(3), a=1, r=0.5, ns=np.ones(3), d=0.0, td_error=1.2)
    >>> batch = rb.sample(batch=32)
    >>> (S, A, R, NS, D, idx, w) = batch
    >>> rb.update_priorities(idx, td_errors=np.random.rand(len(idx)))
    """



    def __init__(
        self,
        cap: int = 300_000,
        prioritized: bool = False,
        alpha: float = 0.6,
        drop_after_sample: bool = False,
    ) -> None:
        if cap <= 0:
            raise ValueError(f"'cap' must be > 0, got {cap}")
        self.cap: int = cap
        self.buf: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.pos: int = 0

        self.prioritized: bool = prioritized
        self.alpha: float = float(alpha)
        self.drop_after_sample: bool = drop_after_sample  # keep False for PER

        # Priority storage (used only if prioritized=True)
        self.priorities: np.ndarray = np.zeros((cap,), dtype=np.float32)
        self.eps: float = 1e-6  # min priority to avoid zeros/NaNs

    def push(self, s, a: int, r: float, ns, d: float, td_error: float | None = None):
        """
        Append a transition to the buffer (overwriting oldest once at capacity).

        Parameters
        ----------
        s : np.ndarray
            State (any shape).
        a : int
            Action index.
        r : float
            Scalar reward.
        ns : np.ndarray
            Next state (same structure as ``s``).
        d : float
            Done flag (float in {0.0, 1.0}; kept as float to match upstream code).
        td_error : float or None, optional
            TD-error for prioritized replay. If ``None``, uses the current max priority
            among stored items (standard PER heuristic).
        """
        item = (s, a, r, ns, d)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item

        if self.prioritized:
            if td_error is None:
                current_len = max(1, len(self.buf))
                max_prio = self.priorities[:current_len].max()
                if not np.isfinite(max_prio) or max_prio <= 0:
                    max_prio = 1.0
                self.priorities[self.pos] = max_prio
            else:
                p = abs(float(td_error))
                if not np.isfinite(p) or p <= 0:
                    p = 1.0
                self.priorities[self.pos] = p

        self.pos = (self.pos + 1) % self.cap

        return self.pos

    def sample(self, batch: int):
        """
        Sample a mini-batch of transitions.

        Parameters
        ----------
        batch : int
            Number of transitions to draw (with replacement).

        Returns
        -------
        tuple of np.ndarray
            ``(S, A, R, NS, D, idx, w)`` where:
            - ``S``: stacked states, shape ``(batch, ...)``
            - ``A``: actions, shape ``(batch,)``
            - ``R``: rewards (float32), shape ``(batch,)``
            - ``NS``: stacked next states, shape ``(batch, ...)``
            - ``D``: dones (float32), shape ``(batch,)``
            - ``idx``: sampled indices, shape ``(batch,)``
            - ``w``: importance weights (float32), shape ``(batch,)``

        Raises
        ------
        ValueError
            If the buffer is empty or ``batch <= 0``.
        """
        n = len(self.buf)
        assert n > 0, "Cannot sample from empty buffer"

        if self.prioritized:
            raw = self.priorities[:n]
            raw = np.where(np.isfinite(raw) & (raw > 0), raw, self.eps)
            ps = np.power(raw, self.alpha)
            Z = ps.sum()
            if not np.isfinite(Z) or Z <= 0:
                probs = np.full(n, 1.0 / n, dtype=np.float32)
            else:
                probs = ps / Z
            idx = np.random.choice(n, size=batch, p=probs)
            weights = (n * probs[idx]) ** (-1)
            weights /= weights.max()
        else:
            idx = np.random.randint(0, n, size=batch)
            weights = np.ones(batch, dtype=np.float32)

        s, a, r, ns, d = zip(*[self.buf[i] for i in idx])
        out = (np.stack(s),
               np.array(a),
               np.array(r, dtype=np.float32),
               np.stack(ns),
               np.array(d, dtype=np.float32),
               idx,
               weights.astype(np.float32))

        if self.drop_after_sample:
            self._delete_indices(idx)

        return out

    def update_priorities(self, idx, td_errors):
        """
        Update priorities for prioritized replay.

        Parameters
        ----------
        idx : Sequence[int] or np.ndarray
            Indices of sampled transitions.
        td_errors : Sequence[float] or np.ndarray
            Corresponding TD-errors (same length as ``idx``).

        Notes
        -----
        - Non-finite or non-positive TD-errors are clamped to ``eps``.
        - No-op if ``prioritized=False``.
        """
        if not self.prioritized:
            return
        td = np.abs(np.asarray(td_errors, dtype=np.float32))
        td[~np.isfinite(td)] = 1.0
        td = np.maximum(td, self.eps)
        n = len(self.buf)
        idx = np.asarray(idx)
        valid = (idx >= 0) & (idx < n)
        self.priorities[idx[valid]] = td[valid]

    def _delete_indices(self, idx):
        """
        Delete transitions (and corresponding priorities) by indices.

        Parameters
        ----------
        idx : Iterable[int]
            Indices to remove. Duplicates are ignored.

        Notes
        -----
        This is an O(N) operation due to list/array deletions. It is intended
        for occasional use (e.g., episodic datasets). For heavy-duty PER use,
        keep ``drop_after_sample=False`` and consider a segment tree.
        """
        idx = sorted(set(int(i) for i in idx), reverse=True)
        for i in idx:
            if i < len(self.buf):
                self.buf.pop(i)
                # keep priorities length aligned (simple but O(n))
                self.priorities = np.delete(self.priorities, i)

    def __len__(self):
        """
        Return the current size of internal buffer.
        :return:
        """
        return len(self.buf)

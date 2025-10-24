'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Cube Gym for solvers, translates cube state to tensors and actions to cube actions.

'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from src.cube import Cube  # your class
from src.solvers.encoders import CubieEncoder

MOVES = [f + s for f in "UDLRFB" for s in ["", "'"]]


def apply_move(cube: Cube, move: str) -> None:
    """
    Apply a face rotation to the cube in place.

    Parameters
    ----------
    cube : Cube
        The cube instance to mutate.
    move : str
        String code for the move. Must start with a face letter in {"U","D","L","R","F","B"}
        and may optionally end with an apostrophe (e.g. "R'") for counter-clockwise rotation.
    """
    face = move[0]
    clockwise = not move.endswith("'")
    cube.rotate(face, clockwise)


@dataclass
class CubeGymCubie:
    """
    Minimal Rubik’s Cube environment for reinforcement learning.

    Implements the classic `(s, a, r, s')` interface:
    - `reset()` returns an encoded observation.
    - `step(action)` applies a cube move and returns the next observation,
      reward, done flag, and info dict.

    Reward design:
        reward = α × Δ(score) − 0.0001
        +5 bonus is added when the cube is fully solved.

    The score function is assumed to increase as the cube approaches the solved state.

    Parameters
    ----------
    encoder : CubieEncoder
        Encoder converting a Cube instance into a numerical observation.
    alpha : float, default=1.0
        Scale factor for reward magnitude.
    max_steps : int, default=100
        Maximum number of moves before the episode terminates.

    Attributes
    ----------
    cube : Cube
        Internal cube instance.
    prev_score : float
        Previous cube score (for delta-based rewards).

    Notes
    -----
    - `Cube` must expose:
        • `.rotate(face: str, clockwise: bool)`
        • `.scramble(length: int)`
        • `.score() -> float`
        • `.is_solved() -> bool`
        • `.get_history() -> pandas.DataFrame`
    - The environment does **not** internally seed random scrambles; reproducibility
      should be handled by the caller.

    Examples
    --------
    >>> env = CubeGymCubie(encoder=CubieEncoder())
    >>> obs = env.reset(scramble_len=3)
    >>> obs.shape
    (256,)
    >>> obs, reward, done, info = env.step(0)
    >>> info["move"]
    'U'
    """

    encoder: CubieEncoder
    alpha: float = 1.0
    max_steps: int = 100

    def reset(self, scramble_len: int = 0) -> np.ndarray:
        """
        Reset the environment and return the initial encoded state.

        Parameters
        ----------
        scramble_len : int, default=0
            Number of random moves to scramble the cube before starting.

        Returns
        -------
        np.ndarray
            Encoded observation representing the scrambled cube.
        """
        self.cube = Cube()
        self.cube.scramble(length=scramble_len)
        self.prev_score = self.cube.score()
        return self.encoder.encode(self.cube)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply a cube move and return the next state, reward, done flag, and info.

        Parameters
        ----------
        action_idx : int
            Index of the move to apply (0–11). Uses ``MOVES`` ordering:
            ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"].

        Returns
        -------
        observation : np.ndarray
            Encoded cube state after the move.
        reward : float
            Change in cube score (scaled by ``alpha``) minus a small step penalty.
        done : bool
            True if the cube is solved or max_steps exceeded.
        info : dict
            Extra diagnostics: ``{"move", "score", "history_len"}``.

        Notes
        -----
        - Adds a +5 bonus when the cube is solved.
        - Uses the cube's internal history to determine episode length.
        """
        move = MOVES[action_idx]
        apply_move(self.cube, move)

        score = self.cube.score()
        delta = score - self.prev_score
        reward = self.alpha * delta - 0.0001  # small penalty encourages faster solving
        self.prev_score = score

        solved = self.cube.is_solved()
        if solved:
            score += 5  # terminal bonus for solving

        # track only the 'solve' phase of cube history
        history = self.cube.get_history()
        history = history[history["phase"] == "solve"]

        done = solved or (len(history) >= self.max_steps)
        obs = self.encoder.encode(self.cube)

        info: Dict[str, Any] = {
            "move": move,
            "score": score,
            "history_len": len(history),
        }
        return obs, reward, done, info

'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Cube Gym for solvers, translates cube state to tensors and actions to cube actions.

'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np, random
from src.cube import Cube  # your class

MOVES = [f + s for f in "UDLRFB" for s in ["", "'"]]

def apply_move(cube, move: str):
    face = move[0]
    if move.endswith("'"):
        cube.rotate(face, False)
    else:
        cube.rotate(face, True)

class IndexCubieEncoder:
    """
    Encodes the cube as a vector of integer indices:
      [corner_perm(8), corner_ori(8), edge_perm(12), edge_ori(12)]
    Shape: (8 + 8 + 12 + 12,) = (40,)
    Each element is an integer index that can be used for learned embeddings.
    """
    dim = 40

    def encode(self, cube) -> np.ndarray:
        corners_perm = [c.piece_idx for c in cube.corners]
        corners_ori  = [c.ori for c in cube.corners]
        edges_perm   = [e.piece_idx for e in cube.edges]
        edges_ori    = [e.ori for e in cube.edges]
        return np.array(corners_perm + corners_ori + edges_perm + edges_ori, dtype=np.int64)


class FlatCubieEncoder:
    """
    Flattens permutation and orientation directly as floats (no one-hot).
    Essentially the same as IndexCubieEncoder but float32 and single vector.
    """
    dim = 40

    def encode(self, cube) -> np.ndarray:
        vals = []
        for c in cube.corners:
            vals += [c.piece_idx, c.ori]
        for e in cube.edges:
            vals += [e.piece_idx, e.ori]
        return np.array(vals, dtype=np.float32)

class CubieEncoder:
    """Pure cubie encoder → float vector (dim=256)."""
    dim = 256
    def encode(self, cube) -> np.ndarray:
        vec=[]
        for c in cube.corners:
            one_perm=np.zeros(8); one_perm[c.piece_idx]=1
            one_ori =np.zeros(3); one_ori[c.ori]=1
            vec+=[one_perm,one_ori]
        for e in cube.edges:
            one_perm=np.zeros(12); one_perm[e.piece_idx]=1
            one_ori =np.zeros(2);  one_ori[e.ori]=1
            vec+=[one_perm,one_ori]
        return np.concatenate(vec).astype(np.float32)



@dataclass
class CubeGymCubie:
    """
    Minimal cube environment for RL:
      - Uses Cube’s own history and score()
      - No external step penalty
      - Reward = Δ(score): positive if cube becomes more solved
    """
    encoder: CubieEncoder
    alpha: float = 1.0  # scale of reward (tune if learning unstable)
    max_steps: int = 100

    def reset(self, scramble_len: int = 0):
        self.cube = Cube()
        self.cube.scramble(length=scramble_len)
        # let cube store its own move history internally
        self.prev_score = self.cube.score()
        return self.encoder.encode(self.cube)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        move = MOVES[action_idx]
        apply_move(self.cube, move)

        score = self.cube.score()
        reward = self.alpha * (score - self.prev_score) - 0.0001 # small step penalty to encourage faster solves
        self.prev_score = score

        solved = self.cube.is_solved()
        if solved:
            score+= 5# bonus for solving
        # history:
        history = self.cube.get_history()
        history = history[history['phase'] == 'solve']
        done = solved or (len(history)>= self.max_steps)

        obs = self.encoder.encode(self.cube)
        info = {"move": move, "score": score, "history_len": len(history)}
        return obs, reward, done, info

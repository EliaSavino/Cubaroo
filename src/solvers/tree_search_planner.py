'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from src.solvers.cube_gym import MOVES, apply_move, inverse_action_idx
import copy
import math
import numpy as np



@dataclass
class MCTSPlanner:
    """
    AlphaZero-lite MCTS for the Rubik's Cube (Cubie) environment.

    Usage:
        planner = MCTSPlanner(max_depth=5, n_sim=800)
        a = planner.choose_action(env)              # action index at root
        seq, val = planner.best_sequence(env)       # up to depth 5 plan and estimated value
        tree = planner.search_tree(env)             # diagnostic info per root-child

    Notes
    -----
    - Leaf evaluation can use either a Q-network (if q_fn is provided) or the env.scorer.
    - Priors can come from a policy function (policy_prior) or default to uniform.
    - This planner is purely *planning-time*; it does not modify env or rewards.
    - Designed to be drop-in with your existing DQNTrainer: call planner before env.step(...).
    """

    # Optional hooks
    q_fn: Optional[callable] = None          # q_fn(cube) -> np.ndarray[12] of Q-values
    policy_prior: Optional[callable] = None  # policy_prior(cube) -> np.ndarray[12] of logits or probs

    # Core knobs
    gamma: float = 0.99
    cpuct: float = 1.5
    n_sim: int = 400
    max_depth: int = 5

    # Implementation detail
    _EPS: float = 1e-9

    # --------------------- Public API ---------------------
    def choose_action(self, env: "CubeGymCubie", prev_action_idx: Optional[int]=None) -> int:
        """Run MCTS at the current env.cube and return the best root action index."""
        root_key, N, Q = self._run_search(env, prev_action_idx=prev_action_idx)
        root_counts = np.array([N.get((root_key, a), 0) for a in range(len(MOVES))], dtype=float)
        if root_counts.sum() == 0:
            # Fallback to Q
            root_q = np.array([Q.get((root_key, a), -1e9) for a in range(len(MOVES))], dtype=float)
            return int(root_q.argmax())
        return int(root_counts.argmax())

    def best_sequence(self, env: "CubeGymCubie") -> Tuple[List[int], float]:
        """Return up to `max_depth` action indices representing the best found path and its value."""
        root_key, N, Q, parent, children, values = self._run_search(env, return_graph=True)
        # Greedy descent from root using mean Q at each level
        path: List[int] = []
        node = root_key
        depth = 0
        while depth < self.max_depth and any((node, a) in Q for a in range(len(MOVES))):
            # among legal children, choose highest Q
            candidates = [(a, Q.get((node, a), -1e9)) for a in range(len(MOVES)) if (node, a) in Q]
            if not candidates:
                break
            a_best = max(candidates, key=lambda t: t[1])[0]
            path.append(a_best)
            node = children.get((node, a_best))
            if node is None:
                break
            depth += 1
        # Estimated value of root is max_a Q(root, a) or 0 if empty
        root_val = max([Q.get((root_key, a), float('-inf')) for a in range(len(MOVES))] or [0.0])
        return path, float(root_val)

    def search_tree(self, env: "CubeGymCubie") -> Dict[str, Any]:
        """Return a diagnostic dict with per-root-child stats: visits, Q, and move strings."""
        root_key, N, Q = self._run_search(env)
        data = []
        for a in range(len(MOVES)):
            data.append({
                "action_idx": a,
                "move": MOVES[a],
                "visits": int(N.get((root_key, a), 0)),
                "Q": float(Q.get((root_key, a), 0.0)),
            })
        data.sort(key=lambda d: (d["visits"], d["Q"]), reverse=True)
        return {"root": root_key, "children": data}

    # --------------------- Internal: single search run ---------------------
    def _run_search(self, env: "CubeGymCubie", return_graph: bool = False, prev_action_idx: Optional[int] = None):
        root_cube = copy.deepcopy(env.cube)
        root_key = self._hash_cube(root_cube)

        N: Dict[Tuple[int, int], int] = {}   # visit counts per edge (node, action)
        W: Dict[Tuple[int, int], float] = {} # total value
        Q: Dict[Tuple[int, int], float] = {} # mean value
        P: Dict[int, np.ndarray] = {}        # priors per node
        expanded: set[int] = set()
        children: Dict[Tuple[int, int], int] = {}  # (node,a) -> child node key
        parent: Dict[int, Tuple[int, int]] = {}    # child node -> (parent_key, action)
        values: Dict[int, float] = {}              # cached leaf values

        for _ in range(self.n_sim):
            path: List[Tuple[int, int]] = []
            cube_sim = copy.deepcopy(root_cube)
            node = root_key
            depth = 0
            done = cube_sim.is_solved()

            # last move taken along THIS simulated path
            last_a: Optional[int] = prev_action_idx

            # ----- selection -----
            while node in expanded and depth < self.max_depth and not done:
                forbidden = inverse_action_idx(last_a) if last_a is not None else None
                a = self._uct_action(node, cube_sim, N, Q, P, forbidden_action=forbidden)
                path.append((node, a))
                apply_move(cube_sim, MOVES[a])
                done = cube_sim.is_solved()
                child_key = self._hash_cube(cube_sim)
                children[(node, a)] = child_key
                if child_key not in parent:
                    parent[child_key] = (node, a)
                node = child_key
                last_a = a  # update last action for next step
                depth += 1

            # ----- expansion -----
            if not done and depth < self.max_depth and node not in expanded:
                expanded.add(node)
                P[node] = self._priors_for(node, cube_sim)

            # ----- evaluation (leaf value) -----
            if done:
                V = 0.0  # terminal; reward already accounted in env during real steps
            else:
                if node in values:
                    V = values[node]
                else:
                    V = self._leaf_value(env, cube_sim)  # ONE model/scorer call
                    values[node] = V

            # ----- backup -----
            for (k, a) in reversed(path):
                N[(k, a)] = N.get((k, a), 0) + 1
                W[(k, a)] = W.get((k, a), 0.0) + V
                Q[(k, a)] = W[(k, a)] / N[(k, a)]

        if return_graph:
            return root_key, N, Q, parent, children, values
        return root_key, N, Q

    # --------------------- Node utilities ---------------------
    def _hash_cube(self, cube) -> int:
        """
        Compute a stable integer hash from cubie permutations and orientations.
        Works as long as cube.corners[i].pos / .ori and cube.edges[i].pos / .ori exist.
        """
        try:
            corner_pos = tuple(c.piece_idx for c in cube.corners)
            corner_ori = tuple(c.ori for c in cube.corners)
            edge_pos = tuple(e.piece_idx for e in cube.edges)
            edge_ori = tuple(e.ori for e in cube.edges)
            return hash((corner_pos, corner_ori, edge_pos, edge_ori))
        except AttributeError:
            # fallback to slower but robust method
            return hash(str(cube))

    def _priors_for(self, node_key: int, cube) -> np.ndarray:
        A = len(MOVES)
        if self.policy_prior is None:
            p = np.ones(A, dtype=float) / A
        else:
            raw = np.asarray(self.policy_prior(cube), dtype=float)
            if raw.ndim != 1 or raw.shape[0] != A:
                raise ValueError("policy_prior must return shape [12] for 12 moves")
            # convert logits to probs if needed
            raw = np.exp(raw - raw.max())
            p = raw / (raw.sum() + self._EPS)
        return p

    def _uct_action(self, node_key: int, cube, N, Q, P, forbidden_action: Optional[int] = None) -> int:
        A = len(MOVES)
        p = P.get(node_key)
        if p is None:
            p = np.ones(A, dtype=float) / A
        total = sum(N.get((node_key, a), 0) for a in range(A)) + self._EPS

        best_a = 0
        best_score = -1e30

        for a in range(A):
            # skip the forbidden action, if any
            if forbidden_action is not None and a == forbidden_action:
                continue

            n = N.get((node_key, a), 0)
            q = Q.get((node_key, a), 0.0)
            u = self.cpuct * p[a] * math.sqrt(total) / (1.0 + n)
            s = q + u
            if s > best_score:
                best_score = s
                best_a = a

        return best_a

    def _leaf_value(self, env: "CubeGymCubie", cube) -> float:
        """Single evaluation at a newly reached leaf.

        If q_fn is provided, we use max_a Q(cube,a); else we fallback to env.scorer.
        """
        if self.q_fn is not None:
            q = np.asarray(self.q_fn(cube), dtype=float)  # shape [12]
            return float(q.max())
        # heuristic fallback
        return float(env.scorer(cube))

    def plan_root(self, env, prev_action_idx: Optional[int] = None) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Run a search and return:
          - a_star: int, best action at root (by visit count)
          - q_mcts: np.ndarray [12], mean backed-up values per root action (0 if unseen)
          - pi_mcts: np.ndarray [12], visit-count distribution at root (sums to 1)
        """
        root_key, N, Q = self._run_search(env, prev_action_idx=prev_action_idx)
        A = len(MOVES)
        counts = np.array([N.get((root_key, a), 0) for a in range(A)], dtype=float)
        q_vec = np.array([Q.get((root_key, a), 0.0) for a in range(A)], dtype=float)
        a_star = int(counts.argmax()) if counts.sum() > 0 else int(q_vec.argmax())
        pi = counts / (counts.sum() + 1e-9)
        return a_star, q_vec, pi
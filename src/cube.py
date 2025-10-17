"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Tuple, Callable, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.cubies import (
    EdgeCubie,
    CornerCubie,
    CORNER_SLOTS,
    EDGE_SLOTS,
    CORNER_PIECE_COLORS,
    EDGE_PIECE_COLORS,
)

# Move tables (on *slot indices*, not on piece ids)
CORN_PERM = {
    "U": [1, 2, 3, 0, 4, 5, 6, 7],
    "D": [0, 1, 2, 3, 5, 6, 7, 4],
    "R": [4, 1, 2, 0, 7, 5, 6, 3],
    "L": [0, 5, 1, 3, 4, 6, 2, 7],
    "F": [1, 5, 2, 3, 0, 4, 6, 7],
    "B": [0, 1, 6, 2, 4, 5, 7, 3],
}
CORN_ORI = {
    "U": [0] * 8,
    "D": [0] * 8,
    "R": [2, 0, 0, 1, 1, 0, 0, 2],
    "L": [0, 1, 2, 0, 0, 2, 1, 0],
    "F": [1, 2, 0, 0, 2, 1, 0, 0],
    "B": [0, 0, 1, 2, 0, 0, 2, 1],
}
EDGE_PERM = {
    "U": [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11],
    "D": [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
    "R": [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
    "L": [0, 1, 9, 3, 4, 5, 10, 7, 8, 6, 2, 11],
    "F": [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
    # 'B':[0,1,2,10,4,5,6,11,8,9,3,7],
    "B": [0, 1, 2, 10, 4, 5, 6, 11, 8, 9, 7, 3],
}
EDGE_ORI = {
    "U": [0] * 12,
    "D": [0] * 12,
    "R": [0] * 12,
    "L": [0] * 12,
    "F": [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    "B": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
}


def invert_perm_and_delta(perm: list[int], delta: list[int], mod: int) -> tuple[list[int], list[int]]:
    """
    Invert a permutation table and its corresponding orientation deltas.

    Given a forward move permutation `perm` (mapping destination → source)
    and the list of orientation deltas `delta` applied to each destination,
    return their inverse counterparts such that:

        perm⁻¹[src] = dest
        delta⁻¹[src] = (-delta[dest]) % mod

    Args:
        perm: List of source indices per destination index.
        delta: List of orientation deltas corresponding to each destination index.
        mod: Modulus used for orientation arithmetic (3 for corners, 2 for edges).

    Returns:
        A tuple (inv_perm, inv_delta) giving the inverse mapping and deltas.
    """
    inv_perm = [0] * len(perm)
    inv_delta = [0] * len(perm)
    for dest_idx, src_idx in enumerate(perm):
        inv_perm[src_idx] = dest_idx
        inv_delta[src_idx] = (-delta[dest_idx]) % mod
    return inv_perm, inv_delta


def track_history(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for Cube.rotate: logs every atomic quarter-turn into `self._history`,
    unless history is disabled. Phase is taken from `self._phase` ("scramble"/"solve").
    """
    @wraps(method)
    def wrapper(self, face: str, clockwise: bool = True) -> Any:
        # call the real rotate first (state change is primary)
        result = method(self, face, clockwise)

        # lazy-init and maybe record
        if not hasattr(self, "_history") or not isinstance(self._history, pd.DataFrame):
            self._history = pd.DataFrame(columns=["step", "face", "clockwise", "phase"])
        if not hasattr(self, "_scramble_len"):
            self._scramble_len = 0
        if not hasattr(self, "_history_enabled"):
            self._history_enabled = True
        if not hasattr(self, "_phase"):
            self._phase = "solve"

        if self._history_enabled:
            step = int(self._history.shape[0])
            self._history.loc[step] = {
                "step": step,
                "face": face.upper(),
                "clockwise": bool(clockwise),
                "phase": self._phase,
            }
        return result
    return wrapper

class Cube:
    """
    High-level Rubik’s Cube state container built on explicit cubie objects.

    This class holds the full cube state as lists of `CornerCubie` and `EdgeCubie`
    instances.  It performs rotations using predefined permutation/orientation
    tables (`CORN_PERM`, `EDGE_PERM`, `CORN_ORI`, `EDGE_ORI`), and exposes
    conversion and visualization utilities.

    Design principles
    -----------------
    • The *cubies* (corner/edge objects) are the only mutable state.
      All higher-level representations (arrays, facelets, plots) are derived
      from them on demand.

    • Orientation (`ori`) is defined relative to the piece’s canonical
      sticker order and is updated according to the move tables.

    • Move tables act on slot indices rather than piece IDs; this ensures
      simple and invertible indexing logic.

    Attributes
    ----------
    corners : list[CornerCubie]
        The eight corner cubies in their current slots.
    edges : list[EdgeCubie]
        The twelve edge cubies in their current slots.
    _COLORS : dict[int, str]
        Mapping from face ID to a human-readable color name used for plots.

    Key methods
    ------------
    rotate(face, clockwise)
        Apply a face turn according to move tables.

    to_arrays()
        Convert cubie positions and orientations into canonical numeric arrays.

    to_facelets()
        Generate a 6×3×3 integer array of facelet colors (for rendering or export).

    plot_3d(), plot_net(), print_net()
        Visualization utilities: 3D matplotlib render, 2D face net, and
        terminal-printable layout respectively.

    assert_invariants()
        Check orientation-parity invariants (Σcorner_ori mod 3 = 0,
        Σedge_ori mod 2 = 0) for physical validity.

    test_move(m)
        Sanity test verifying that m + m' and m⁴ return the cube to its
        original state.

    Notes
    -----
    The coordinate convention follows the standard face ordering:

        0 = U (up/white)
        1 = R (right/blue)
        2 = F (front/orange)
        3 = D (down/yellow)
        4 = L (left/green)
        5 = B (back/red)

    The cube is oriented such that +z = up, +y = front, +x = right.

    Example
    -------
        c = Cube()
        c.rotate("R")
        c.print_net()
    """

    _COLORS = {0: "white", 1: "blue", 2: "orange", 3: "yellow", 4: "green", 5: "red"}

    def __init__(self):
        self.corners: List[CornerCubie] = [
            CornerCubie(
                slot_name=CORNER_SLOTS[i],
                ori=0,
                stickers=CORNER_PIECE_COLORS[i],
                piece_idx=i,
            )
            for i in range(8)
        ]
        self.edges: List[EdgeCubie] = [
            EdgeCubie(
                slot_name=EDGE_SLOTS[i],
                ori=0,
                stickers=EDGE_PIECE_COLORS[i],
                piece_idx=i,
            )
            for i in range(12)
        ]

        self._init_history_fields()

    def _init_history_fields(self) -> None:
        """
        Ensure history fields exist (idempotent).

        Creates:
            - self._history: pd.DataFrame with columns:
                ['step', 'face', 'clockwise', 'phase']
              where 'phase' is 'scramble' or 'solve'.
            - self._scramble_len: int, number of rows in history that belong to the scramble.
        """
        if not hasattr(self, "_history") or not isinstance(self._history, pd.DataFrame):
            self._history = pd.DataFrame(columns=["step", "face", "clockwise", "phase"])
        if not hasattr(self, "_scramble_len"):
            self._scramble_len = 0

    @contextmanager
    def history_phase(self, phase: str):
        """
        Temporarily set the history 'phase' for recorded moves ('scramble' or 'solve').
        Usage:
            with cube.history_phase('scramble'):
                cube.rotate('R'); cube.rotate('U', False)
        """
        prev = getattr(self, "_phase", "solve")
        self._phase = phase
        try:
            yield
        finally:
            self._phase = prev

    @contextmanager
    def no_history(self):
        """
        Temporarily disable history recording (e.g., for test_move or internal checks).
        """
        prev = getattr(self, "_history_enabled", True)
        self._history_enabled = False
        try:
            yield
        finally:
            self._history_enabled = prev

        def clear_history(self) -> None:
            """Clear the history DataFrame and reset the scramble checkpoint."""
            self._history = pd.DataFrame(columns=["step", "face", "clockwise", "phase"])
            self._scramble_len = 0

    def moves_since_scramble(self) -> int:
        """Number of moves logged after the scramble checkpoint."""
        return max(0, int(self._history.shape[0]) - int(self._scramble_len))

    def get_history(self) -> pd.DataFrame:
        """
        Return a copy of the move history DataFrame.

        Columns:
            step (int)         : 0-based move index
            face (str)         : 'U','D','R','L','F','B'
            clockwise (bool)   : True for CW, False for CCW
            phase (str)        : 'scramble' or 'solve'
        """
        self._init_history_fields()
        return self._history.copy()

    @track_history
    def rotate(self, face: str, clockwise: bool = True) -> None:
        """
        Apply a face rotation to the cube.

        The move tables (CORN_PERM / EDGE_PERM and CORN_ORI / EDGE_ORI)
        are defined on *slot indices*, not piece IDs. This method re-seats
        the affected cubies according to those tables and updates their
        orientations in place.

        Args:
            face: Face identifier ("U", "D", "R", "L", "F", "B").
            clockwise: If False, performs the inverse (counterclockwise) rotation.
        """
        cperm = CORN_PERM[face]
        cdel = CORN_ORI[face]
        eperm = EDGE_PERM[face]
        edel = EDGE_ORI[face]
        if not clockwise:
            cperm, cdel = invert_perm_and_delta(cperm, cdel, 3)
            eperm, edel = invert_perm_and_delta(eperm, edel, 2)

        # re-seat corner objects according to perm, updating ori in place
        old_corners = self.corners[:]
        new_corners = [None] * 8
        for dest_idx, src_idx in enumerate(cperm):
            cubie = old_corners[src_idx]
            cubie.move_to(CORNER_SLOTS[dest_idx], cdel[dest_idx])
            new_corners[dest_idx] = cubie
        self.corners = new_corners

        # re-seat edge objects
        old_edges = self.edges[:]
        new_edges = [None] * 12
        for dest_idx, src_idx in enumerate(eperm):
            cubie = old_edges[src_idx]
            cubie.move_to(EDGE_SLOTS[dest_idx], edel[dest_idx])
            new_edges[dest_idx] = cubie
        self.edges = new_edges

    def scramble(self, length: int = 25, seed: int | None = None) -> None:
        """
        Apply a random scramble and mark its length as the scramble checkpoint.

        Constraints:
            - No consecutive turns of the same face.
            - Softly avoid immediate same-axis repeats (UD, RL, FB), but permit if needed.

        Args:
            length: Number of quarter-turns to apply.
            seed: Optional RNG seed for reproducibility.

        Side effects:
            - Applies moves to the cube.
            - Records each move with phase='scramble'.
            - Sets self._scramble_len to the new total history length.
        """
        import random
        rng = random.Random(seed)
        faces = ["U", "D", "R", "L", "F", "B"]
        axis = {"U": "UD", "D": "UD", "R": "RL", "L": "RL", "F": "FB", "B": "FB"}
        prev_face = prev_axis = None

        with self.history_phase("scramble"):
            for _ in range(length):
                cand = [f for f in faces if f != prev_face and axis[f] != prev_axis] or \
                       [f for f in faces if f != prev_face]
                f = rng.choice(cand)
                cw = rng.random() < 0.5
                self.rotate(f, cw)
                prev_face, prev_axis = f, axis[f]

        self._scramble_len = int(self._history.shape[0])

    def solved_fraction(self) -> float:
        """
        Fraction of correctly placed & oriented pieces (corners+edges).

        - Corner correct if in home slot with ori % 3 == 0.
        - Edge   correct if in home slot with ori % 2 == 0.

        Returns:
            Float in [0, 1].
        """
        ok = 0
        for i, c in enumerate(self.corners):
            if c.slot_name == CORNER_SLOTS[i] and (c.ori % 3) == 0:
                ok += 1
        for i, e in enumerate(self.edges):
            if e.slot_name == EDGE_SLOTS[i] and (e.ori % 2) == 0:
                ok += 1
        return ok / 20.0

    def score(self) -> float:
        """
        Heuristic score that rewards being solved with fewer *post-scramble* moves.

            score = solved_fraction() / max(1, moves_since_scramble()
            edit we drop the denom to just solved_fraction() (for now, we're trying different reward policies)

        Notes
        -----
        - Ignores scramble length when penalizing: the timer starts after scrambling.
        - Swap this out later for a domain-specific objective if desired.
        """
        denom = max(1, self.moves_since_scramble())
        return self.solved_fraction()
    # ---------- VIEWS ----------
    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the cubie state into the canonical array representation.

        Returns:
            (corner_pos, corner_ori, edge_pos, edge_ori)
            - corner_pos[i]: index of the corner piece occupying slot i.
            - corner_ori[i]: orientation (0–2) of that corner.
            - edge_pos[i]: index of the edge piece occupying slot i.
            - edge_ori[i]: orientation (0–1) of that edge.
        """
        # positions are: which *piece id* currently sits in each slot index
        corner_pos = np.empty(8, dtype=np.int8)
        corner_ori = np.empty(8, dtype=np.int8)
        for i, slot in enumerate(CORNER_SLOTS):
            # find cubie whose slot_name == slot
            c = self.corners[i]  # by construction corners[i] lives at slot i
            piece_id = CORNER_SLOTS.index(self._home_slot_of_corner(c.stickers))
            corner_pos[i] = piece_id
            corner_ori[i] = c.ori

        edge_pos = np.empty(12, dtype=np.int8)
        edge_ori = np.empty(12, dtype=np.int8)
        for i, slot in enumerate(EDGE_SLOTS):
            e = self.edges[i]
            piece_id = EDGE_SLOTS.index(self._home_slot_of_edge(e.stickers))
            edge_pos[i] = piece_id
            edge_ori[i] = e.ori
        return corner_pos, corner_ori, edge_pos, edge_ori

    def to_facelets(self) -> np.ndarray:
        """
        Generate a 6×3×3 integer array of facelet colors from the cubie state.

        Each facelet position is filled according to the cubie’s stickers,
        slot orientation, and the fixed coordinate tables in CORNER/EDGE_FACELETS.

        Returns:
            A NumPy array F[6,3,3] of face color indices (0–5).
        """

        F = np.empty((6, 3, 3), dtype=int)
        for f in range(6):
            F[f, :, :] = f  # fill + centers

        for cubie in self.corners:
            for face, r, c, col in cubie.placements_for_slot():
                F[face, r, c] = col

        for cubie in self.edges:
            for face, r, c, col in cubie.placements_for_slot():
                F[face, r, c] = col

        return F

    # ---------- helpers ----------
    def _home_slot_of_corner(self, stickers: Tuple[int, int, int]) -> str:
        # piece id equals its home slot index; find by color tuple
        idx = CORNER_PIECE_COLORS.index(stickers)
        return CORNER_SLOTS[idx]

    def _home_slot_of_edge(self, stickers: Tuple[int, int]) -> str:
        idx = EDGE_PIECE_COLORS.index(stickers)
        return EDGE_SLOTS[idx]

    # quick sanity
    def assert_invariants(self) -> None:
        """
        Verify orientation parity invariants for a valid cube state.

        Raises:
            AssertionError: If the sum of corner orientations mod 3
                            or edge orientations mod 2 is non-zero.
        """
        co = sum(c.ori for c in self.corners) % 3
        eo = sum(e.ori for e in self.edges) % 2
        assert co == 0 and eo == 0, (co, eo)

    def test_move(self, m: str) -> None:
        """
        Test a single face move for internal consistency.

        Performs:
          - m followed by m'  → identity
          - m⁴                → identity
          - Orientation parity invariants check.

        Args:
            m: Face identifier ("U", "D", "R", "L", "F", "B").
        """

        snap = (
            [(c.slot_name, c.ori, c.stickers) for c in self.corners],
            [(e.slot_name, e.ori, e.stickers) for e in self.edges],
        )
        self.rotate(m, True)
        self.rotate(m, False)
        back = (
            [(c.slot_name, c.ori, c.stickers) for c in self.corners],
            [(e.slot_name, e.ori, e.stickers) for e in self.edges],
        )
        assert snap == back
        for _ in range(4):
            self.rotate(m, True)
        back2 = (
            [(c.slot_name, c.ori, c.stickers) for c in self.corners],
            [(e.slot_name, e.ori, e.stickers) for e in self.edges],
        )
        assert snap == back2
        self.assert_invariants()

    def plot_3d(self, ax: plt.Axes | None = None, figsize: tuple[int, int] = (6, 6), edgecolor: str = "k") -> None:
        """
        Render the cube in a 3D matplotlib view.

        The cube is centered at the origin with coordinates spanning [-1.5, 1.5]
        in each dimension. The B face is mirrored along the x-axis to maintain
        correct handedness relative to the facelet coordinate system.

        Args:
            ax: Optional matplotlib 3D axis to plot on. If None, creates a new figure.
            figsize: Size of the figure (if created internally).
            edgecolor: Edge color for square outlines.
        """
        F = self.to_facelets()

        # create axes if needed
        fig = None
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")

        ax.set_box_aspect([1, 1, 1])

        # face: (origin, u-axis, v-axis) in 3D space
        step = 1.0
        half = 0.5*3
        face_defs = {
            0: ((-half, -half, half), (step, 0, 0), (0, step, 0)),  # U  (z = +)
            3: ((-half, -half, -half), (step, 0, 0), (0, step, 0)),  # D  (z = -)
            2: ((-half, half, -half), (step, 0, 0), (0, 0, step)),  # F  (y = +)
            5: ((half, -half, -half), (-step, 0, 0), (0, 0, step)),  # B  (y = -)
            1: ((half, -half, -half), (0, step, 0), (0, 0, step)),  # R  (x = +)
            4: ((-half, -half, -half), (0, step, 0), (0, 0, step)),  # L  (x = -)
        }

        for f, (origin, du, dv) in face_defs.items():
            for r in range(3):
                for c in range(3):
                    col = self._COLORS[int(F[f, r, c])]
                    # square corners in (u,v) space → 3D
                    corners = []
                    for u, v in [(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]:
                        x = origin[0] + du[0] * u + dv[0] * v
                        y = origin[1] + du[1] * u + dv[1] * v
                        z = origin[2] + du[2] * u + dv[2] * v
                        corners.append((x, y, z))
                    poly = Poly3DCollection([corners])
                    poly.set_facecolor(col)
                    poly.set_edgecolor(edgecolor)
                    ax.add_collection3d(poly)

        ax.set_axis_off()
        # light padding around the cube
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        if fig is not None:
            plt.show()

    def print_net(self, use_color: bool = True) -> None:
        """
        Print a compact text-based cube net to the terminal.

              [U]
        [L] [F] [R] [B]
              [D]

        Args:
            use_color: If True, apply ANSI color codes to facelet numbers
                       for readability in supported terminals.
        """

        F = self.to_facelets()
        # face units (0..3), we’ll scale by 3 below
        layout = {
            0: (0, 1),  # U above F
            4: (1, 0),  # L F R B in a row
            2: (1, 1),
            1: (1, 2),
            5: (1, 3),
            3: (2, 1),  # D below F
        }

        COLOR_CODES = {
            0: "\033[97m",
            1: "\033[94m",
            2: "\033[95m",
            3: "\033[93m",
            4: "\033[92m",
            5: "\033[91m",
        }
        RESET = "\033[0m"

        SCALE = 3  # 3 stickers per face side
        max_rt = max(rt for rt, _ in layout.values())
        max_ct = max(ct for _, ct in layout.values())
        rows = (max_rt + 1) * SCALE
        cols = (max_ct + 1) * SCALE
        grid = [[" " for _ in range(cols)] for _ in range(rows)]

        for face_id, (rt, ct) in layout.items():
            top = rt * SCALE
            left = ct * SCALE
            for r in range(3):
                for c in range(3):
                    rr = top + r
                    cc = left + c
                    val = int(F[face_id, r, c])
                    grid[rr][cc] = (
                        f"{COLOR_CODES[val]}{val}{RESET}" if use_color else str(val)
                    )

        for row in grid:
            print(" ".join(row))

    def is_solved(self)->bool:
        """
        Check if the cube is in a solved state.
        :return:
        """
        for i, c in enumerate(self.corners):
            if c.slot_name != CORNER_SLOTS[i] or (c.ori % 3) != 0:
                return False
        for i, e in enumerate(self.edges):
            if e.slot_name != EDGE_SLOTS[i] or (e.ori % 2) != 0:
                return False
        return True

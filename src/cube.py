"""Core Rubik's Cube state model and rendering helpers."""
import os
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "cubaroo-cache")
_MPL_CONFIG_DIR = os.path.join(_CACHE_DIR, "matplotlib")
os.makedirs(_MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CONFIG_DIR)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .cubies import (
    EdgeCubie,
    CornerCubie,
    CORNER_SLOTS,
    EDGE_SLOTS,
    CORNER_PIECE_COLORS,
    EDGE_PIECE_COLORS,
)
from .visualisation.utils_visualization import _color_lookup, _unit_cubie_quads

__all__ = ["Cube", "VALID_FACES", "HISTORY_COLUMNS"]

# Move tables (on *slot indices*, not on piece ids)
CORN_PERM = {
    "U": [3, 0, 1, 2, 4, 5, 6, 7],
    "D": [0, 1, 2, 3, 5, 6, 7, 4],
    "R": [4, 1, 2, 0, 7, 5, 6, 3],
    "L": [0, 2, 6, 3, 4, 1, 5, 7],
    "F": [1, 5, 2, 3, 0, 4, 6, 7],
    "B": [0, 1, 3, 7, 4, 5, 2, 6],
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
    "U": [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
    "D": [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
    "R": [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
    "L": [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
    "F": [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
    # 'B':[0,1,2,10,4,5,6,11,8,9,3,7],
    "B": [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
}
EDGE_ORI = {
    "U": [0] * 12,
    "D": [0] * 12,
    "R": [0] * 12,
    "L": [0] * 12,
    "F": [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    "B": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
}

VALID_FACES: tuple[str, ...] = ("U", "D", "R", "L", "F", "B")
HISTORY_COLUMNS: list[str] = ["step", "face", "clockwise", "phase"]
EXPECTED_CORNER_PIECES = set(range(len(CORNER_SLOTS)))
EXPECTED_EDGE_PIECES = set(range(len(EDGE_SLOTS)))


def _permutation_parity(values: list[int]) -> int:
    """Return permutation parity: 0 for even, 1 for odd."""
    parity = 0
    seen = [False] * len(values)
    for start in range(len(values)):
        if seen[start]:
            continue
        cycle_len = 0
        idx = start
        while not seen[idx]:
            seen[idx] = True
            idx = values[idx]
            cycle_len += 1
        if cycle_len > 0:
            parity ^= (cycle_len - 1) & 1
    return parity


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
            self._history = pd.DataFrame(columns=HISTORY_COLUMNS)
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
            self._history = pd.DataFrame(columns=HISTORY_COLUMNS)
        if not hasattr(self, "_scramble_len"):
            self._scramble_len = 0
        if not hasattr(self, "_history_enabled"):
            self._history_enabled = True
        if not hasattr(self, "_phase"):
            self._phase = "solve"

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
        """Clear the move history and reset the scramble checkpoint."""
        self._history = pd.DataFrame(columns=HISTORY_COLUMNS)
        self._scramble_len = 0
        self._phase = "solve"
        self._history_enabled = True

    def copy(self) -> "Cube":
        """Return a deep copy of the cube state."""
        return deepcopy(self)

    def state_key(
        self,
    ) -> tuple[tuple[tuple[int | None, int], ...], tuple[tuple[int | None, int], ...]]:
        """Return a hashable representation of the current cube state."""
        return (
            tuple((corner.piece_idx, corner.ori) for corner in self.corners),
            tuple((edge.piece_idx, edge.ori) for edge in self.edges),
        )

    def consistency_issues(self) -> list[str]:
        """Return structural issues found in the current cube representation."""
        issues: list[str] = []

        if len(self.corners) != len(CORNER_SLOTS):
            issues.append(f"Expected {len(CORNER_SLOTS)} corners, found {len(self.corners)}.")
        if len(self.edges) != len(EDGE_SLOTS):
            issues.append(f"Expected {len(EDGE_SLOTS)} edges, found {len(self.edges)}.")
        if issues:
            return issues

        corner_slots = [corner.slot_name for corner in self.corners]
        edge_slots = [edge.slot_name for edge in self.edges]
        if corner_slots != CORNER_SLOTS:
            issues.append("Corner slots are not aligned with canonical slot order.")
        if edge_slots != EDGE_SLOTS:
            issues.append("Edge slots are not aligned with canonical slot order.")

        for corner in self.corners:
            try:
                corner.validate()
            except ValueError as exc:
                issues.append(str(exc))
        for edge in self.edges:
            try:
                edge.validate()
            except ValueError as exc:
                issues.append(str(exc))

        corner_piece_ids = [corner.piece_idx for corner in self.corners]
        edge_piece_ids = [edge.piece_idx for edge in self.edges]
        if any(piece is None for piece in corner_piece_ids):
            issues.append("Corner piece indices must be assigned for all corners.")
        if any(piece is None for piece in edge_piece_ids):
            issues.append("Edge piece indices must be assigned for all edges.")

        if not issues:
            if set(int(piece) for piece in corner_piece_ids) != EXPECTED_CORNER_PIECES:
                issues.append("Corner piece indices are not a complete permutation of 0..7.")
            if set(int(piece) for piece in edge_piece_ids) != EXPECTED_EDGE_PIECES:
                issues.append("Edge piece indices are not a complete permutation of 0..11.")

        corner_stickers = [corner.stickers for corner in self.corners]
        edge_stickers = [edge.stickers for edge in self.edges]
        if len(set(corner_stickers)) != len(corner_stickers):
            issues.append("Corner sticker tuples must be unique.")
        if len(set(edge_stickers)) != len(edge_stickers):
            issues.append("Edge sticker tuples must be unique.")
        if set(corner_stickers) != set(CORNER_PIECE_COLORS):
            issues.append("Corner sticker tuples do not match the canonical corner pieces.")
        if set(edge_stickers) != set(EDGE_PIECE_COLORS):
            issues.append("Edge sticker tuples do not match the canonical edge pieces.")

        co = sum(corner.ori for corner in self.corners) % 3
        eo = sum(edge.ori for edge in self.edges) % 2
        if co != 0:
            issues.append(f"Corner orientation parity invalid: sum mod 3 = {co}.")
        if eo != 0:
            issues.append(f"Edge orientation parity invalid: sum mod 2 = {eo}.")

        if not issues:
            corner_perm = [int(piece) for piece in corner_piece_ids]
            edge_perm = [int(piece) for piece in edge_piece_ids]
            if _permutation_parity(corner_perm) != _permutation_parity(edge_perm):
                issues.append("Corner and edge permutation parity do not match.")

        facelets = self.to_facelets()
        if facelets.shape != (6, 3, 3):
            issues.append(f"Facelet array shape must be (6, 3, 3), got {facelets.shape}.")
        else:
            unique, counts = np.unique(facelets, return_counts=True)
            color_counts = dict(zip(unique.tolist(), counts.tolist()))
            for color in range(6):
                if color_counts.get(color, 0) != 9:
                    issues.append(
                        f"Color {color} should appear exactly 9 times, found {color_counts.get(color, 0)}."
                    )

        return issues

    def assert_consistent(self) -> None:
        """Raise when the current cube representation is structurally inconsistent."""
        issues = self.consistency_issues()
        if issues:
            raise ValueError("; ".join(issues))

    @staticmethod
    def _validate_face(face: str) -> str:
        normalized = face.upper()
        if normalized not in VALID_FACES:
            raise ValueError(f"Unsupported face {face!r}. Expected one of {VALID_FACES}.")
        return normalized

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
        face = self._validate_face(face)
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
        if length < 0:
            raise ValueError("Scramble length must be non-negative.")

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
            if c.piece_idx == i and (c.ori % 3) == 0:
                ok += 1
        for i, e in enumerate(self.edges):
            if e.piece_idx == i and (e.ori % 2) == 0:
                ok += 1
        return ok / 20.0

    def score(self) -> float:
        """
        Return the built-in baseline score for the cube state.

        The core cube object intentionally keeps this simple and stable:
        it reports the fraction of solved cubies and leaves richer reward
        shaping to the dedicated scorers in :mod:`src.scorer`.
        """
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
        for i, corner in enumerate(self.corners):
            piece_id = corner.piece_idx
            if piece_id is None:
                piece_id = CORNER_SLOTS.index(self._home_slot_of_corner(corner.stickers))
            corner_pos[i] = piece_id
            corner_ori[i] = corner.ori

        edge_pos = np.empty(12, dtype=np.int8)
        edge_ori = np.empty(12, dtype=np.int8)
        for i, edge in enumerate(self.edges):
            piece_id = edge.piece_idx
            if piece_id is None:
                piece_id = EDGE_SLOTS.index(self._home_slot_of_edge(edge.stickers))
            edge_pos[i] = piece_id
            edge_ori[i] = edge.ori
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
        issues = [
            issue for issue in self.consistency_issues()
            if "orientation parity" in issue or "permutation parity" in issue
        ]
        if issues:
            raise AssertionError("; ".join(issues))

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

    def plot_3d_cubies(
            self,
            ax: plt.Axes | None = None,
            figsize: tuple[int, int] = (6, 6),
            edgecolor: str = "k",
            pose_override: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
            cubie_size: float = 0.94,
    ) -> None:
        """
        Render as 27 cubies with stickers.
        Colors come from F = to_facelets() (6×3×3) — already correct in your code.
        `pose_override` lets the animator rotate/translate specific cubies per frame:
          pose_override[cubie_idx] = (R(3×3), t(3,))
        Cubie indexing used here: idx = (ix+1)*9 + (iy+1)*3 + (iz+1), for ix,iy,iz in {-1,0,1}.
        """
        F = self.to_facelets()

        fig = None
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

        # Precompute local quads at origin
        local_quads = _unit_cubie_quads(size=cubie_size)

        coords = [-1.0, 0.0, 1.0]

        def cubie_index(ix: int, iy: int, iz: int) -> int:
            return (ix + 1) * 9 + (iy + 1) * 3 + (iz + 1)

        for ix, x in enumerate(coords, start=-1):
            for iy, y in enumerate(coords, start=-1):
                for iz, z in enumerate(coords, start=-1):
                    idx = cubie_index(ix, iy, iz)

                    # Default pose: identity rotation, translated to grid center
                    R = np.eye(3)
                    t = np.array([x, y, z], dtype=float)

                    # Animator may override (for the rotating layer)
                    if pose_override and idx in pose_override:
                        R, t = pose_override[idx]

                    # Which of the 6 faces are exposed at the hull?
                    visible = []
                    if np.isclose(z, 1.0): visible.append('U')
                    if np.isclose(z, -1.0): visible.append('D')
                    if np.isclose(y, 1.0): visible.append('F')
                    if np.isclose(y, -1.0): visible.append('B')
                    if np.isclose(x, 1.0): visible.append('R')
                    if np.isclose(x, -1.0): visible.append('L')

                    # Transform and draw visible faces
                    for face_name in visible:
                        quad_local = local_quads[face_name]  # (4,3) at origin
                        quad_world = (R @ quad_local.T).T + t  # (4,3)
                        color_idx = _color_lookup(F, face_name, x, y, z)
                        col = self._COLORS[color_idx]

                        poly = Poly3DCollection([quad_world], linewidths=0.5)
                        poly.set_facecolor(col)
                        poly.set_edgecolor(edgecolor)
                        ax.add_collection3d(poly)

        ax.set_axis_off()
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_zlim(-1.6, 1.6)
        if fig is not None:
            plt.show()


    def format_net(self, use_color: bool = True) -> str:
        """Return a compact text net representation of the cube."""
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

        return "\n".join(" ".join(row) for row in grid)

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
        print(self.format_net(use_color=use_color))

    def is_solved(self) -> bool:
        """Return ``True`` when every cubie is in its home slot and orientation."""
        for i, c in enumerate(self.corners):
            if c.piece_idx != i or (c.ori % 3) != 0:
                return False
        for i, e in enumerate(self.edges):
            if e.piece_idx != i or (e.ori % 2) != 0:
                return False
        return True

    def __repr__(self) -> str:
        return f"Cube(solved={self.is_solved()}, moves_since_scramble={self.moves_since_scramble()})"

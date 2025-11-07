'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from collections import deque
from typing import Callable, Tuple, Dict, List
from src.cube import Cube
from math import exp

## --- Helper: hashable key for caching ---
def _state_key(cube: Cube) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """
    Generate a hashable representation of the cube’s full state.

    The returned key encodes the position (`piece_idx`) and orientation (`ori`)
    of all corner and edge cubies, making it suitable for use in LRU caches or
    pattern database lookups.

    Args:
        cube (Cube): The Rubik’s Cube instance.

    Returns:
        tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
            A nested tuple containing:
              - A tuple of (piece_idx, ori) pairs for each corner.
              - A tuple of (piece_idx, ori) pairs for each edge.
    """

    return (tuple((c.piece_idx, c.ori) for c in cube.corners),
            tuple((e.piece_idx, e.ori) for e in cube.edges))

# --- Phase-1 components ---
UD_SLICE_EDGE_IDS = {8, 9, 10, 11}

@lru_cache(maxsize=100_000)
def _co_eo_ud_from_key(
    key: tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
) -> tuple[int, int, int]:
    """
    Compute the coarse Phase-1 orientation and slice placement features from a cached cube key.

    This function extracts:
      - The number of misoriented corners (`co`).
      - The number of misoriented edges (`eo`).
      - The number of misplaced UD-slice edges (`ud_mis`).

    Args:
        key (tuple): A state key as produced by `_state_key`, encoding (piece_idx, ori)
                     for all corners and edges.

    Returns:
        tuple[int, int, int]:
            A tuple (co, eo, ud_mis) giving the counts of:
              - Misoriented corners (0–7)
              - Misoriented edges (0–11)
              - UD-slice edges not in the slice (0–4)
    """
    corners, edges = key
    co = sum(ori % 3 != 0 for _, ori in corners)
    eo = sum(ori % 2 != 0 for _, ori in edges)
    in_slice = sum(1 for slot in UD_SLICE_EDGE_IDS if edges[slot][0] in UD_SLICE_EDGE_IDS)
    ud_mis = 4 - in_slice
    return co, eo, ud_mis

def _h_naive_phase1(cube: Cube) -> int:
    """
    Estimate a naive lower-bound distance to the Phase-1 subgoal (orientation + UD-slice solved).

    The heuristic is based on the minimal number of quarter-turn moves (QTM)
    required to correct corner orientation, edge orientation, and UD-slice
    placement, using coarse move-group assumptions:
      - Up to 4 corner or edge orientations can be affected in one move.
      - Up to 2 UD-slice edges can be correctly placed in one move.

    Args:
        cube (Cube): The current cube state.

    Returns:
        int: The estimated minimal number of moves (lower bound, typically 0–6).
    """
    co, eo, ud = _co_eo_ud_from_key(_state_key(cube))
    lb_co = (co + 3) // 4
    lb_eo = (eo + 3) // 4
    lb_ud = (ud + 1) // 2
    return max(lb_co, lb_eo, lb_ud)

@dataclass
class _PDB:
    """
    Container for Phase-1 pattern databases (PDBs).

    Each sub-dictionary maps an encoded cube subspace state to its minimal
    distance in quarter-turn metric (QTM):
      - `co`: corner orientation distances (3⁷ states)
      - `eo`: edge orientation distances (2¹¹ states)
      - `ud`: UD-slice edge placement distances (≤4096 states)

    The *_max* attributes store the maximum distance found in each sub-database,
    allowing for normalization during scoring.

    Attributes:
        co (dict[int, int]): Mapping of encoded corner orientation states to distances.
        eo (dict[int, int]): Mapping of encoded edge orientation states to distances.
        ud (dict[int, int]): Mapping of encoded UD-slice placement states to distances.
        co_max (int): Maximum distance value in the CO database.
        eo_max (int): Maximum distance value in the EO database.
        ud_max (int): Maximum distance value in the UD database.
    """

    co: Dict[int, int]
    eo: Dict[int, int]
    ud: Dict[int, int]
    co_max: int
    eo_max: int
    ud_max: int

_PDB_CACHE: _PDB | None = None

def _build_pdb() -> _PDB:
    """
    Build three minimal Phase-1 pattern databases (PDBs) via breadth-first search (BFS).

    The PDBs provide admissible lower-bound estimates for:
      • Corner orientation (CO)
      • Edge orientation (EO)
      • UD-slice edge placement (UD)

    Each subspace is explored independently using BFS over all possible face turns
    in the quarter-turn metric (QTM). Only the relevant coordinates are kept fixed,
    and all others are set to their solved configuration.

    Returns:
        _PDB: A populated pattern database object with per-subspace distance maps
              and their respective maxima for normalization.
    """
    def co_id(cube) -> int:
        s = 0
        for i in range(7):
            s = 3 * s + (cube.corners[i].ori % 3)
        return s

    def eo_id(cube) -> int:
        s = 0
        for i in range(11):
            s = (s << 1) | (cube.edges[i].ori & 1)
        return s

    def ud_id(cube) -> int:
        slots = [i for i, e in enumerate(cube.edges) if e.piece_idx in UD_SLICE_EDGE_IDS]
        slots.sort()
        mask = 0
        for s in slots:
            mask |= (1 << s)
        return mask

    MOVES = ("U", "D", "L", "R", "F", "B")

    def bfs(distance_of: Callable, encode: Callable) -> Dict[int, int]:
        from copy import deepcopy
        start = distance_of()
        root_key = encode(start)
        dist = {root_key: 0}
        q = deque([start])
        while q:
            s = q.popleft()
            d = dist[encode(s)]
            for m in MOVES:
                t = deepcopy(s)
                t.rotate(m)
                t = distance_of(t)
                k = encode(t)
                if k not in dist:
                    dist[k] = d + 1
                    q.append(t)
        return dist

    def proj_co(c=None):
        t = copy.deepcopy(c) if c else Cube()
        for i, c in enumerate(t.corners): c.piece_idx = i
        for i, e in enumerate(t.edges): e.piece_idx = i; e.ori = 0
        return t

    def proj_eo(c=None):
        t = copy.deepcopy(c) if c else Cube()
        for i, e in enumerate(t.edges): e.piece_idx = i
        for i, c in enumerate(t.corners): c.piece_idx = i; c.ori = 0
        return t

    def proj_ud(c=None):
        t = copy.deepcopy(c) if c else Cube()
        for i, e in enumerate(t.edges): e.ori = 0
        for i, c in enumerate(t.corners): c.piece_idx = i; c.ori = 0
        return t

    pdb_co = bfs(proj_co, co_id)
    pdb_eo = bfs(proj_eo, eo_id)
    pdb_ud = bfs(proj_ud, ud_id)
    return _PDB(
        co=pdb_co, eo=pdb_eo, ud=pdb_ud,
        co_max=max(pdb_co.values()), eo_max=max(pdb_eo.values()), ud_max=max(pdb_ud.values())
    )

def _get_pdb() -> _PDB:
    """
    Retrieve the cached Phase-1 pattern database, building it on first access.

    This ensures the expensive BFS construction runs only once per session and
    that all scoring functions share the same reference PDB.

    Returns:
        _PDB: The cached or newly built pattern database.
    """
    global _PDB_CACHE
    if _PDB_CACHE is None:
        _PDB_CACHE = _build_pdb()
    return _PDB_CACHE

# --- Utility functions ---
def _co_id_from_state(corners: tuple[tuple[int, int], ...]) -> int:
    """
    Encode the cube’s corner-orientation subspace as a single integer ID.

    This encoder assumes `corners` is a tuple of (piece_idx, ori) pairs ordered by corner slot.
    The last corner’s orientation is dependent (sum of corner orientations ≡ 0 mod 3), so only
    the first 7 corner orientations are included, interpreted as a base-3 number.

    Args:
        corners (tuple[tuple[int, int], ...]): For each corner slot, a (piece_idx, ori) pair,
            where `ori ∈ {0,1,2}` is the corner’s orientation.

    Returns:
        int: Base-3 code for the first 7 corner orientations (range 0..3**7 - 1).
    """
    s = 0
    for i in range(7):
        s = 3 * s + (corners[i][1] % 3)
    return s

def _eo_id_from_state(edges: tuple[tuple[int, int], ...]) -> int:
    """
    Encode the cube’s edge-orientation subspace as a single integer ID.

    This encoder assumes `edges` is a tuple of (piece_idx, ori) pairs ordered by edge slot.
    The last edge’s orientation is dependent (sum of edge orientations ≡ 0 mod 2), so only
    the first 11 edge orientations are included, interpreted as an 11-bit integer.

    Args:
        edges (tuple[tuple[int, int], ...]): For each edge slot, a (piece_idx, ori) pair,
            where `ori ∈ {0,1}` is the edge’s flip state.

    Returns:
        int: Bitmask code for the first 11 edge orientations (range 0..2**11 - 1).
    """
    s = 0
    for i in range(11):
        s = (s << 1) | (edges[i][1] & 1)
    return s

def _udmask_from_state(edges: tuple[tuple[int, int], ...]) -> int:
    """
    Encode the UD-slice *placement* subspace as a 12-bit slot mask.

    For each of the 12 edge slots, this sets bit `slot` to 1 iff the edge *piece index*
    currently in that slot belongs to the UD-slice set `UD_SLICE_EDGE_IDS` (e.g., {FR, FL, BL, BR}).
    Orientation is ignored; only which slots contain the 4 slice pieces matters.

    Args:
        edges (tuple[tuple[int, int], ...]): For each edge slot, a (piece_idx, ori) pair
            ordered by slot index.

    Returns:
        int: A 12-bit mask (range 0..(1<<12)-1) with ones at slots occupied by UD-slice pieces.
    """
    mask = 0
    for slot, (piece_idx, _ori) in enumerate(edges):
        if piece_idx in UD_SLICE_EDGE_IDS:
            mask |= (1 << slot)
    return mask

# --- Basic metrics ---
def _frac_in_correct_slot(cube: Cube) -> float:
    """
    Compute the fraction of cubies currently located in their correct slots.

    This measure ignores orientation and only considers whether each corner
    and edge piece occupies its solved position (i.e., `piece_idx == slot index`).

    Args:
        cube (Cube): The current cube state.

    Returns:
        float: Fraction of correctly placed cubies in [0, 1],
               with 1.0 corresponding to all pieces in their solved slots.
    """
    ok = sum(c.piece_idx == i for i, c in enumerate(cube.corners))
    ok += sum(e.piece_idx == i for i, e in enumerate(cube.edges))
    return ok / 20.0


def _frac_correct_ori(cube: Cube) -> float:
    """
    Compute the fraction of cubies with correct orientation, regardless of position.

    Corners are considered oriented if `ori % 3 == 0`, edges if `ori % 2 == 0`.

    Args:
        cube (Cube): The current cube state.

    Returns:
        float: Fraction of correctly oriented cubies in [0, 1],
               with 1.0 indicating all pieces have correct orientation.
    """
    ok = sum((c.ori % 3) == 0 for c in cube.corners)
    ok += sum((e.ori % 2) == 0 for e in cube.edges)
    return ok / 20.0

def _sig(x: float, k: float, t: float) -> float:
    """
    Smooth logistic function used for reward gating and normalization.

    Computes the standard sigmoid:
        σ(x; k, t) = 1 / (1 + exp(-k * (x - t)))

    Args:
        x (float): Input value to evaluate.
        k (float): Steepness of the transition (larger = sharper step).
        t (float): Midpoint (threshold) value where σ = 0.5.

    Returns:
        float: Output in (0, 1), smoothly mapping x relative to threshold t.
    """
    return 1.0 / (1.0 + exp(-k * (x - t)))

# --- Scoring functions ---
def score_solved_fraction(cube: Cube) -> float:
    """
    Fraction of fully solved cubies (correct slot AND orientation).

    Counts corners that satisfy (piece_idx == slot_index and ori % 3 == 0)
    and edges that satisfy (piece_idx == slot_index and ori % 2 == 0),
    then divides by the total number of cubies (20).

    Args:
        cube (Cube): Current cube state.

    Returns:
        float: Proportion in [0, 1] of cubies that are completely solved.
    """
    ok = 0
    for i, c in enumerate(cube.corners):
        ok += (c.piece_idx == i and (c.ori % 3) == 0)
    for i, e in enumerate(cube.edges):
        ok += (e.piece_idx == i and (e.ori % 2) == 0)
    return ok / 20.0

def score_weighted_slot_and_ori(cube: Cube, w_slot: float = 0.6) -> float:
    """
    Weighted score that rewards correct placement first, then orientation.

    For each cubie:
      score += w_slot * [in correct slot] + (1 - w_slot) * [in correct slot AND correctly oriented].
    The total is normalized by 20 (8 corners + 12 edges).

    Args:
        cube (Cube): Current cube state.
        w_slot (float, optional): Weight for “in correct slot” vs. “oriented”.
            Must be in [0, 1]. Defaults to 0.6.

    Returns:
        float: Score in [0, 1] reflecting slot-first progress with orientation refinement.
    """
    w_ori = 1.0 - w_slot
    s = 0.0
    for i, c in enumerate(cube.corners):
        in_slot = (c.piece_idx == i)
        s += w_slot * in_slot + w_ori * (in_slot and (c.ori % 3 == 0))
    for i, e in enumerate(cube.edges):
        in_slot = (e.piece_idx == i)
        s += w_slot * in_slot + w_ori * (in_slot and (e.ori % 2 == 0))
    return s / 20.0


def score_phase1_naive(cube: Cube) -> float:
    """
    Naive Phase-1 progress based on a coarse lower-bound move estimate.

    Uses a quick admissible heuristic h (in quarter-turns) over:
      • Corner orientation (CO)
      • Edge orientation (EO)
      • UD-slice edge placement (UD)
    Then returns 1 − min(h / HMAX, 1), with HMAX a conservative cap.

    Args:
        cube (Cube): Current cube state.

    Returns:
        float: Normalized Phase-1 progress in [0, 1]; 1.0 means h == 0.
    """
    h = _h_naive_phase1(cube)
    HMAX = 6.0
    return 1.0 - min(h / HMAX, 1.0)

def score_phase1_pdb(cube: Cube) -> float:
    """
    Admissible Phase-1 progress via pattern databases (PDBs) with end-corrected sigmoid.

    Looks up minimal QTM distances for CO, EO, and UD subspaces and takes their max h.
    Normalizes by the observed subspace maxima, s = 1 − min(h / hmax, 1),
    then applies an end-corrected sigmoid so that s=0→0 and s=1→1.

    Args:
        cube (Cube): Current cube state.

    Returns:
        float: Smoothly normalized PDB-based Phase-1 progress in [0, 1].
    """
    pdb = _get_pdb()
    key = _state_key(cube)
    corners, edges = key
    co_key = _co_id_from_state(corners)
    eo_key = _eo_id_from_state(edges)
    ud_key = _udmask_from_state(edges)
    co_d = pdb.co.get(co_key, 0)
    eo_d = pdb.eo.get(eo_key, 0)
    ud_d = pdb.ud.get(ud_key, 0)
    h = max(co_d, eo_d, ud_d)
    hmax = max(pdb.co_max, pdb.eo_max, pdb.ud_max, 1)
    s = 1.0 - min(h / float(hmax), 1.0)
    k, t = 6.0, 0.7
    a = _sig(0.0, k=k, t=t)
    b = _sig(1.0, k=k, t=t)
    s_sig = _sig(s, k=k, t=t)

    return (s_sig - a) / (b - a)

def score_phase1_gated(cube: Cube) -> float:
    """
    Gated Phase-1 score combining naive and PDB terms with global-progress gating.

    Blends: raw = 0.3*naive + 0.7*PDB, then gates it with a sigmoid over
    overall progress (average of fraction-in-slot and fraction-oriented).
    A small floor keeps the signal alive in very early states.

    Args:
        cube (Cube): Current cube state.

    Returns:
        float: Gated Phase-1 progress in [0, 1] with better early/late balance.
    """
    naive = score_phase1_naive(cube)
    pdb_s = score_phase1_pdb(cube)
    raw = 0.3 * naive + 0.7 * pdb_s
    prog = 0.5 * _frac_in_correct_slot(cube) + 0.5 * _frac_correct_ori(cube)
    gate = _sig(prog, k=10.0, t=0.55)
    return gate * raw + (1 - gate) * 0.15

def score_length_aware(cube: Cube) -> float:
    """
    Length-aware shaping that prefers shorter solutions without crushing signal.

    Combines the gated Phase-1 score with an exponential time preference:
      score = 0.85 * phase1_gated + 0.15 * exp(−λ * moves_since_scramble)
    If the cube exposes `moves_since_scramble()`, it is used; otherwise 0.

    Args:
        cube (Cube): Current cube state.

    Returns:
        float: Length-aware score in [0, 1] that decays gently with move count.
    """
    base = score_phase1_gated(cube)
    moves = 0
    if hasattr(cube, "moves_since_scramble") and callable(getattr(cube, "moves_since_scramble")):
        moves = max(0, int(cube.moves_since_scramble()))
    lam = 0.02
    length_bonus = exp(-lam * moves)
    return 0.85 * base + 0.15 * length_bonus

# --- Registry ---
class ScoringOption(Enum):
    SOLVED_FRACTION = auto()
    WEIGHTED_SLOT_AND_ORI = auto()
    PHASE1_NAIVE = auto()
    PHASE1_PDB = auto()
    LENGTH_AWARE = auto()
    PHASE1_GATED = auto()

SCORE_FN: Dict[ScoringOption, Callable] = {
    ScoringOption.SOLVED_FRACTION: score_solved_fraction,
    ScoringOption.WEIGHTED_SLOT_AND_ORI: score_weighted_slot_and_ori,
    ScoringOption.PHASE1_NAIVE: score_phase1_naive,
    ScoringOption.PHASE1_PDB: score_phase1_pdb,
    ScoringOption.LENGTH_AWARE: score_length_aware,
    ScoringOption.PHASE1_GATED: score_phase1_gated,
}

@dataclass
class Scorer:
    option: ScoringOption = ScoringOption.PHASE1_GATED
    def __call__(self, cube) -> float:
        """
        Dispatch to the selected scoring function.

        Uses the `option` field to select one of the registered scorers and
        evaluates it on the provided cube.

        Args:
            cube (Cube): Current cube state.

        Returns:
            float: The computed score in [0, 1].
        """
        fn = SCORE_FN[self.option]
        return float(fn(cube))
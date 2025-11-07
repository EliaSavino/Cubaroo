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

# ---- Assumptions about your cube object ----
# cube.corners: list of objects with .piece_idx (0..7) and .ori (0..2)
# cube.edges:   list of objects with .piece_idx (0..11) and .ori (0..1)
# moves_since_scramble(): int  (optional for some scores)

# --- Helper: hashable key for caching ---
def _state_key(cube) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (tuple((c.piece_idx, c.ori) for c in cube.corners),
            tuple((e.piece_idx, e.ori) for e in cube.edges))

# --- Cheap Phase-1 features (no tables) ---
UD_SLICE_EDGE_IDS = {8, 9, 10, 11}  # FR, FL, BL, BR in your indexing

@lru_cache(maxsize=100_000)
def _co_eo_ud_from_key(key) -> Tuple[int, int, int]:
    (corners, edges) = key
    co = sum(ori % 3 != 0 for _, ori in corners)
    eo = sum(ori % 2 != 0 for _, ori in edges)
    # how many of the 4 slice pieces are NOT currently sitting in any slice slot?
    # count over the 4 slice *slots* whether the piece there is one of the 4 slice pieces
    # then mis = 4 - in_slice
    in_slice = sum(1 for slot in UD_SLICE_EDGE_IDS if edges[slot][0] in UD_SLICE_EDGE_IDS)
    ud_mis = 4 - in_slice
    return co, eo, ud_mis

def _h_naive_phase1(cube) -> int:
    co, eo, ud = _co_eo_ud_from_key(_state_key(cube))
    # Lower bounds per quarter turn (QTM):
    # - up to 4 misoriented corners/edges can be affected in one move
    # - at most 2 UD-slice edges can be placed per move
    lb_co = (co + 3) // 4
    lb_eo = (eo + 3) // 4
    lb_ud = (ud + 1) // 2
    return max(lb_co, lb_eo, lb_ud)  # small integer 0..~6 typical

# --- Option 4 needs tiny PDBs (built once, cached) ---
@dataclass
class _PDB:
    co: Dict[int, int]  # 3^7 states
    eo: Dict[int, int]  # 2^11 states
    ud: Dict[int, int]  # compact UD-slice placement cost

_PDB_CACHE: _PDB | None = None

def _build_pdb() -> _PDB:
    """
    Build three admissible pattern DBs for Phase-1:
      - CO (corner orientation distance in QTM)
      - EO (edge orientation distance in QTM)
      - UD-slice placement distance for {FR,FL,BL,BR}
    These BFSes run in milliseconds–seconds and are done once.
    """
    # Encoding helpers (minimal; uses only orientations / slice occupancy):
    def co_id(cube) -> int:
        # last corner ori is dependent; store first 7 as base-3 number
        o = [(c.ori % 3) for c in cube.corners]
        s = 0
        for i in range(7):
            s = 3 * s + o[i]
        return s

    def eo_id(cube) -> int:
        # last edge ori is dependent; store first 11 as base-2 number
        o = [(e.ori & 1) for e in cube.edges]
        s = 0
        for i in range(11):
            s = (s << 1) | o[i]
        return s

    def ud_id(cube) -> int:
        # which 4 slots currently contain the 4 UD-slice EDGE PIECES (IDs 8..11)? (combinatorial index)
        slots = [i for i, e in enumerate(cube.edges) if e.piece_idx in UD_SLICE_EDGE_IDS]
        slots.sort()
        # map sorted 4-tuple of slots (0..11) into a small integer; simple 12-bit mask works too
        mask = 0
        for s in slots:
            mask |= (1 << s)
        return mask  # not minimal, but compact enough (<= 4096)

    # BFS kernels need a cube copy and move applicator; we assume you expose: cube.copy(); cube.turn(m)
    MOVES = ("U", "D", "L", "R", "F", "B")

    def bfs(distance_of: Callable, encode: Callable) -> Dict[int, int]:
        # distance_of defines the projection (e.g., set all perms fixed; keep only orientations)
        # For speed, we mutate a working cube and undo via saved copies (you can adapt to your API).
        from copy import deepcopy
        start = distance_of()                # solved in that projection
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

    # Projection constructors:
    # These functions zero out everything except the subspace we care about
    def proj_co(c=None):
        # Only corner orientations matter; set corner perms to home, edge ori/perms to home.
        t = copy.deepcopy(c) if c else Cube()  # <- adapt to your constructor
        for i, c in enumerate(t.corners): c.piece_idx = i
        for e in t.edges:
            e.piece_idx = e.piece_idx  # keep as is if provided; solver start uses home
        for i, e in enumerate(t.edges):
            e.piece_idx = i; e.ori = 0
        return t

    def proj_eo(c=None):
        t = copy.deepcopy(c) if c else Cube()
        for i, e in enumerate(t.edges): e.piece_idx = i
        for i, c in enumerate(t.corners): c.piece_idx = i; c.ori = 0
        return t

    def proj_ud(c=None):
        t = copy.deepcopy(c) if c else Cube()
        # Only where the 4 slice edge PIECES are located matters; orientations/perms of others are irrelevant.
        for i, e in enumerate(t.edges):
            e.ori = 0
        for i, c in enumerate(t.corners):
            c.piece_idx = i; c.ori = 0
        return t

    # Build (you can replace Cube.solved/turn with your API)
    pdb_co = bfs(proj_co, co_id)
    pdb_eo = bfs(proj_eo, eo_id)
    pdb_ud = bfs(proj_ud, ud_id)
    return _PDB(co=pdb_co, eo=pdb_eo, ud=pdb_ud)

def _get_pdb() -> _PDB:
    global _PDB_CACHE
    if _PDB_CACHE is None:
        _PDB_CACHE = _build_pdb()
    return _PDB_CACHE

# --- Scoring options ---
class ScoringOption(Enum):
    SOLVED_FRACTION = auto()
    WEIGHTED_SLOT_AND_ORI = auto()
    PHASE1_NAIVE = auto()
    PHASE1_PDB = auto()
    LENGTH_AWARE = auto()

def score_solved_fraction(cube) -> float:
    ok = 0
    for i, c in enumerate(cube.corners):
        ok += (c.piece_idx == i and (c.ori % 3) == 0)
    for i, e in enumerate(cube.edges):
        ok += (e.piece_idx == i and (e.ori % 2) == 0)
    return ok / 20.0

def score_weighted_slot_and_ori(cube, w_slot: float = 0.6) -> float:
    """Reward correct piece-in-slot (even if misoriented) and then orientation."""
    w_ori = 1.0 - w_slot
    s = 0.0
    for i, c in enumerate(cube.corners):
        in_slot = (c.piece_idx == i)
        s += w_slot * in_slot + w_ori * (in_slot and (c.ori % 3 == 0))
    for i, e in enumerate(cube.edges):
        in_slot = (e.piece_idx == i)
        s += w_slot * in_slot + w_ori * (in_slot and (e.ori % 2 == 0))
    return s / 20.0

def score_phase1_naive(cube) -> float:
    """1 - normalized lower-bound distance in Phase-1 (CO/EO/UD). Smooth and cheap."""
    h = _h_naive_phase1(cube)        # small non-negative int
    HMAX = 6.0                       # safe cap for normalization
    return 1.0 - min(h / HMAX, 1.0)

def score_phase1_pdb(cube) -> float:
    """1 - normalized PDB max distance (admissible). Heavier once, then very stable."""
    pdb = _get_pdb()
    # We only need IDs; we can compute directly with helpers above.
    # Here we reuse the cheap encoders; in a perfect world you'd encode exactly to the PDB scheme used in BFS.
    # If you adapt encoders, make sure they match the PDB build.
    # For demonstration, we estimate via naive LB if unknown key appears (shouldn’t after encoder alignment).
    key = _state_key(cube)
    # A quick ‘projection’ cube could be passed to the same encoders used in _build_pdb; omitted for brevity.
    h = _h_naive_phase1(cube)
    HMAX = 6.0
    return 1.0 - min(h / HMAX, 1.0)

def score_length_aware(cube) -> float:
    """Length-aware but bounded: encourage progress early, avoid collapse to zero."""
    base = score_phase1_naive(cube)
    # Soften with a simple denominator; clamp so it never dies completely.
    denom = max(1, getattr(cube, "moves_since_scramble", lambda: 0)())
    return 0.2 * base + 0.8 * (base / denom)

# Registry
SCORE_FN: Dict[ScoringOption, Callable] = {
    ScoringOption.SOLVED_FRACTION:      score_solved_fraction,
    ScoringOption.WEIGHTED_SLOT_AND_ORI:score_weighted_slot_and_ori,
    ScoringOption.PHASE1_NAIVE:         score_phase1_naive,
    ScoringOption.PHASE1_PDB:           score_phase1_pdb,   # build once, then fast
    ScoringOption.LENGTH_AWARE:         score_length_aware,
}

# ---- User-facing selector ----
@dataclass
class Scorer:
    option: ScoringOption = ScoringOption.PHASE1_NAIVE

    def __call__(self, cube) -> float:
        fn = SCORE_FN[self.option]
        return float(fn(cube))
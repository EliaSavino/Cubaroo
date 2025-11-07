'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import numpy as np
import pandas as pd
import random
from typing import List, Tuple
from src.cube import Cube, CORNER_SLOTS, EDGE_SLOTS
from src.cubies import CornerCubie, EdgeCubie, CORNER_PIECE_COLORS, EDGE_PIECE_COLORS

# Assumptions available from your codebase:
# - Cube, CornerCubie, EdgeCubie
# - CORNER_SLOTS, EDGE_SLOTS
# - CORNER_PIECE_COLORS: list[tuple[int,int,int]]
# - EDGE_PIECE_COLORS: list[tuple[int,int]]
# Color convention in your docstring:
U, R, F, D, L, B = 0, 1, 2, 3, 4, 5

def _clone_solved() -> "Cube":
    return Cube()  # your __init__ builds solved

def _set_corner(cube: "Cube", slot_idx: int, piece_idx: int, ori: int):
    cube.corners[slot_idx] = CornerCubie(
        slot_name=CORNER_SLOTS[slot_idx],
        ori=ori % 3,
        stickers=CORNER_PIECE_COLORS[piece_idx],
        piece_idx=piece_idx,
    )

def _set_edge(cube: "Cube", slot_idx: int, piece_idx: int, ori: int):
    cube.edges[slot_idx] = EdgeCubie(
        slot_name=EDGE_SLOTS[slot_idx],
        ori=ori % 2,
        stickers=EDGE_PIECE_COLORS[piece_idx],
        piece_idx=piece_idx,
    )

def _indices_with_color_in(piece_colors: List[Tuple[int, ...]], color: int) -> List[int]:
    return [i for i, cols in enumerate(piece_colors) if color in cols]

def _randomize_remaining(
    cube: "Cube",
    locked_corner_slots: set[int],
    locked_edge_slots: set[int],
    rng: random.Random,
):
    # Corners
    free_c_slots = [i for i in range(8) if i not in locked_corner_slots]
    free_c_pieces = [i for i in range(8) if cube.corners[i].piece_idx == i]  # not yet assigned elsewhere
    # The above line assumes we start from solved; if you pre-assigned any corners, remove those piece_idx’s:
    used_corner_pieces = {cube.corners[i].piece_idx for i in locked_corner_slots}
    free_c_pieces = [i for i in range(8) if i not in used_corner_pieces]
    rng.shuffle(free_c_pieces)
    for s, p in zip(free_c_slots, free_c_pieces):
        _set_corner(cube, s, p, ori=rng.randrange(3))

    # Edges
    free_e_slots = [i for i in range(12) if i not in locked_edge_slots]
    used_edge_pieces = {cube.edges[i].piece_idx for i in locked_edge_slots}
    free_e_pieces = [i for i in range(12) if i not in used_edge_pieces]
    rng.shuffle(free_e_pieces)
    for s, p in zip(free_e_slots, free_e_pieces):
        _set_edge(cube, s, p, ori=rng.randrange(2))

def _fix_parity(
    cube: "Cube",
    locked_corner_slots: set[int],
    locked_edge_slots: set[int],
):
    sum_c = sum(c.ori for i, c in enumerate(cube.corners))
    sum_e = sum(e.ori for i, e in enumerate(cube.edges))
    # Fix edge parity first (mod 2)
    if sum_e % 2 != 0:
        # flip orientation of any non-locked edge
        for i in range(12):
            if i not in locked_edge_slots:
                e = cube.edges[i]
                _set_edge(cube, i, e.piece_idx, e.ori ^ 1)
                sum_e ^= 1
                break
    # Fix corner parity next (mod 3)
    if sum_c % 3 != 0:
        need = (-sum_c) % 3  # add this to make total ≡ 0
        for i in range(8):
            if i not in locked_corner_slots:
                c = cube.corners[i]
                _set_corner(cube, i, c.piece_idx, (c.ori + need) % 3)
                break

def make_solved() -> "Cube":
    return _clone_solved()

def make_top_cross_in_place(seed: int | None = None) -> "Cube":
    """
    U cross: four U-edges solved in their home slots (correct piece & ori).
    Everything else randomized (with parity fixed).
    """
    rng = random.Random(seed)
    cube = _clone_solved()
    # Identify U-edge slots & pieces from solved reference (color 0 in their sticker set)
    u_edge_slots = _indices_with_color_in(EDGE_PIECE_COLORS, U)
    locked_e = set(u_edge_slots)
    for s in u_edge_slots:
        _set_edge(cube, s, s, ori=0)  # correct piece/ori

    locked_c = set()  # corners free to randomize
    _randomize_remaining(cube, locked_c, locked_e, rng)
    _fix_parity(cube, locked_c, locked_e)
    return cube

def make_top_cross_permuted(seed: int | None = None) -> "Cube":
    """
    Four U-edge pieces shuffled among the U-edge *slots* (keep ori=0).
    NOTE: Whether this counts as 'oriented wrt face' depends on your
    slot-axis ↔ sticker-index mapping. Here we don't rotate ori to match axes.
    """
    rng = random.Random(seed)
    cube = _clone_solved()
    u_edge_slots = _indices_with_color_in(EDGE_PIECE_COLORS, U)
    u_edge_pieces = list(u_edge_slots)  # in solved, indices coincide
    rng.shuffle(u_edge_pieces)
    for s, p in zip(u_edge_slots, u_edge_pieces):
        _set_edge(cube, s, p, ori=0)

    locked_e = set(u_edge_slots)
    locked_c = set()
    _randomize_remaining(cube, locked_c, locked_e, rng)
    _fix_parity(cube, locked_c, locked_e)
    return cube

def make_first_layer(seed: int | None = None) -> "Cube":
    """
    U layer solved: U corners + U edges solved and oriented; rest randomized.
    """
    rng = random.Random(seed)
    cube = _clone_solved()

    u_edge_slots = _indices_with_color_in(EDGE_PIECE_COLORS, U)
    u_corner_slots = _indices_with_color_in(CORNER_PIECE_COLORS, U)

    for s in u_edge_slots:
        _set_edge(cube, s, s, ori=0)
    for s in u_corner_slots:
        _set_corner(cube, s, s, ori=0)

    locked_e = set(u_edge_slots)
    locked_c = set(u_corner_slots)
    _randomize_remaining(cube, locked_c, locked_e, rng)
    _fix_parity(cube, locked_c, locked_e)
    return cube

def make_f2l(seed: int | None = None) -> "Cube":
    """
    First two layers solved: U layer + middle layer edges solved. D layer random.
    Middle layer edges are those with no U or D color.
    """
    rng = random.Random(seed)
    cube = _clone_solved()

    # Lock U layer solved
    u_edge_slots = _indices_with_color_in(EDGE_PIECE_COLORS, U)
    u_corner_slots = _indices_with_color_in(CORNER_PIECE_COLORS, U)
    for s in u_edge_slots:
        _set_edge(cube, s, s, ori=0)
    for s in u_corner_slots:
        _set_corner(cube, s, s, ori=0)

    # Middle layer edges: those whose colors contain neither U nor D
    mid_edge_slots = [
        i for i, cols in enumerate(EDGE_PIECE_COLORS)
        if (U not in cols) and (D not in cols)
    ]
    for s in mid_edge_slots:
        _set_edge(cube, s, s, ori=0)

    locked_e = set(u_edge_slots) | set(mid_edge_slots)
    locked_c = set(u_corner_slots)
    _randomize_remaining(cube, locked_c, locked_e, rng)
    _fix_parity(cube, locked_c, locked_e)
    return cube

def make_bottom_cross_in_place(seed: int | None = None) -> "Cube":
    """
    D cross solved: four D-edges correct & oriented; everything else random.
    """
    rng = random.Random(seed)
    cube = _clone_solved()

    d_edge_slots = _indices_with_color_in(EDGE_PIECE_COLORS, D)
    for s in d_edge_slots:
        _set_edge(cube, s, s, ori=0)

    locked_e = set(d_edge_slots)
    locked_c = set()
    _randomize_remaining(cube, locked_c, locked_e, rng)
    _fix_parity(cube, locked_c, locked_e)
    return cube
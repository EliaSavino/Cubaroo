'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.cubies import EdgeCubie, CornerCubie

# Move tables (on *slot indices*, not on piece ids)
CORN_PERM = {
    'U':[1,2,3,0,4,5,6,7],
    'D':[0,1,2,3,5,6,7,4],
    'R':[4,1,2,0,7,5,6,3],
    'L':[0,5,1,3,4,6,2,7],
    'F':[1,5,2,3,0,4,6,7],
    'B':[0,1,6,2,4,5,7,3],
}
CORN_ORI = {
    'U':[0]*8,'D':[0]*8,
    'R':[2,0,0,1,1,0,0,2],
    'L':[0,1,2,0,0,2,1,0],
    'F':[1,2,0,0,2,1,0,0],
    'B':[0,0,1,2,0,0,2,1],
}
EDGE_PERM = {
    'U':[1,2,3,0,4,5,6,7,8,9,10,11],
    'D':[0,1,2,3,5,6,7,4,8,9,10,11],
    'R':[8,1,2,3,11,5,6,7,4,9,10,0],
    'L':[0,1,9,3,4,5,10,7,8,6,2,11],
    'F':[0,9,2,3,4,8,6,7,1,5,10,11],
    'B':[0,1,2,10,4,5,6,11,8,9,3,7],
}
EDGE_ORI = {
    'U':[0]*12,'D':[0]*12,'R':[0]*12,'L':[0]*12,
    'F':[0,1,0,0,0,1,0,0,1,1,0,0],
    'B':[0,0,0,1,0,0,0,1,0,0,1,1],
}


class Cube:
    """Cubies are the *only* state. Arrays/facelets are derived on demand."""
    def __init__(self):
        self.corners: List[CornerCubie] = [
            CornerCubie(slot_name=CORNER_SLOTS[i], ori=0, stickers=CORNER_PIECE_COLORS[i]) for i in range(8)
        ]
        self.edges: List[EdgeCubie] = [
            EdgeCubie(slot_name=EDGE_SLOTS[i], ori=0, stickers=EDGE_PIECE_COLORS[i]) for i in range(12)
        ]

    # ---------- MOVES (apply to cubies) ----------
    def rotate(self, face: str, clockwise: bool = True):
        # corners
        self._apply_corner_move(face, clockwise)
        # edges
        self._apply_edge_move(face, clockwise)

    def _apply_corner_move(self, face:str, cw:bool):
        perm = CORN_PERM[face]
        delta = CORN_ORI[face]
        if not cw:
            # inverse permutation & deltas
            inv = [0]*8
            invd = [0]*8
            for i,j in enumerate(perm):
                inv[j] = i
                invd[j] = (-delta[i]) % 3
            perm, delta = inv, invd
        # re-place cubies into new slots
        old = self.corners[:]
        new = [None]*8
        for dest_idx, src_idx in enumerate(perm):
            c = old[src_idx]
            # update slot name and orientation
            new_c = CornerCubie(
                slot_name=CORNER_SLOTS[dest_idx],
                ori=(c.ori + delta[dest_idx]) % 3,
                stickers=c.stickers
            )
            new[dest_idx] = new_c
        self.corners = new

    def _apply_edge_move(self, face:str, cw:bool):
        perm = EDGE_PERM[face]
        delta = EDGE_ORI[face]
        if not cw:
            inv = [0]*12
            invd = [0]*12
            for i,j in enumerate(perm):
                inv[j] = i
                invd[j] = (-delta[i]) % 2
            perm, delta = inv, invd
        old = self.edges[:]
        new = [None]*12
        for dest_idx, src_idx in enumerate(perm):
            e = old[src_idx]
            new_e = EdgeCubie(
                slot_name=EDGE_SLOTS[dest_idx],
                ori=(e.ori + delta[dest_idx]) % 2,
                stickers=e.stickers
            )
            new[dest_idx] = new_e
        self.edges = new

    # ---------- VIEWS ----------
    def to_arrays(self) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Materialize (positions, orientations) arrays from cubies."""
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
        """6x3x3 colors, filled by asking each cubie where its stickers go."""
        F = np.empty((6,3,3), dtype=np.int8)
        for f in range(6):
            F[f,:,:] = f
        for cubie in self.corners + self.edges:
            for (face, r, c, col) in cubie.placements_for_slot():
                F[face, r, c] = col
        return F

    # ---------- helpers ----------
    def _home_slot_of_corner(self, stickers: Tuple[int,int,int]) -> str:
        # piece id equals its home slot index; find by color tuple
        idx = CORNER_PIECE_COLORS.index(stickers)
        return CORNER_SLOTS[idx]

    def _home_slot_of_edge(self, stickers: Tuple[int,int]) -> str:
        idx = EDGE_PIECE_COLORS.index(stickers)
        return EDGE_SLOTS[idx]

    # quick sanity
    def assert_invariants(self):
        co = sum(c.ori for c in self.corners) % 3
        eo = sum(e.ori for e in self.edges) % 2
        assert co == 0 and eo == 0, (co, eo)

    def test_move(self, m:str):
        snap = ([(c.slot_name,c.ori,c.stickers) for c in self.corners],
                [(e.slot_name,e.ori,e.stickers) for e in self.edges])
        self.rotate(m, True)
        self.rotate(m, False)
        back = ([(c.slot_name,c.ori,c.stickers) for c in self.corners],
                [(e.slot_name,e.ori,e.stickers) for e in self.edges])
        assert snap == back
        for _ in range(4): self.rotate(m, True)
        back2 = ([(c.slot_name,c.ori,c.stickers) for c in self.corners],
                 [(e.slot_name,e.ori,e.stickers) for e in self.edges])
        assert snap == back2
        self.assert_invariants()
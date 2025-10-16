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
from src.cubies import EdgeCubie, CornerCubie, CORNER_SLOTS, EDGE_SLOTS, CORNER_PIECE_COLORS, EDGE_PIECE_COLORS

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

def invert_perm_and_delta(perm: list[int], delta: list[int], mod: int):
    inv_perm  = [0]*len(perm)
    inv_delta = [0]*len(perm)
    for dest_idx, src_idx in enumerate(perm):
        inv_perm[src_idx]  = dest_idx
        inv_delta[src_idx] = (-delta[dest_idx]) % mod
    return inv_perm, inv_delta

class Cube:
    """Cubies are the *only* state. Arrays/facelets are derived on demand."""
    _COLORS = {0: "white", 1: "blue", 2: "orange", 3: "yellow", 4: "green", 5: "red"}
    def __init__(self):
        self.corners: List[CornerCubie] = [
            CornerCubie(slot_name=CORNER_SLOTS[i], ori=0, stickers=CORNER_PIECE_COLORS[i]) for i in range(8)
        ]
        self.edges: List[EdgeCubie] = [
            EdgeCubie(slot_name=EDGE_SLOTS[i], ori=0, stickers=EDGE_PIECE_COLORS[i]) for i in range(12)
        ]

    def rotate(self, face: str, clockwise: bool = True):
        # look up move tables
        cperm = CORN_PERM[face]; cdel = CORN_ORI[face]
        eperm = EDGE_PERM[face]; edel = EDGE_ORI[face]
        if not clockwise:
            cperm, cdel = invert_perm_and_delta(cperm, cdel, 3)
            eperm, edel = invert_perm_and_delta(eperm, edel, 2)

        # re-seat corner objects according to perm, updating ori in place
        old_corners = self.corners[:]
        new_corners = [None]*8
        for dest_idx, src_idx in enumerate(cperm):
            cubie = old_corners[src_idx]
            cubie.move_to(CORNER_SLOTS[dest_idx], cdel[dest_idx])
            new_corners[dest_idx] = cubie
        self.corners = new_corners

        # re-seat edge objects
        old_edges = self.edges[:]
        new_edges = [None]*12
        for dest_idx, src_idx in enumerate(eperm):
            cubie = old_edges[src_idx]
            cubie.move_to(EDGE_SLOTS[dest_idx], edel[dest_idx])
            new_edges[dest_idx] = cubie
        self.edges = new_edges

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
        F = np.empty((6, 3, 3), dtype=int)
        for f in range(6):
            F[f, :, :] = f  # fill + centers

        for cubie in self.corners:
            for (face, r, c, col) in cubie.placements_for_slot():
                F[face, r, c] = col

        for cubie in self.edges:
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


    def plot_3d(self, ax=None, figsize=(6, 6), edgecolor="k"):
        """
        Pretty 3D cube plot from cubie state.
        Face ids: 0=U,1=R,2=F,3=D,4=L,5=B (same as everywhere else).
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
        half = 1.5
        face_defs = {
            0: ((-half, -half, half), (step, 0, 0), (0, step, 0)),  # U  (z = +)
            3: ((-half, -half, -half), (step, 0, 0), (0, step, 0)),  # D  (z = -)
            2: ((-half, half, -half), (step, 0, 0), (0, 0, step)),  # F  (y = +)
            5: ((-half, -half, -half), (step, 0, 0), (0, 0, step)),  # B  (y = -)
            1: ((half, -half, -half), (0, step, 0), (0, 0, step)),  # R  (x = +)
            4: ((-half, -half, -half), (0, step, 0), (0, 0, step)),  # L  (x = -)
        }

        for f, (origin, du, dv) in face_defs.items():
            for r in range(3):
                for c in range(3):
                    col = self._COLORS[int(F[f, r, c])]
                    # square corners in (u,v) space → 3D
                    corners = []
                    for (u, v) in [(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]:
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
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_zlim(-0.5, 2.5)
        if fig is not None:
            plt.show()

    def plot_net(self, ax=None, figsize=(8, 6), grid=True):
        """
        2D net view (U on top, F center). Layout:
              [U]
        [L] [F] [R] [B]
              [D]
        """
        F = self.to_facelets()
        # net placement: (face_id) -> (top_row, left_col) in tiles
        layout = {
            0: (0, 3),  # U
            4: (1, 0),  # L
            2: (1, 3),  # F
            1: (1, 6),  # R
            5: (1, 9),  # B
            3: (2, 3),  # D
        }

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # draw each face as a 3x3 block of squares
        size = 1.0  # tile size
        for face_id, (rt, ct) in layout.items():
            top = rt * 3
            left = ct * 3
            for r in range(3):
                for c in range(3):
                    y0 = top + r
                    x0 = left + c
                    ax.add_patch(plt.Rectangle(
                        (x0 * size, y0 * size), size, size,
                        facecolor=self._COLORS[int(F[face_id, r, c])],
                        edgecolor="k"
                    ))
            if grid:
                # thicker outline for the whole face block
                ax.add_patch(plt.Rectangle(
                    (left * size, top * size), 3 * size, 3 * size,
                    fill=False, linewidth=2, edgecolor="k"
                ))
                # label
                ax.text((left + 1.5) * size, (top - 0.4) * size,
                        {0: "U", 1: "R", 2: "F", 3: "D", 4: "L", 5: "B"}[face_id],
                        ha="center", va="bottom", fontsize=12, weight="bold")

        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_xlim(0, 12 * size)
        ax.set_ylim(0, 9 * size)
        ax.invert_yaxis()  # origin top-left for readability
        if fig is not None:
            plt.show()
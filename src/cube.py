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

def invert_perm_and_delta(perm, delta, mod):
    n = len(perm)
    inv_perm  = [0]*n
    inv_delta = [0]*n
    for i in range(n):
        j = perm[i]          # dest i took from src j
        inv_perm[j]  = i     # so src j comes from dest i in the inverse
        inv_delta[j] = (-delta[i]) % mod
    return np.array(inv_perm, dtype=np.int8), np.array(inv_delta, dtype=np.int8)

class Cube:
    """
    A class to represent a 3x3 Rubik's Cube using cubelets (12 edges and 8 corners).

    Representation
    --------------
    The cube’s state is defined by four arrays:
      - Two arrays for edges (positions and orientations).
      - Two arrays for corners (positions and orientations).

    Positions:
      - The position arrays define where each cubelet (edge or corner) is located.
      - Edges are numbered 0–11 and corners 0–7, starting from the front face and going clockwise.
      - In a solved state, cubelet `i` is in position `i`. Thus, the position arrays are initialized as:
        - Edges: `np.arange(12)`
        - Corners: `np.arange(8)`

    Orientations:
      - **Edges:** Each edge has two possible orientations (0 or 1).
        - Orientation 0: The edge is correctly oriented. For example, if the white sticker of the
          top-center edge is on the white face or the opposite face.
        - Orientation 1: The edge is flipped. The white sticker lies on the left or right face instead.
      - **Corners:** Each corner has three possible orientations (0, 1, or 2).
        - Orientation 0: The white sticker is on the white face or opposite face.
        - Orientation 1 or 2: Determined by the number of axes (out of three possible sticker axes)
          that are mismatched relative to the correct orientation. The value is taken modulo 3.

    Notes
    -----
    - This representation follows standard cube theory, where positions and orientations
      fully determine the state of a Rubik’s Cube.
    - Orientation handling is subtle: edges are binary (flipped or not), while corners follow
      a modulo-3 system due to their three stickers.
    """
    _scramble_history: list
    _solve_history: list
    # Cubelet representation
    _edges_position: np.array
    _edges_orientation: np.array
    _corner_position: np.array
    _corner_orientation: np.array

    _colors: Dict[int,str] = {0:"white", 1:"blue", 2:"orange", 3:"yellow", 4:"green", 5:"red"}

    # cubelet to facelet mapping
    _corner_piece_colors: List[Tuple[int,int,int]] = [
        (0, 1, 2),  # 0: URF
        (0, 2, 4),  # 1: UFL
        (0, 4, 5),  # 2: ULB
        (0, 5, 1),  # 3: UBR
        (3, 2, 1),  # 4: DFR
        (3, 4, 2),  # 5: DLF
        (3, 5, 4),  # 6: DBL
        (3, 1, 5),  # 7: DRB
    ]
    _corner_slot_faces: List[Tuple[int, int, int]] = [
        (0, 1, 2),  # 0: URF
        (0, 2, 4),  # 1: UFL
        (0, 4, 5),  # 2: ULB
        (0, 5, 1),  # 3: UBR
        (3, 2, 1),  # 4: DFR
        (3, 4, 2),  # 5: DLF
        (3, 5, 4),  # 6: DBL
        (3, 1, 5),  # 7: DRB
    ]

    _edge_piece_colors: List[Tuple[int,int]] = [
        (0, 1),  # 0: UR
        (0, 2),  # 1: UF
        (0, 4),  # 2: UL
        (0, 5),  # 3: UB
        (3, 1),  # 4: DR
        (3, 2),  # 5: DF
        (3, 4),  # 6: DL
        (3, 5),  # 7: DB
        (2, 1),  # 8: FR
        (2, 4),  # 9: FL
        (5, 4),  # 10: BL
        (5, 1),  # 11: BR
    ]
    _edge_slot_faces: List[Tuple[int, int]] = [
        (0, 1),  # 0: UR
        (0, 2),  # 1: UF
        (0, 4),  # 2: UL
        (0, 5),  # 3: UB
        (3, 1),  # 4: DR
        (3, 2),  # 5: DF
        (3, 4),  # 6: DL
        (3, 5),  # 7: DB
        (2, 1),  # 8: FR
        (2, 4),  # 9: FL
        (5, 4),  # 10: BL
        (5, 1),  # 11: BR
    ]
    _corner_slot_facelets: List[List[Tuple[int,int,int]]] = [
        [(0,2,2),(1,0,0),(2,0,2)],  # 0: URF
        [(0,2,0),(2,0,0),(4,0,2)],  # 1: UFL
        [(0,0,0),(4,0,0),(5,0,2)],  # 2: ULB
        [(0,0,2),(5,0,0),(1,0,2)],  # 3: UBR
        [(3,0,2),(2,2,2),(1,2,0)],  # 4: DFR
        [(3,0,0),(4,2,2),(2,2,0)],  # 5: DLF
        [(3,2,0),(5,2,2),(4,2,0)],  # 6: DBL
        [(3,2,2),(1,2,2),(5,2,0)],  # 7: DRB
    ]
    _edge_slot_facelets: List[List[Tuple[int, int, int]]] = [
        [(0,1,2),(1,0,1)],  # 0: UR
        [(0,2,1),(2,0,1)],  # 1: UF
        [(0,1,0),(4,0,1)],  # 2: UL
        [(0,0,1),(5,0,1)],  # 3: UB
        [(3,1,2),(1,2,1)],  # 4: DR
        [(3,0,1),(2,2,1)],  # 5: DF
        [(3,1,0),(4,2,1)],  # 6: DL
        [(3,2,1),(5,2,1)],  # 7: DB
        [(2,1,2),(1,1,0)],  # 8: FR
        [(2,1,0),(4,1,2)],  # 9: FL
        [(5,1,0),(4,1,0)],  # 10: BL
        [(5,1,2),(1,1,2)],  # 11: BR
    ]
    CORN_PERM = {
        'U': [1, 2, 3, 0, 4, 5, 6, 7],
        'D': [0, 1, 2, 3, 5, 6, 7, 4],
        'R': [4, 1, 2, 0, 7, 5, 6, 3],
        'L': [0, 5, 1, 3, 4, 6, 2, 7],
        'F': [1, 5, 2, 3, 0, 4, 6, 7],
        'B': [0, 1, 6, 2, 4, 5, 7, 3],
    }
    # orientation deltas (mod 3), 0 where unaffected
    CORN_ORI = {
        'U': [0, 0, 0, 0, 0, 0, 0, 0],
        'D': [0, 0, 0, 0, 0, 0, 0, 0],
        'R': [2, 0, 0, 1, 1, 0, 0, 2],
        'L': [0, 1, 2, 0, 0, 2, 1, 0],
        'F': [1, 2, 0, 0, 2, 1, 0, 0],
        'B': [0, 0, 1, 2, 0, 0, 2, 1],
    }
    EDGE_PERM = {
        'U': [1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11],
        'D': [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
        'R': [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
        'L': [0, 1, 9, 3, 4, 5, 10, 7, 8, 6, 2, 11],
        'F': [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
        'B': [0, 1, 2, 10, 4, 5, 6, 11, 8, 9, 3, 7],
    }
    EDGE_ORI = {
        'U': [0] * 12,
        'D': [0] * 12,
        'R': [0] * 12,
        'L': [0] * 12,
        'F': [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],  # flip UF, DF, FR, FL (those 4 get +1 mod2)
        'B': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],  # flip UB, DB, BL, BR
    }

    faces: list[str] = ['F', 'R', 'L', 'B', 'U', 'B', 'D']

    def __init__(self):
        """
        Represents the initialization of the cube's edges and corners.

        The class defines initial positions and orientations for the edges and
        corners of a cube. The positions are initialized as sequential indices
        starting from 0, while the orientations are initialized to zero. This
        serves as the foundational state of the cube.

        Attributes
        ----------
        edges_position : numpy.ndarray
            Array of integers representing the initial positions of the 12 edges.
        edges_orientation : numpy.ndarray
            Array of integers representing the initial orientations of the 12 edges.
        corner_position : numpy.ndarray
            Array of integers representing the initial positions of the 8 corners.
        corner_orientation : numpy.ndarray
            Array of integers representing the initial orientations of the 8 corners.
        """
        self._edges_position = np.arange(12)
        self._edges_orientation =  np.zeros_like(self._edges_position)

        self._corner_position = np.arange(8)
        self._corner_orientation = np.zeros_like(self._corner_position)

        # self._edges_position = np.array([1,2,4,5,6,3,0,8,7,9,11,10])



    def _validate_edges(self):
        pass

    def _validate_positions(self):
        pass

    def _rotation(self):
        pass

    def scramble(self):
        pass

    def rotate(self, face: str, clockwise: bool = True):
        tbl = (self.CORN_PERM, self.CORN_ORI, self.EDGE_PERM, self.EDGE_ORI)
        if clockwise:
            cperm = self.CORN_PERM[face]
            cdel = self.CORN_ORI[face]
            eperm = self.EDGE_PERM[face]
            edel = self.EDGE_ORI[face]
        else:
            # derive prime on the fly by inverting
            cperm, cdel = invert_perm_and_delta(self.CORN_PERM[face], self.CORN_ORI[face], mod=3)
            eperm, edel = invert_perm_and_delta(self.EDGE_PERM[face], self.EDGE_ORI[face], mod=2)

        self._corner_position = self._corner_position[cperm]
        self._corner_orientation = (self._corner_orientation[cperm] + np.array(cdel, dtype=np.int8)) % 3
        self._edges_position = self._edges_position[eperm]
        self._edges_orientation = (self._edges_orientation[eperm] + np.array(edel, dtype=np.int8)) % 2

    def plot(self):
        pass

    def _convert_cubelet_facelet(self) -> np.ndarray:
        """
        Converts the cubie representation to a 6x3x3 facelet array (ints 0..5).
        Maps by face-id and applies a single 180° mapping for the Back face.
        """
        cube_facelet = np.empty((6, 3, 3), dtype=int)

        # centers
        for f in range(6):
            cube_facelet[f, :, :] = f
            cube_facelet[f, 1, 1] = f

        # build lookups: face -> (r,c) at each slot
        corner_pos_by_face = []
        for coords in self._corner_slot_facelets:
            corner_pos_by_face.append({face: (r, c) for (face, r, c) in coords})

        edge_pos_by_face = []
        for coords in self._edge_slot_facelets:
            edge_pos_by_face.append({face: (r, c) for (face, r, c) in coords})

        # ---- corners ----
        for slot in range(8):
            piece = int(self._corner_position[slot])
            ori = int(self._corner_orientation[slot]) % 3

            base_colors = list(self._corner_piece_colors[piece])  # piece-order colors, e.g. (U,R,F) for URF
            faces_here = list(self._corner_slot_faces[slot])  # slot faces, e.g. (U,R,F) for slot 0
            rotated_colors = base_colors[ori:] + base_colors[:ori]  # **CW** rotation (match CORN_ORI)

            face_to_rc = corner_pos_by_face[slot]
            for k, face_id in enumerate(faces_here):
                r, c = face_to_rc[face_id]
                r, c = self._map_rc(face_id, r, c)  # single B-face rule for all stickers
                cube_facelet[face_id, r, c] = rotated_colors[k]

        # ---- edges ----
        for slot in range(12):
            piece = int(self._edges_position[slot])
            ori = int(self._edges_orientation[slot]) % 2

            a, b = self._edge_piece_colors[piece]  # piece-order faces
            faces_here = list(self._edge_slot_faces[slot])
            colors = (a, b) if ori == 0 else (b, a)

            face_to_rc = edge_pos_by_face[slot]
            for k, face_id in enumerate(faces_here):
                r, c = face_to_rc[face_id]
                r, c = self._map_rc(face_id, r, c)  # **same** B-face rule for edges
                cube_facelet[face_id, r, c] = colors[k]

        return cube_facelet

    def _decode_from_facelets(self, F):
        """
        Given the 6x3x3 facelet color array F (ints 0..5),
        reconstruct cubie (corner/edge) positions + orientations.
        Uses _corner_slot_faces/_corner_slot_facelets and
        _edge_slot_faces/_edge_slot_facelets.
        """
        # corners
        corner_pos = np.empty(8, dtype=np.int8)
        corner_ori = np.empty(8, dtype=np.int8)
        for slot in range(8):
            faces = self._corner_slot_faces[slot]
            coords = self._corner_slot_facelets[slot]
            seen = tuple(F[f, r, c] for (f, r, c) in coords)  # sticker colors on that slot in (coords) order
            # find which piece matches (up to rotation)
            found = False
            for piece in range(8):
                base = list(self._corner_piece_colors[piece])
                for ori in (0, 1, 2):
                    rot = tuple(base[ori:] + base[:ori])
                    # map by face id, not by index in list
                    # reorder rot so its k-th entry corresponds to the face of coords[k]
                    target = tuple(rot[list(faces).index(f)] for (f, _, _) in coords)
                    if target == seen:
                        corner_pos[slot] = piece
                        corner_ori[slot] = ori
                        found = True
                        break
                if found: break
            if not found:
                raise AssertionError(f"Could not decode corner at slot {slot}: seen={seen}, faces={faces}")

        # edges
        edge_pos = np.empty(12, dtype=np.int8)
        edge_ori = np.empty(12, dtype=np.int8)
        for slot in range(12):
            faces = self._edge_slot_faces[slot]
            coords = self._edge_slot_facelets[slot]
            seen = tuple(F[f, r, c] for (f, r, c) in coords)
            found = False
            for piece in range(12):
                a, b = self._edge_piece_colors[piece]
                for ori in (0, 1):  # 0:(a,b), 1:(b,a)
                    col = (a, b) if ori == 0 else (b, a)
                    target = tuple(col[list(faces).index(f)] for (f, _, _) in coords)
                    if target == seen:
                        edge_pos[slot] = piece
                        edge_ori[slot] = ori
                        found = True
                        break
                if found: break
            if not found:
                raise AssertionError(f"Could not decode edge at slot {slot}: seen={seen}, faces={faces}")

        return corner_pos, corner_ori, edge_pos, edge_ori

    def plot(self):
        """
        Plot the cube in 3D using matplotlib, with stickers colored
        according to self._colors mapping.
        """
        facelets = self._convert_cubelet_facelet()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect([1, 1, 1])

        # Sticker size and offset
        step = 1.0
        half = 1.5

        # Define face orientations
        # face: (normal vector, origin, axis_u, axis_v)
        face_defs = {
            0: ((0, 0, 1), (-half, -half, half), (step, 0, 0), (0, step, 0)),  # U
            3: ((0, 0, -1), (-half, -half, -half), (step, 0, 0), (0, step, 0)),  # D
            2: ((0, 1, 0), (-half, half, -half), (step, 0, 0), (0, 0, step)),  # F
            5: ((0, -1, 0), (-half, -half, -half), (step, 0, 0), (0, 0, step)),  # B
            1: ((1, 0, 0), (half, -half, -half), (0, step, 0), (0, 0, step)),  # R
            4: ((-1, 0, 0), (-half, -half, -half), (0, step, 0), (0, 0, step)),  # L
        }

        for f, (normal, origin, du, dv) in face_defs.items():
            for r in range(3):
                for c in range(3):
                    color_id = int(facelets[f, r, c])
                    color = self._colors[color_id]

                    # Compute square corners in 3D
                    square = []
                    for (u, v) in [(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]:
                        x = origin[0] + du[0] * u + dv[0] * v
                        y = origin[1] + du[1] * u + dv[1] * v
                        z = origin[2] + du[2] * u + dv[2] * v
                        square.append((x, y, z))

                    poly = Poly3DCollection([square])
                    poly.set_facecolor(color.lower())  # 'w','r','g','y','o','b'
                    poly.set_edgecolor("k")
                    ax.add_collection3d(poly)

        # Turn off axes
        ax.set_axis_off()
        plt.show()

    def _assert_invariants(self):
        assert int(self._corner_orientation.sum()) % 3 == 0
        assert int(self._edges_orientation.sum()) % 2 == 0

    def _test_move(self, m):
        cp, co = self._corner_position.copy(), self._corner_orientation.copy()
        ep, eo = self._edges_position.copy(), self._edges_orientation.copy()

        # move + inverse → identity
        self.rotate(m, True);
        self.rotate(m, False)
        assert np.all(self._corner_position == cp);
        assert np.all(self._edges_position == ep)
        assert np.all(self._corner_orientation == co);
        assert np.all(self._edges_orientation == eo)

        # 4× same quarter-turn → identity
        for _ in range(4): self.rotate(m, True)
        assert np.all(self._corner_position == cp);
        assert np.all(self._edges_position == ep)
        assert np.all(self._corner_orientation == co);
        assert np.all(self._edges_orientation == eo)

        self._assert_invariants()

    def assert_roundtrip(self):
        """
        Keep your roundtrip test, but this now matches the +ori convention.
        """
        F = self._convert_cubelet_facelet()
        cp, co, ep, eo = self._decode_from_facelets(F)
        assert np.all(cp == self._corner_position), ("corner positions mismatch", cp, self._corner_position)
        assert np.all(co % 3 == self._corner_orientation % 3), ("corner orientations mismatch", co,
                                                                self._corner_orientation)
        assert np.all(ep == self._edges_position), ("edge positions mismatch", ep, self._edges_position)
        assert np.all(eo % 2 == self._edges_orientation % 2), ("edge orientations mismatch", eo,
                                                               self._edges_orientation)

    def _build_face_grids(self):
        """
        Define a consistent (r,c) frame for every face when viewed head-on.

        Convention (Singmaster):
          Faces ids: 0=U, 1=R, 2=F, 3=D, 4=L, 5=B.
          For each face, r increases downward, c increases rightward *as seen from that face*.

        We choose the following adjacency (standard net):
          - On U: top row touches B, bottom row touches F; left col touches L, right col touches R.
          - On F: top row touches U, bottom row touches D; left col L, right col R.
          - On R: top row U, bottom row D; left col F, right col B.
          - On L: top row U, bottom row D; left col B, right col F.
          - On D: top row F, bottom row B; left col L, right col R.
          - On B: top row U, bottom row D; left col R, right col L.  (note: mirrored)
        """
        grids = {f: np.empty((3, 3), dtype=object) for f in range(6)}
        # Fill with coordinate tuples
        for f in range(6):
            for r in range(3):
                for c in range(3):
                    grids[f][r, c] = (f, r, c)
        return grids

    def _map_rc(self, face_id: int, r: int, c: int) -> Tuple[int, int]:
        """
        Map logical (r,c) to storage (r,c) for a given face.
        Use a single consistent rule: rotate Back (face 5) by 180°.
        This aligns handedness of B with F so strips move as expected.
        """
        if face_id == 5:  # Back
            return 2 - r, 2 - c
        return r, c

    def _map_edge_rc(self, face_id: int, r: int, c: int):
        # Back face edges are mirrored horizontally
        if face_id == 5:  # B
            return r, 2 - c
        return r, c

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

    _colors: Dict[int,str] = {0:"white", 1:"red", 2:"green", 3:"yellow", 4:"orange", 5:"blue"}

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
    _corner_slot_facelets: List[List[Tuple[int,int,int]]] = [

        [(0, 2, 2), (1, 0, 0), (2, 0, 2)],  # 0: URF
        [(0, 2, 0), (2, 0, 0), (4, 0, 2)],  # 1: UFL
        [(0, 0, 0), (4, 0, 0), (5, 0, 2)],  # 2: ULB
        [(0, 0, 2), (5, 0, 0), (1, 0, 2)],  # 3: UBR
        [(3, 0, 2), (2, 2, 2), (1, 2, 0)],  # 4: DFR
        [(3, 0, 0), (4, 2, 2), (2, 2, 0)],  # 5: DLF
        [(3, 2, 0), (5, 2, 2), (4, 2, 0)],  # 6: DBL
        [(3, 2, 2), (1, 2, 2), (5, 2, 0)],  # 7: DRB
    ]
    _edge_slot_facelets: List[List[Tuple[int, int, int]]] = [
        [(0, 1, 2), (1, 0, 1)],  # 0: UR
        [(0, 2, 1), (2, 0, 1)],  # 1: UF
        [(0, 1, 0), (4, 0, 1)],  # 2: UL
        [(0, 0, 1), (5, 0, 1)],  # 3: UB
        [(3, 1, 2), (1, 2, 1)],  # 4: DR
        [(3, 0, 1), (2, 2, 1)],  # 5: DF
        [(3, 1, 0), (4, 2, 1)],  # 6: DL
        [(3, 2, 1), (5, 2, 1)],  # 7: DB
        [(2, 1, 2), (1, 1, 0)],  # 8: FR
        [(2, 1, 0), (4, 1, 2)],  # 9: FL
        [(5, 1, 0), (4, 1, 0)],  # 10: BL
        [(5, 1, 2), (1, 1, 2)],  # 11: BR
    ]

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



    def _validate_edges(self):
        pass

    def _validate_positions(self):
        pass

    def _rotation(self):
        pass

    def scramble(self):
        pass

    def rotate(self, face:str, clockwise: bool = True):
        pass

    def plot(self):
        pass

    def _convert_cubelet_facelet(self) -> np.ndarray:
        """
        Converts the cubelet representation (positions + orientations)
        into a 6×3×3 facelet array with ints 0–5 = face colors.

        Returns
        -------
        cube_facelet : np.ndarray
            Array of shape (6,3,3) with facelet colors.
        """
        cube_facelet = np.empty((6, 3, 3), dtype=int)

        # --- centers ---
        for f in range(6):
            cube_facelet[f, :, :] = f  # fill with default
            cube_facelet[f, 1, 1] = f  # center explicitly set

        # --- corners ---
        for slot in range(8):
            piece = self._corner_position[slot]
            ori = self._corner_orientation[slot]

            # canonical colors for this piece
            colors = list(self._corner_piece_colors[piece])

            # rotate by ori (mod 3)
            rotated = colors[ori:] + colors[:ori]

            # drop them into their three facelet coords
            for k, (f, r, c) in enumerate(self._corner_slot_facelets[slot]):
                cube_facelet[f, r, c] = rotated[k]

        # --- edges ---
        for slot in range(12):
            piece = self._edges_position[slot]
            ori = self._edges_orientation[slot]

            a, b = self._edge_piece_colors[piece]
            if ori == 0:
                colors = (a, b)
            else:
                colors = (b, a)

            for k, (f, r, c) in enumerate(self._edge_slot_facelets[slot]):
                cube_facelet[f, r, c] = colors[k]

        return cube_facelet

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




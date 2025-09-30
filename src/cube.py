'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import numpy as np
import pandas as pd


class Cube:

    _scramble_history: list
    _solve_history: list
    # Cubelet representation
    _edges_position: np.array
    _edges_orientation: np.array
    _corner_position: np.array
    _corner_orientation: np.array

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

    def _convert_cubelet_facelet(self):
        pass




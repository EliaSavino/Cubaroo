'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import numpy as np




def _cubie_faces(center, size=0.9):
    """
    Return dict face_name -> (quad 4x3) for the 6 faces of a cubie centered at `center`.
    Face names: 'U','D','F','B','R','L' (global directions).
    """
    cx, cy, cz = center
    s = size / 2.0
    # axis-aligned cube
    # Quads are defined CCW as seen from outside so lighting looks right.
    quads = {
        'U': np.array([[cx - s, cy + s, cz + s],
                       [cx + s, cy + s, cz + s],
                       [cx + s, cy + s, cz - s],
                       [cx - s, cy + s, cz - s]]),
        'D': np.array([[cx - s, cy - s, cz - s],
                       [cx + s, cy - s, cz - s],
                       [cx + s, cy - s, cz + s],
                       [cx - s, cy - s, cz + s]]),
        'F': np.array([[cx - s, cy - s, cz + s],
                       [cx + s, cy - s, cz + s],
                       [cx + s, cy + s, cz + s],
                       [cx - s, cy + s, cz + s]]),
        'B': np.array([[cx + s, cy - s, cz - s],
                       [cx - s, cy - s, cz - s],
                       [cx - s, cy + s, cz - s],
                       [cx + s, cy + s, cz - s]]),
        'R': np.array([[cx + s, cy - s, cz + s],
                       [cx + s, cy - s, cz - s],
                       [cx + s, cy + s, cz - s],
                       [cx + s, cy + s, cz + s]]),
        'L': np.array([[cx - s, cy - s, cz - s],
                       [cx - s, cy - s, cz + s],
                       [cx - s, cy + s, cz + s],
                       [cx - s, cy + s, cz - s]]),
    }
    return quads

def _rowcol_from_coord(val):
    """Map coordinate in {-1,0,1} to row/col in {0,1,2} using -1->0, 0->1, 1->2."""
    # robust to float fuzz
    if np.isclose(val, -1.0): return 0
    if np.isclose(val,  0.0): return 1
    return 2


def _rowcol_from_coord(val: float) -> int:
    # Map grid {-1,0,1} → {0,1,2} robustly
    if np.isclose(val, -1.0): return 0
    if np.isclose(val,  0.0): return 1
    return 2

def _color_lookup(F: np.ndarray, face_name: str, x: float, y: float, z: float) -> int:
    """
    Use the same face indexing you already use elsewhere:
      0: U (+z), 3: D (−z), 2: F (+y), 5: B (−y), 1: R (+x), 4: L (−x)
    Keep B mirrored on x like your original plot_3d.
    """
    rc = _rowcol_from_coord

    if face_name == 'U':        # z = +1
        f = 0; r = rc(z); c = rc(-x)
    elif face_name == 'D':      # z = -1
        f = 3; r = rc(y); c = rc(x)
    elif face_name == 'F':      # y = +1
        f = 2; r = rc(z); c = rc(x)
    elif face_name == 'B':      # y = -1  (mirror x)
        f = 5; r = rc(z); c = 2 - rc(x)
    elif face_name == 'R':      # x = +1
        f = 1; r = rc(z); c = rc(y)
    else:                       # 'L' x = -1
        f = 4; r = rc(z); c = rc(y)

    return int(F[f, r, c])

def _unit_cubie_quads(size: float = 0.94) -> dict[str, np.ndarray]:
    """Axis-aligned cubie at origin; returns 6 face quads (4×3 each)."""
    s = size / 2.0
    return {
        'U': np.array([[-s, +s, +s], [+s, +s, +s], [+s, +s, -s], [-s, +s, -s]]),
        'D': np.array([[-s, -s, -s], [+s, -s, -s], [+s, -s, +s], [-s, -s, +s]]),
        'F': np.array([[-s, -s, +s], [+s, -s, +s], [+s, +s, +s], [-s, +s, +s]]),
        'B': np.array([[+s, -s, -s], [-s, -s, -s], [-s, +s, -s], [+s, +s, -s]]),
        'R': np.array([[+s, -s, +s], [+s, -s, -s], [+s, +s, -s], [+s, +s, +s]]),
        'L': np.array([[-s, -s, -s], [-s, -s, +s], [-s, +s, +s], [-s, +s, -s]]),
    }
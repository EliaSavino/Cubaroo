'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Face ids: 0=U, 1=R, 2=F, 3=D, 4=L, 5=B
CORNER_SLOTS = ["URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB"]
EDGE_SLOTS   = ["UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR"]

# Slot → faces in SLOT ORDER (this order is the key for orientation projection)
SLOT2FACES_CORNER: Dict[str, Tuple[int, int, int]] = {
    "URF": (0, 1, 2),
    "UFL": (0, 2, 4),
    "ULB": (0, 4, 5),
    "UBR": (0, 5, 1),
    "DFR": (3, 2, 1),
    "DLF": (3, 4, 2),
    "DBL": (3, 5, 4),
    "DRB": (3, 1, 5),
}
SLOT2FACES_EDGE: Dict[str, Tuple[int, int]] = {
    "UR": (0, 1), "UF": (0, 2), "UL": (0, 4), "UB": (0, 5),
    "DR": (3, 1), "DF": (3, 2), "DL": (3, 4), "DB": (3, 5),
    "FR": (2, 1), "FL": (2, 4), "BL": (5, 4), "BR": (5, 1),
}

# Facelet coordinates for each slot, in the SAME slot order
CORNER_FACELETS: Dict[str, List[Tuple[int, int, int]]] = {
    "URF": [(0, 2, 2), (1, 0, 0), (2, 0, 2)],
    "UFL": [(0, 2, 0), (2, 0, 0), (4, 0, 2)],
    "ULB": [(0, 0, 0), (4, 0, 0), (5, 0, 2)],
    "UBR": [(0, 0, 2), (5, 0, 0), (1, 0, 2)],
    "DFR": [(3, 0, 2), (2, 2, 2), (1, 2, 0)],
    "DLF": [(3, 0, 0), (4, 2, 2), (2, 2, 0)],
    "DBL": [(3, 2, 0), (5, 2, 2), (4, 2, 0)],
    "DRB": [(3, 2, 2), (1, 2, 2), (5, 2, 0)],
}
EDGE_FACELETS: Dict[str, List[Tuple[int, int, int]]] = {
    "UR": [(0, 1, 2), (1, 0, 1)],
    "UF": [(0, 2, 1), (2, 0, 1)],
    "UL": [(0, 1, 0), (4, 0, 1)],
    "UB": [(0, 0, 1), (5, 0, 1)],
    "DR": [(3, 1, 2), (1, 2, 1)],
    "DF": [(3, 0, 1), (2, 2, 1)],
    "DL": [(3, 1, 0), (4, 2, 1)],
    "DB": [(3, 2, 1), (5, 2, 1)],
    "FR": [(2, 1, 2), (1, 1, 0)],
    "FL": [(2, 1, 0), (4, 1, 2)],
    # back is mirrored once here and nowhere else:
    "BL": [(5, 1, 2), (4, 1, 0)],
    "BR": [(5, 1, 0), (1, 1, 2)],
}

def _rot(t: Tuple[int, ...], k: int) -> Tuple[int, ...]:
    k %= len(t)
    return t[k:] + t[:k]

@dataclass
class Cubie:
    """
    Base class. 'stickers' are the cubie's canonical piece-order faces.
    For a corner piece (URF), stickers e.g. = (U,R,F) as face ids (0..5).
    For an edge piece (UR), stickers e.g. = (U,R).
    """
    stickers: Tuple[int, ...]     # canonical piece-order faces
    slot_name: str                # current slot name
    ori: int = 0                  # orientation (mod depends on subclass)

    # override in subclasses
    ORI_MOD: int = 1
    SLOT2FACES: Dict[str, Tuple[int, ...]] = None
    FACELETS: Dict[str, List[Tuple[int, int, int]]] = None

    def set_position(self, slot_name: str) -> None:
        self.slot_name = slot_name

    def set_orientation(self, ori: int) -> None:
        self.ori = ori % self.ORI_MOD

    def move_to(self, slot_name: str, ori: int) -> None:
        self.set_position(slot_name)
        self.set_orientation(ori)

    def stickers_in_slot_order(self) -> Tuple[int, ...]:
        """
        Return the cubie's sticker colors (face ids) in the current slot's order,
        after applying this cubie's orientation.
        The trick: rotate in the *piece index space* (self.stickers),
        then select by the index of each target face.
        """
        faces_here = list(self.SLOT2FACES[self.slot_name])
        base = list(self.stickers)
        rotated = list(_rot(tuple(base), self.ori))
        # for each target face f in faces_here, pick rotated[ base.index(f) ]
        try:
            return tuple(rotated[base.index(f)] for f in faces_here)
        except ValueError as e:
            # If this triggers, the piece is in an impossible slot (face sets don’t match).
            raise AssertionError(
                f"Piece {self.stickers} cannot sit in slot {self.slot_name} (faces={faces_here})."
            ) from e

    def placements_for_slot(self) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of placements (face, r, c, color_id) for this cubie
        given its current slot and orientation. The order matches the slot order.
        """
        colors = self.stickers_in_slot_order()
        coords = self.FACELETS[self.slot_name]
        return [(f, r, c, col) for (col, (f, r, c)) in zip(colors, coords)]

class CornerCubie(Cubie):
    ORI_MOD = 3
    SLOT2FACES = SLOT2FACES_CORNER
    FACELETS = CORNER_FACELETS

class EdgeCubie(Cubie):
    ORI_MOD = 2
    SLOT2FACES = SLOT2FACES_EDGE
    FACELETS = EDGE_FACELETS

    def stickers_in_slot_order(self) -> Tuple[int, ...]:
        """
        Edges: orientation 0 keeps (a,b), orientation 1 swaps them in piece index space,
        then reorders to the slot's faces (same method as base).
        """
        faces_here = list(self.SLOT2FACES[self.slot_name])
        base = list(self.stickers)
        rotated = base if (self.ori % 2) == 0 else [base[1], base[0]]
        try:
            return tuple(rotated[base.index(f)] for f in faces_here)
        except ValueError as e:
            raise AssertionError(
                f"Edge {self.stickers} cannot sit in slot {self.slot_name} (faces={faces_here})."
            ) from e
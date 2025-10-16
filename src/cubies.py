'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

from dataclasses import dataclass
from typing import Dict, List, Tuple, ClassVar

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
# canonical *piece* colors (piece id equals its home slot index)
CORNER_PIECE_COLORS: List[Tuple[int,int,int]] = [
    (0,1,2),(0,2,4),(0,4,5),(0,5,1),(3,2,1),(3,4,2),(3,5,4),(3,1,5)
]
EDGE_PIECE_COLORS: List[Tuple[int,int]] = [
    (0,1),(0,2),(0,4),(0,5),(3,1),(3,2),(3,4),(3,5),(2,1),(2,4),(5,4),(5,1)
]

def _rot(t: Tuple[int, ...], k: int) -> Tuple[int, ...]:
    k %= len(t)
    return t[k:] + t[:k]

@dataclass
class Cubie:
    # current seat (slot) + orientation (relative to the piece’s canonical sticker order)
    slot_name: str
    ori: int
    # immutable: the piece’s own sticker colors in canonical order
    stickers: Tuple[int, ...]
    # runtime: index of the piece (0..7 / 0..11) if you want to track it
    piece_idx: int | None = None

    # class-level config supplied by subclasses
    ORI_MOD: ClassVar[int]
    SLOT2FACES: ClassVar[Dict[str, Tuple[int, ...]]]
    FACELETS: ClassVar[Dict[str, List[Tuple[int,int,int]]]]

    def move_to(self, dest_slot: str, delta_ori: int) -> None:
        """Re-seat the SAME cubie object into a new slot and update its orientation in place."""
        self.slot_name = dest_slot
        self.ori = (self.ori + delta_ori) % self.ORI_MOD

    # --- rendering helpers ---
    def stickers_in_slot_order(self) -> Tuple[int, ...]:
        """Return this piece’s sticker colors arranged to the slot order, given current ori."""
        if self.ORI_MOD == 3:  # corner
            base = list(self.stickers)
            o = self.ori % 3
            return tuple(base[o:] + base[:o])  # rotate CW by ori
        else:  # edge (mod 2)
            a, b = self.stickers
            return (a, b) if (self.ori % 2) == 0 else (b, a)

    def placements_for_slot(self):
        """
        Yield (face, r, c, color) for this cubie’s stickers, using current slot+ori.
        Zips slot faces with the oriented colors, then looks up absolute facelet coords.
        """
        colors = self.stickers_in_slot_order()
        faces  = self.SLOT2FACES[self.slot_name]
        coords = self.FACELETS[self.slot_name]
        for (face_id, (f, r, c), col) in zip(faces, coords, colors):
            # sanity: face_id should equal f in a correct table set; no need to assert though
            yield (face_id, r, c, col)

class CornerCubie(Cubie):
    ORI_MOD = 3
    SLOT2FACES = SLOT2FACES_CORNER
    FACELETS = CORNER_FACELETS

class EdgeCubie(Cubie):
    ORI_MOD = 2
    SLOT2FACES = SLOT2FACES_EDGE
    FACELETS = EDGE_FACELETS
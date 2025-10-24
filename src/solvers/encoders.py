'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from typing import Protocol, runtime_checkable
import numpy as np


# =============================================================================
# ── Protocol ──────────────────────────────────────────────────────────────────
# =============================================================================

@runtime_checkable
class CubeEncoderProtocol(Protocol):
    """
    Protocol for cube state encoders.

    Any conforming encoder must expose:
        - ``dim`` : int — dimensionality of the encoded vector.
        - ``encode(cube) -> np.ndarray`` — converts a cube object into a
          NumPy vector representation suitable for ML models.

    Notes
    -----
    This allows flexible typing such that any class implementing `.encode`
    and `.dim` can be used interchangeably with type checkers and IDEs.

    Examples
    --------
    >>> def forward_pass(encoder: CubeEncoderProtocol, cube):
    ...     x = encoder.encode(cube)
    ...     return x.shape == (encoder.dim,)
    """
    dim: int

    def encode(self, cube) -> np.ndarray: ...


# =============================================================================
# ── Concrete Implementations ─────────────────────────────────────────────────
# =============================================================================

class IndexCubieEncoder:
    """
    Encodes the cube as a vector of **integer indices**.

    Layout:
        [corner_perm(8), corner_ori(8), edge_perm(12), edge_ori(12)]

    Total dimension: ``8 + 8 + 12 + 12 = 40``

    Each element is an integer ID that can be directly used for learned
    embeddings (e.g., Transformer tokenization).

    Attributes
    ----------
    dim : int
        Dimensionality of the encoded vector (40).
    """

    dim: int = 40

    def encode(self, cube) -> np.ndarray:
        """
        Encode the cube into a vector of integer indices.

        Parameters
        ----------
        cube : object
            Cube instance exposing:
                - ``cube.corners``: iterable of corner cubies with attributes
                  `.piece_idx` and `.ori`.
                - ``cube.edges``: iterable of edge cubies with attributes
                  `.piece_idx` and `.ori`.

        Returns
        -------
        np.ndarray
            Integer vector of shape (40,), dtype int64.
        """
        corners_perm = [c.piece_idx for c in cube.corners]
        corners_ori  = [c.ori for c in cube.corners]
        edges_perm   = [e.piece_idx for e in cube.edges]
        edges_ori    = [e.ori for e in cube.edges]
        return np.array(
            corners_perm + corners_ori + edges_perm + edges_ori,
            dtype=np.int64,
        )


class FlatCubieEncoder:
    """
    Flattens permutation and orientation directly as **float values**.

    Equivalent information as :class:`IndexCubieEncoder` but concatenated as
    float32 for use in simple dense networks (no one-hot encoding).

    Attributes
    ----------
    dim : int
        Dimensionality of the encoded vector (40).
    """

    dim: int = 40

    def encode(self, cube) -> np.ndarray:
        """
        Encode the cube into a flattened float vector.

        Parameters
        ----------
        cube : object
            Cube instance exposing `.corners` and `.edges`, each with
            attributes `.piece_idx` and `.ori`.

        Returns
        -------
        np.ndarray
            Float vector of shape (40,), dtype float32.
        """
        vals: list[float] = []
        for c in cube.corners:
            vals += [c.piece_idx, c.ori]
        for e in cube.edges:
            vals += [e.piece_idx, e.ori]
        return np.array(vals, dtype=np.float32)


class CubieEncoder:
    """
    One-hot encoder for cube permutation & orientation features.

    Produces a **256-dimensional float vector** concatenating one-hot
    encodings for each cubie:

    - 8 corners × (perm:8 + ori:3) = 88
    - 12 edges × (perm:12 + ori:2) = 168
      → Total = 256

    Attributes
    ----------
    dim : int
        Dimensionality of the encoded vector (256).
    """

    dim: int = 256

    def encode(self, cube) -> np.ndarray:
        """
        Encode cube cubies as a concatenated one-hot vector.

        Parameters
        ----------
        cube : object
            Cube with lists ``corners`` and ``edges`` whose elements have
            integer attributes ``piece_idx`` and ``ori``.

        Returns
        -------
        np.ndarray
            Float32 one-hot vector of shape (256,).
        """
        vec: list[np.ndarray] = []

        for c in cube.corners:
            one_perm = np.zeros(8, dtype=np.float32)
            one_perm[c.piece_idx] = 1.0
            one_ori = np.zeros(3, dtype=np.float32)
            one_ori[c.ori] = 1.0
            vec.extend([one_perm, one_ori])

        for e in cube.edges:
            one_perm = np.zeros(12, dtype=np.float32)
            one_perm[e.piece_idx] = 1.0
            one_ori = np.zeros(2, dtype=np.float32)
            one_ori[e.ori] = 1.0
            vec.extend([one_perm, one_ori])

        return np.concatenate(vec).astype(np.float32)
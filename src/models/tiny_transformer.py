'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
from typing import Optional
import torch
from torch import nn
import math
N_ACTIONS = 12  # fixed for Rubik's Cube (6 faces * 3 turns)




class PositionalEncoding(nn.Module):
    """
    Standard sine–cosine positional encoding (Vaswani et al., 2017).

    Builds a fixed table ``pe`` of shape ``[1, max_len, d_model]`` and adds it to
    the input sequence.

    Parameters
    ----------
    d_model : int
        Embedding dimension (model width).
    max_len : int, default=64
        Maximum supported sequence length.

    Notes
    -----
    - Works for both even and odd ``d_model``; if odd, the last odd channel will
      only receive the sine term (cosine slice is shorter), which is standard.
    - ``pe`` is registered as a non-persistent buffer so checkpoints stay small.

    Examples
    --------
    >>> pe = PositionalEncoding(128, max_len=40)
    >>> x = torch.zeros(2, 40, 128)
    >>> y = pe(x)
    >>> y.shape
    torch.Size([2, 40, 128])
    """

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )  # [ceil(d_model/2)]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # [1, max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, T, D]`` with ``D == d_model``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[B, T, D]`` with positions added.
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerQNet(nn.Module):
    """
    Transformer-based Q-network for **index/flat cubie encodings**.

    Input tokens follow the fixed order (length 40):

        ``[corner_perm(8), corner_ori(8), edge_perm(12), edge_ori(12)]``

    Each field is an integer index (i.e., categorical feature). We embed each stream
    with a dedicated embedding table and **sum** the pairwise streams per cubie
    (perm + ori), resulting in 20 tokens total (8 corners + 12 edges), each with
    ``d_model`` channels. A Transformer encoder then produces a sequence representation
    which is mean-pooled and projected to per-action Q-values.

    Parameters
    ----------
    d_model : int, default=128
        Embedding width and Transformer model dimension.
    nhead : int, default=8
        Number of attention heads.
    num_layers : int, default=3
        Number of Transformer encoder layers.
    dim_feedforward : int, default=256
        Hidden size of the encoder MLP.
    dropout : float, default=0.1
        Dropout rate inside encoder layers.
    n_actions : int, default=N_ACTIONS (12)
        Number of discrete actions.

    Inputs
    ------
    tokens : torch.LongTensor
        Shape ``[B, 40]``; integer indices as produced by `IndexCubieEncoder`.

    Returns
    -------
    torch.Tensor
        Q-values of shape ``[B, n_actions]``.

    Notes
    -----
    - Vocab sizes: corner_perm=8, corner_ori=3, edge_perm=12, edge_ori=2.
    - Positional encoding length is fixed to the **20** token sequence post-merge.
    - Pooling uses simple mean over tokens; you can swap to ``x[:, 0]`` or attention
      pooling without changing the interface.

    Examples
    --------
    >>> net = TransformerQNet(d_model=128, nhead=8, num_layers=2)
    >>> tok = torch.randint(0, 12, (4, 40), dtype=torch.long)
    >>> q = net(tok)
    >>> q.shape
    torch.Size([4, 12])
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()
        T_in = 40  # input token length (IndexCubieEncoder)
        T_out = 20  # tokens after merging (8 corners + 12 edges)

        # Embedding tables per feature stream
        self.emb_corner_perm = nn.Embedding(8, d_model)
        self.emb_corner_ori = nn.Embedding(3, d_model)
        self.emb_edge_perm = nn.Embedding(12, d_model)
        self.emb_edge_ori = nn.Embedding(2, d_model)

        # Positional encoding over merged sequence length
        self.pos = PositionalEncoding(d_model, max_len=T_out)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simple head over pooled representation
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions),
        )

        # Slices on the 40-token input
        self.idx_cperm = slice(0, 8)
        self.idx_cori = slice(8, 16)
        self.idx_eperm = slice(16, 28)
        self.idx_eori = slice(28, 40)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute per-action Q-values from index tokens.

        Parameters
        ----------
        tokens : torch.Tensor
            Long tensor of shape ``[B, 40]`` with index tokens.

        Returns
        -------
        torch.Tensor
            Q-values of shape ``[B, n_actions]``.

        Raises
        ------
        AssertionError
            If ``tokens.dtype`` is not ``torch.long``.
        """
        assert tokens.dtype == torch.long, "TransformerQNet expects integer token indices"

        # Split streams: [B, 8], [B, 8], [B, 12], [B, 12]
        cperm = tokens[:, self.idx_cperm]
        cori = tokens[:, self.idx_cori]
        eperm = tokens[:, self.idx_eperm]
        eori = tokens[:, self.idx_eori]

        # Embed each stream
        ecperm = self.emb_corner_perm(cperm)  # [B, 8, D]
        ecori = self.emb_corner_ori(cori)     # [B, 8, D]
        eeperm = self.emb_edge_perm(eperm)    # [B, 12, D]
        eeori = self.emb_edge_ori(eori)       # [B, 12, D]

        # Merge perm + ori per cubie, then concatenate corners and edges
        corners = ecperm + ecori              # [B, 8, D]
        edges = eeperm + eeori                # [B, 12, D]
        x = torch.cat([corners, edges], dim=1)  # [B, 20, D]

        # Positional encoding + Transformer encoder
        x = self.pos(x)    # [B, 20, D]
        x = self.enc(x)    # [B, 20, D]

        # Pool and predict Q-values
        x = x.mean(dim=1)  # [B, D]
        q = self.head(x)   # [B, n_actions]
        return q
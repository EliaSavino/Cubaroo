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
    """Standard sine-cosine positional encoding."""
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerQNet(nn.Module):
    """
    Q-network for index/flat cubie encodings (40 tokens total):
      order = [corner_perm(8), corner_ori(8), edge_perm(12), edge_ori(12)]
    Each field is an integer index; we embed each stream separately:
      - corner_perm: vocab 8
      - corner_ori : vocab 3
      - edge_perm  : vocab 12
      - edge_ori   : vocab 2
    We then sum the four stream-embeddings per position (like multiple feature types per token).

    Input expected:
      tokens: LongTensor [B, 40]  (from IndexCubieEncoder)
    Output:
      Q-values: [B, n_actions]
    """
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        T = 40  # fixed sequence length

        # Embedding tables per feature stream
        self.emb_corner_perm = nn.Embedding(8, d_model)
        self.emb_corner_ori  = nn.Embedding(3, d_model)
        self.emb_edge_perm   = nn.Embedding(12, d_model)
        self.emb_edge_ori    = nn.Embedding(2, d_model)

        self.pos = PositionalEncoding(d_model, max_len=T)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool with mean over tokens (or take token-0; mean works fine here)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_actions),
        )

        # Precompute slices
        self.idx_cperm = slice(0, 8)
        self.idx_cori  = slice(8, 16)
        self.idx_eperm = slice(16, 28)
        self.idx_eori  = slice(28, 40)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, 40] LongTensor
        returns: [B, n_actions]
        """
        assert tokens.dtype == torch.long, "TransformerQNet expects integer token indices"

        cperm = tokens[:, self.idx_cperm]  # [B, 8]
        cori  = tokens[:, self.idx_cori]   # [B, 8]
        eperm = tokens[:, self.idx_eperm]  # [B, 12]
        eori  = tokens[:, self.idx_eori]   # [B, 12]

        # embed each stream and sum at each token position
        # We need matching lengths: construct 40 embeddings by concatenating the streams.
        ecperm = self.emb_corner_perm(cperm)  # [B, 8, D]
        ecori  = self.emb_corner_ori(cori)    # [B, 8, D]
        eeperm = self.emb_edge_perm(eperm)    # [B, 12, D]
        eeori  = self.emb_edge_ori(eori)      # [B, 12, D]

        # sum pairs at matching positions
        corners = ecperm + ecori              # [B, 8, D]
        edges   = eeperm + eeori              # [B, 12, D]
        x = torch.cat([corners, edges], dim=1)  # [B, 20, D]  (one token per cubie)

        # Optional: duplicate tokens to include ori as separate tokens (not needed here).
        # Add positional encodings and encode
        x = self.pos(x)                       # [B, 20, D]
        x = self.enc(x)                       # [B, 20, D]

        # Pool
        x = x.mean(dim=1)                     # [B, D]
        q = self.head(x)                      # [B, n_actions]
        return q
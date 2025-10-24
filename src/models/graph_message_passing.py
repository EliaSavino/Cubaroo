'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import torch
from torch import nn
import torch.nn.functional as F

from src.models.embedding_model import CubieTokenEmbedding

N_ACTIONS = 12
# fixed for Rubik's Cube (6 faces * 3 turns)
def cube_bipartite_edges(device):
    """
    Hard-coded corner<->edge adjacency for a standard cube net.
    Returns two index tensors (src, dst) for undirected edges (both directions).
    """
    # Each corner touches 3 edges. Define pairs (corner_idx, edge_idx+8).
    pairs = [
        (0, 8), (0, 9), (0,10),
        (1, 9), (1,11), (1,12),
        (2,10), (2,13), (2,14),
        (3,11), (3,15), (3,12),
        (4,16), (4,17), (4,8),
        (5,17), (5,18), (5,11),
        (6,18), (6,19), (6,13),
        (7,19), (7,16), (7,15),
    ]
    c2e = torch.tensor(pairs, dtype=torch.long, device=device)
    e2c = torch.stack([c2e[:,1], c2e[:,0]], dim=1)  # reverse
    edges = torch.cat([c2e, torch.flip(c2e, dims=[1]), e2c, torch.flip(e2c, dims=[1])], dim=0)
    # edges shape [E,2] of (src,dst)
    return edges[:,0], edges[:,1]

class GraphLayer(nn.Module):
    """ Simple residual message passing: h' = LN(h + σ( A h W )) """
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model, bias=False)
        self.ln  = nn.LayerNorm(d_model)
    def forward(self, H, src, dst):
        # H: [B,N,D], edges via (src,dst)
        B, N, D = H.shape
        m = self.lin(H)                          # [B,N,D]
        agg = torch.zeros_like(m)
        agg.index_add_(1, dst, m[:, src, :])     # sum messages into dst
        return self.ln(H + F.gelu(agg))

class CubeGNNQNet(nn.Module):
    """
    20-node cube graph (8 corners + 12 edges). 3–5 GNN layers usually work well.
    """
    def __init__(self, d_model=128, layers=4, n_actions=N_ACTIONS):
        super().__init__()
        self.embed = CubieTokenEmbedding(d_model)
        self.gnns  = nn.ModuleList([GraphLayer(d_model) for _ in range(layers)])
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_actions)
        )
    def forward(self, tokens):
        X = self.embed(tokens)               # [B,20,D]
        device = X.device
        src, dst = cube_bipartite_edges(device)
        H = X
        for g in self.gnns:
            H = g(H, src, dst)
        G = H.mean(dim=1)                    # global mean
        return self.readout(G)
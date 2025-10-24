'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''
import torch
from torch import nn


class CubieTokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.emb_corner_perm = nn.Embedding(8, d_model)
        self.emb_corner_ori  = nn.Embedding(3, d_model)
        self.emb_edge_perm   = nn.Embedding(12, d_model)
        self.emb_edge_ori    = nn.Embedding(2, d_model)
        self.idx_cperm = slice(0, 8)
        self.idx_cori  = slice(8, 16)
        self.idx_eperm = slice(16, 28)
        self.idx_eori  = slice(28, 40)
    def forward(self, tokens):
        cperm = tokens[:, self.idx_cperm]
        cori  = tokens[:, self.idx_cori]
        eperm = tokens[:, self.idx_eperm]
        eori  = tokens[:, self.idx_eori]
        corners = self.emb_corner_perm(cperm) + self.emb_corner_ori(cori)   # [B,8,D]
        edges   = self.emb_edge_perm(eperm)  + self.emb_edge_ori(eori)      # [B,12,D]
        return torch.cat([corners, edges], dim=1)

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
N_ACTIONS = 12  # fixed for Rubik's Cube (6 faces * 3 turns)

class ResidualConv1D(nn.Module):
    def __init__(self, d, k=3, dilation=1):
        super().__init__()
        pad = (k-1)//2 * dilation
        self.conv1 = nn.Conv1d(d, d, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(d, d, k, padding=pad, dilation=dilation)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):  # x: [B,T,D]
        y = x.transpose(1,2)            # [B,D,T]
        y = F.gelu(self.conv1(y))
        y = self.conv2(y)
        y = y.transpose(1,2)            # [B,T,D]
        return self.ln(x + y)

class TCNQNet(nn.Module):
    """
    Token embeddings -> stack of dilated 1D convs -> mean pool -> Q.
    """
    def __init__(self, d_model=128, blocks=4, n_actions=N_ACTIONS):
        super().__init__()
        self.embed = CubieTokenEmbedding(d_model)
        dilations = [1, 2, 4, 8][:blocks]
        self.blocks = nn.ModuleList([ResidualConv1D(d_model, k=3, dilation=d) for d in dilations])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, n_actions)
        )
    def forward(self, tokens):
        x = self.embed(tokens)          # [B,20,D]
        for b in self.blocks:
            x = b(x)
        g = x.mean(dim=1)               # [B,D]
        return self.head(g)
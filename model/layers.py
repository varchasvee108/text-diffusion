import torch.nn as nn
import torch
from core.config import Config
import math


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2

        indices = torch.arange(half_dim, device=device)
        emb_scale = math.log(1000) / (half_dim - 1)

        freqs = torch.exp(indices * -emb_scale)

        emb = timesteps.unsqueeze(1) * freqs.unsqueeze(0)

        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

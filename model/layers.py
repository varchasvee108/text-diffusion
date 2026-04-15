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


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.model.n_embd)
        self.ln2 = nn.LayerNorm(config.model.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.model.n_embd,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.model.n_embd, config.model.hidden_dim),
            nn.GELU(),
            nn.Linear(config.model.hidden_dim, config.model.n_embd),
        )

        self.adaln_mod = nn.Sequential(
            nn.SiLU(), nn.Linear(config.model.n_embd, 6 * config.model.n_embd)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:

        chunks = self.adaln_mod(t_emb).unsqueeze(1).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        x_norm = self.ln1(x)
        x_mod = x_norm * (1 + scale_msa) + shift_msa

        attn, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa * attn

        x_mod = self.ln2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_mod)
        return x

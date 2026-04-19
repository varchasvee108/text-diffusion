import torch
import torch.nn as nn
from model.layers import SinusoidalEmbeddings, TransformerBlock
from core.config import Config


class DiffusionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.data.vocab_size is None:
            raise ValueError(
                "vocab isze must be defined and set in config before initialisation"
            )
        self.tok_embd = nn.Embedding(config.data.vocab_size, config.model.n_embd)
        self.pos_embd = nn.Parameter(
            torch.randn(1, config.data.block_size, config.model.n_embd)
        )
        self.time_embd = nn.Sequential(
            SinusoidalEmbeddings(config.model.time_embd),
            nn.Linear(config.model.time_embd, 4 * config.model.time_embd),
            nn.SiLU(),
            nn.Linear(4 * config.model.time_embd, config.model.n_embd),
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config=config) for _ in range(config.model.num_layers)]
        )
        self.adaln_mod = nn.Sequential(
            nn.SiLU(), nn.Linear(config.model.n_embd, 2 * config.model.n_embd)
        )
        self.ln_f = nn.LayerNorm(config.model.n_embd, elementwise_affine=False)
        self.proj_out = nn.Linear(config.model.n_embd, config.model.n_embd)

    def forward(self, x, t, key_padding_mask=None):

        if x.ndim == 2:
            B, T = x.shape
            x = self.tok_embd(x)
        else:
            B, T, C = x.shape

        x = x + self.pos_embd[:, :T, :]

        t_emb = self.time_embd(t)  # (B, n_embd)

        for block in self.transformer_blocks:
            x = block(x, t_emb, key_padding_mask=key_padding_mask)

        shift, scale = self.adaln_mod(t_emb).unsqueeze(1).chunk(2, dim=-1)

        x = self.ln_f(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)
        return x

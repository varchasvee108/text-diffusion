import torch
from core.config import Config


class Diffusion:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.timesteps = config.diffusion.timesteps

        self.betas = torch.linspace(
            config.diffusion.beta_start,
            config.diffusion.beta_end,
            self.timesteps,
            device=device,
        )

        self.alphas = 1.0 - self.betas

        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)

import torch
from core.config import Config
from tqdm import tqdm


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

    def sample_timesteps(self, batch_size: int):
        return torch.randint(
            self.timesteps, (batch_size,), device=self.device, dtype=torch.long
        )

    def add_noise_to_embeddings(self, x0, t, noise=None) -> torch.Tensor:

        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_hat = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    def predict_x0(self, x_t, t, noise_pred):
        sqrt_alpha_hat = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)

        return (x_t - sqrt_one_minus * noise_pred) / sqrt_alpha_hat

    def p_sample(self, model, x, t, temp=1.0):
        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_hat = self.alpha_cumprod[t].view(-1, 1, 1)

        noise_pred = model(x, t)

        mean = self.sqrt_recip_alpha[t].view(-1, 1, 1) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred
        )
        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, 1, 1)
        return mean + (mask * torch.sqrt(beta * temp) * noise)

    @torch.inference_mode()
    def sample(self, model, shape, temp=1.0):
        model.eval()
        x = torch.randn(shape, device=self.device)
        pbar = tqdm(reversed(range(self.config.diffusion.timesteps)))

        for t in pbar:
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, temp=temp)

        return x

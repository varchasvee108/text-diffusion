import torch
from torch.amp import GradScaler, autocast  # type:ignore
from pathlib import Path
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from core.config import Config
import torch.nn as nn
from diffusion.diffusion import Diffusion
from torch.optim import Optimizer
from model.model import DiffusionModel


class Trainer:
    def __init__(
        self,
        config: Config,
        train_dataloader,
        val_dataloader,
        tokenizer,
        model: DiffusionModel,
        diffusion: Diffusion,
        optimizer: Optimizer,
        scheduler,
        device: torch.device,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.step = 0

        self.scaler = GradScaler(enabled=(device.type == "cuda"))

        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        wandb.init(
            project=self.config.project.name,
            config=self.config.model_dump(),
            name=f"{self.config.project.seed}",
        )

    def train(self):
        self.model.train()
        pbar = tqdm(total=self.config.training.max_steps, desc="Training")

        train_iter = iter(self.train_dataloader)

        for _ in pbar:
            try:
                batch = next(train_iter)

            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)

            loss = self._train_step(batch)

            self.step += 1
            # self._log_train(loss)
            # self._eval()
            # self._save_checkpoint()
        wandb.finish()

    def _train_step(self, batch):
        x0 = batch["input_ids"].to(self.device)
        t = self.diffusion.sample_timesteps(x0.shape[0])

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=self.device.type, enabled=(self.device.type == "cuda")
        ):
            x0_emb = self.model.tok_embd(x0)
            noise = torch.randn_like(x0_emb)

            xt = self.diffusion.add_noise_to_embeddings(x0=x0, t=t, noise=noise)
            pred_noise = self.model(xt, t)

            loss = F.mse_loss(pred_noise, noise)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer=self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

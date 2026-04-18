import torch
from torch.amp import GradScaler, autocast  # type:ignore
from pathlib import Path
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from core.config import Config

from diffusion.diffusion import Diffusion
from torch.optim import Optimizer
from model.model import DiffusionModel
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        config: Config,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tokenizer: PreTrainedTokenizerBase,
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
        self.best_loss = float("inf")

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
            pbar.update(1)
            self._log_train(loss)
            self._log_eval()
            self._save_checkpoint(loss)
        wandb.finish()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(self.val_dataloader, desc="Evaluation")

        for batch in pbar:
            loss = self._eval(batch)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        sample_text = self._sample_text()
        wandb.log(
            {
                "val/loss": avg_loss,
                "step": self.step,
                "sample_text": wandb.Html(f"<pre>{sample_text}<pre>"),
            }
        )

    def _train_step(self, batch):
        x0 = batch["input_ids"].to(self.device)
        t = self.diffusion.sample_timesteps(x0.shape[0])

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=self.device.type, enabled=(self.device.type == "cuda")
        ):
            x0_emb = self.model.tok_embd(x0)
            noise = torch.randn_like(x0_emb)

            xt = self.diffusion.add_noise_to_embeddings(x0=x0_emb, t=t, noise=noise)
            attn_mask = batch["attention_mask"].to(self.device)
            key_padding_mask = attn_mask == 0
            pred_noise = self.model(xt, t, key_padding_mask=key_padding_mask)

            loss = F.mse_loss(pred_noise, noise)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer=self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def _eval(self, batch):
        x0 = batch["input_ids"].to(self.device)
        t = self.diffusion.sample_timesteps(x0.shape[0])

        with autocast(device_type=self.device.type, enabled=(torch.device == "cuda")):
            x0_emb = self.model.tok_embd(x0)
            noise = torch.randn_like(x0_emb)

            xt = self.diffusion.add_noise_to_embeddings(x0=x0, t=t, noise=noise)
            pred_noise = self.model(xt, t)
            loss = F.mse_loss(pred_noise, noise)
        return loss.item()

    @torch.no_grad()
    def _sample_text(self):
        shape = (1, self.config.data.block_size, self.config.model.n_embd)

        sampled_emb = self.diffusion.sample(self.model, shape, temp=1.0)

        logits = torch.matmul(sampled_emb, self.model.tok_embd.weight.T)

        tokens = torch.argmax(logits, dim=-1)
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def _log_train(self, loss):
        if self.step > 0 and self.step % 100 == 0:
            wandb.log(
                {
                    "train/loss": loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.step,
                }
            )

    def _log_eval(self):
        if self.step > 0 and self.step % 500 == 0:
            self.evaluate()

    def _save_checkpoint(self, loss: float):

        last_path = Path(self.checkpoint_dir / "ckpt_last.pt")
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            last_path,
        )
        if loss < self.best_loss:
            self.best_loss = loss
            path = Path(self.checkpoint_dir / f"ckpt_{self.step}_best-loss.pt")
            torch.save(
                {
                    "step": self.step,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                path,
            )

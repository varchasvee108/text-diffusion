import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler  # type: ignore

from core.config import Config
from diffusion.diffusion import Diffusion
from model.model import DiffusionModel
from trainer.trainer import Trainer
from data.dataloader import build_dataloader
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def build_tokenizer(config: Config):
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_scheduler(config: Config, optimizer):
    return get_scheduler(
        name=config.training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )


def build_optimizer(config: Config, model):
    decay = set()
    no_decay = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith("bias") or "ln" in name.lower() or "norm" in name.lower():
            no_decay.add(name)
        else:
            decay.add(name)

    param_dict = {name: param for name, param in model.named_parameters()}

    optim_groups = [
        {
            "params": [param_dict[n] for n in sorted(decay)],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [param_dict[n] for n in sorted(no_decay)],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(
        optim_groups,
        lr=config.training.lr,
        betas=config.training.betas,
    )


def build_model(config: Config, tokenizer, device):
    config = config.model_copy(deep=True)
    config.data.vocab_size = len(tokenizer)
    return DiffusionModel(config).to(device)


def build_diffusion(config: Config, device):
    return Diffusion(config, device)


def build_trainer(config: Config):
    device = get_device()

    tokenizer = build_tokenizer(config)
    train_dataloader, val_dataloader = build_dataloader(config, tokenizer=tokenizer)

    model = build_model(config, tokenizer, device)
    diffusion = build_diffusion(config, device)

    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    trainer = Trainer(
        config=config,
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
    )
    return trainer

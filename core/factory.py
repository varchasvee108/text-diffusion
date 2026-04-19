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
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def build_scheduler(config: Config, optimizer):
    return get_scheduler(
        name=config.training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps
        // config.training.grad_accum_steps,
        num_training_steps=config.training.max_steps
        // config.training.grad_accum_steps,
    )


def build_optimizer(config, model):

    decay_params = []
    no_decay_params = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            if isinstance(module, (nn.LayerNorm, nn.Embedding)):
                no_decay_params.append(param)
            elif param_name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        optim_groups,
        lr=config.training.lr,
        betas=config.training.betas,
    )


def build_model(config: Config, tokenizer, device):
    config = config.model_copy(deep=True)
    config.data.vocab_size = len(tokenizer)

    model = DiffusionModel(config).to(device)

    return model


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
        tokenizer=tokenizer,
        device=device,
    )
    return trainer

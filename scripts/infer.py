import torch
from core.config import Config
from core.factory import build_trainer
from pathlib import Path


def main():
    config = Config.load_config(Path("config/config.toml"))
    trainer = build_trainer(config)

    ckpt = torch.load(Path("checkpoints/ckpt_last.pt"), map_location=trainer.device)
    trainer.model.load_state_dict(ckpt["model"])
    trainer.model.eval()

    text = trainer._sample_text()

    print("\n === Generated Text === \n")
    print(text)


if __name__ == "__main__":
    main()

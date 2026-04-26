from core.config import Config
from pathlib import Path
from core.factory import build_trainer


def main():
    config = Config.load_config(Path("config/config.toml"))
    trainer = build_trainer(config)

    ckpt_path = Path("checkpoints/ckpt_best_loss.pt")

    if ckpt_path.exists():
        trainer.load_checkpoint(ckpt_path)
    else:
        print("No checkpoint found. Starting fresh training.")

    trainer.train()


if __name__ == "__main__":
    main()

from core.config import Config
from pathlib import Path
from core.factory import build_trainer


def main():
    config = Config.load_config(Path("config/config.toml"))
    trainer = build_trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

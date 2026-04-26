from pathlib import Path
import tomllib
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    seed: int = Field(ge=0)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    block_size: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    tokenizer: str
    vocab_size: int | None = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_embd: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    dropout: float = Field(gt=0.0, lt=1.0)
    time_embd: int = Field(gt=0)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = Field(gt=0.0)
    max_steps: int = Field(gt=0)
    warmup_steps: int = Field(gt=0)
    betas: tuple[float, float]
    weight_decay: float = Field(gt=0.0)
    grad_clip: float = Field(gt=0.0)
    eval_interval: int = Field(gt=0)
    save_interval: int = Field(gt=0)
    scheduler: Literal["cosine", "linear", "constant", "cosine_with_restarts"] = (
        "cosine_with_restarts"
    )
    grad_accum_steps: int = Field(gt=0)


class DiffusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timesteps: int = Field(gt=0)
    beta_schedule: Literal["linear", "cosine", "sqrt"] = "linear"
    beta_start: float = Field(gt=0.0, lt=1.0)
    beta_end: float = Field(gt=0.0, lt=1.0)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    diffusion: DiffusionConfig

    @classmethod
    def load_config(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "rb") as f:
            toml_dict = tomllib.load(f)
        return cls.model_validate(toml_dict)

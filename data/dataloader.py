from datasets import load_dataset
from torch.utils.data import DataLoader
from core.config import Config


def build_dataloader(config: Config, tokenizer):
    dataset = load_dataset(config.data.dataset, "wikitext-2-raw-v1")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config.data.block_size,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
    train_dataloader = DataLoader(
        dataset=tokenized["train"],  # type:ignore
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataloader = DataLoader(
        dataset=tokenized["validation"],  # type:ignore
        batch_size=config.data.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader

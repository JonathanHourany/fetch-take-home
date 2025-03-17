from pathlib import Path
from typing import Sequence, Sized

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from llm_tools.datasets import AGNewsDataset
from llm_tools.modules.models import SentenceTransformer
from llm_tools.utils import (
    compute_validation_loss,
    generate_model_name,
    plot_losses,
    plot_tsne,
)

OPENAI_AG_SAMPLES = "https://raw.githubusercontent.com/openai/openai-cookbook/refs/heads/main/examples/data/AG_news_samples.csv"


def main(
    max_seq_length: int = 256,
    num_epochs: int = 12,
    embed_dim: int = 384,
    num_attention_heads: int = 8,
    num_transformer_blocks: int = 9,
    device="cpu",
    base_output_path: Path = Path("./models"),
) -> None:
    torch.set_default_device(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_percent: float = 0.8
    val_percent: float = 1 - train_percent

    train_data = AGNewsDataset(
        df=OPENAI_AG_SAMPLES,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    train_dataset, val_dataset = random_split(train_data, [train_percent, val_percent])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = SentenceTransformer(
        num_attention_heads=num_attention_heads,
        max_seq_length=max_seq_length,
        num_tokens=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_transformer_blocks=num_transformer_blocks,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    loss_record: list[float] = []
    val_loss_record: list[float] = []
    gradient_norms: dict[str, list[float]] = {}

    for epoch in range(num_epochs):
        for batch_number, train_batch in enumerate(train_dataloader, start=1):
            logits = model(
                train_batch["input_ids"], attention_mask=train_batch["attention_mask"]
            )

            loss = criterion(
                logits,
                train_batch["label"],
            )

            loss_record.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_val_loss = compute_validation_loss(
                model=model, data_loader=val_dataloader, criterion=criterion
            )

            val_loss_record.append(average_val_loss)

            print(
                f"Epoch: {epoch} | Batch: {batch_number:>3}/{len(train_dataloader)} "
                f"Train Loss: {loss.item():>7.4f} | Val Loss: {average_val_loss:>7.4f}"
            )

    # Save the model
    model_name = generate_model_name(
        model_name="sentence",
        num_transformer_blocks=num_transformer_blocks,
        model=model,
    )
    model_dir = base_output_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = Path(model_name).with_suffix(".pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": model.hparams,  # Store hyperparameters
        },
        model_dir / model_file,
    )

    # Plot a handful of sentence embeddings
    # Really, this should also be done on a validation/test set also
    train_subset = train_data[:150]
    sentence_embeddings = model(
        train_subset["input_ids"], train_subset["attention_mask"]
    )

    # Plot the embeddings
    # I'm using the row index as the annotation to better analyze data points that
    # appear outside their clusters
    label_map = np.array(["World", "Sports", "Business", "Sci/Tech"])
    labels_text = label_map[train_subset["label"]]
    tsne_fig = plot_tsne(
        sentence_embeddings.detach().numpy(),
        labels=labels_text,
        annotations=range(sentence_embeddings.shape[0]),
    )
    tsne_fig.savefig(model_dir / "sentence_embeddings_tsne.png")

    losses_fig = plot_losses(
        loss_record, val_loss_record, labels=("Training", "Validation")
    )
    losses_fig.savefig(model_dir / "losses.png")


if __name__ == "__main__":
    main(
        max_seq_length=258,
        num_epochs=2,
        embed_dim=384,
        num_attention_heads=4,
        num_transformer_blocks=2,
        device="cpu",
        base_output_path=Path("./models"),
    )

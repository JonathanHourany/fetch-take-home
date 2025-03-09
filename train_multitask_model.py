from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from llm_tools.datasets import TextDataset
from llm_tools.modules.models import MultiTaskModel
from llm_tools.utils import (
    compute_validation_loss,
    generate_model_name,
    plot_losses,
    plot_tsne,
    roundrobin_iters,
    track_gradient_norms,
)

OPENAI_AG_SAMPLES = "https://raw.githubusercontent.com/openai/openai-cookbook/refs/heads/main/examples/data/AG_news_samples.csv"
IMDB_DATA_PATH = Path("data/imdb-dataset.csv.zip")
TASKS = ("topic", "sentiment")


def main(
    max_seq_length: int = 256,
    num_epochs: int = 12,
    embed_dim: int = 384,
    num_attention_heads: int = 8,
    num_transformer_blocks: int = 9,
    batch_size=8,
    device="cpu",
    base_output_path: Path = Path("./models"),
) -> None:
    torch.set_default_device(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_percent: float = 0.8
    val_percent: float = 1 - train_percent

    ag_df = pd.read_csv(
        OPENAI_AG_SAMPLES,
        usecols=["description", "label_int"],
    )
    ag_df["label_int"] = (
        ag_df["label_int"] - 1
    )  # Labels start at 1, need to start at 0 for CrossEntropy

    # Balance the datasets and cast the str labels to float for BCEWithLogitsLoss
    imdb_df = pd.read_csv(IMDB_DATA_PATH, nrows=len(ag_df))
    sentiment_map = {"negative": np.float32(0), "positive": np.float32(1)}
    imdb_df["sentiment"] = imdb_df["sentiment"].map(sentiment_map)

    ag_dataset = TextDataset(
        text=ag_df.description.to_list(),
        labels=ag_df.label_int.to_list(),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    imdb_dataset = TextDataset(
        text=imdb_df.review.to_list(),
        labels=imdb_df.sentiment.values,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    ag_train_dataset, ag_val_dataset = random_split(
        ag_dataset, [train_percent, val_percent]
    )
    imdb_train_dataset, imdb_val_dataset = random_split(
        imdb_dataset, [train_percent, val_percent]
    )

    ag_train_dataloader = DataLoader(
        ag_train_dataset, batch_size=batch_size, shuffle=True
    )
    ag_val_dataloader = DataLoader(ag_val_dataset, batch_size=batch_size, shuffle=False)
    imdb_train_dataloader = DataLoader(
        imdb_train_dataset, batch_size=batch_size, shuffle=True
    )
    imdb_val_dataloader = DataLoader(
        imdb_val_dataset, batch_size=batch_size, shuffle=False
    )

    total_batches = len(ag_train_dataloader) + len(imdb_train_dataloader)

    mt_model = MultiTaskModel(
        num_attention_heads=num_attention_heads,
        max_seq_length=max_seq_length,
        num_tokens=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_topics=4,
        num_sentiments=2,
        dropout=0.1,
    )

    model_name = generate_model_name(
        model_name="multitask",
        num_transformer_blocks=num_transformer_blocks,
        model=mt_model,
    )
    model_dir = base_output_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(mt_model.parameters(), lr=1e-3)
    topics_criterion = torch.nn.CrossEntropyLoss()
    sentiment_criterion = torch.nn.BCEWithLogitsLoss()
    train_loss_record: list[float] = []
    ag_average_val_loss_record: list[float] = []
    imdb_average_val_loss_record: list[float] = []
    gradient_norms: dict[str, list[float]] = {}

    mt_model.train()

    for epoch in range(num_epochs):
        for batch_number, (train_batch, task) in enumerate(
            roundrobin_iters(ag_train_dataloader, imdb_train_dataloader, tasks=TASKS),
            start=1,
        ):
            optimizer.zero_grad()

            logits = mt_model(
                input_ids=train_batch["input_ids"],
                attention_mask=train_batch["attention_mask"],
                task=task,
            )
            if task == "topic":
                loss = topics_criterion(logits, train_batch["label"])
                average_val_loss = compute_validation_loss(
                    mt_model,
                    ag_val_dataloader,
                    topics_criterion,
                    task,
                    squeeze_logits=False,
                )
                ag_average_val_loss_record.append(average_val_loss)

            elif task == "sentiment":
                loss = sentiment_criterion(
                    logits.squeeze(1), train_batch["label"].float()
                )
                average_val_loss = compute_validation_loss(
                    mt_model,
                    imdb_val_dataloader,
                    sentiment_criterion,
                    task,
                    squeeze_logits=True,
                )
                imdb_average_val_loss_record.append(average_val_loss)

            loss.backward()

            train_loss_record.append(loss.item())
            track_gradient_norms(mt_model, gradient_norms)

            optimizer.step()

            print(
                f"Epoch: {epoch} | Batch: {batch_number:>3}/{total_batches} | "
                f"Train Loss: {loss.item():>7.4f} | Val Loss: {average_val_loss:>7.4f}"
            )

    # Save the model
    # TODO: This should be part of the class or a Mixin
    model_file = Path(model_name).with_suffix(".pt")
    torch.save(
        {
            "state_dict": mt_model.state_dict(),
            "hparams": mt_model.hparams,  # Store hyperparameters
        },
        model_dir / model_file,
    )

    ### Plotting and Metrics ###
    mt_model.eval()

    # Plot a handful of sentence embeddings
    sentence_subset = ag_dataset[:250]
    sentence_embeddings = mt_model(
        input_ids=sentence_subset["input_ids"],
        task="topic",
        attention_mask=sentence_subset["attention_mask"],
    )

    # Plot the embeddings
    # I'm using the row index as the annotation to better analyze data points that
    # appear outside their clusters
    label_map = np.array(["World", "Sports", "Business", "Sci/Tech"])
    labels_text = label_map[sentence_subset["label"]]
    tsne_fig = plot_tsne(
        sentence_embeddings.detach().numpy(),
        labels=labels_text,
        annotations=range(sentence_embeddings.shape[0]),
    )
    tsne_fig.savefig(model_dir / "sentence_embeddings_tsne.png")

    losses_fig = plot_losses(
        train_loss_record,
        ag_average_val_loss_record,
        imdb_average_val_loss_record,
        labels=("Training", "AG News Validation", "IMDB Sentiment Validation"),
    )
    losses_fig.savefig(model_dir / "losses.png")

    # Print out and record classification reports
    sentiment_subset = imdb_dataset[:1000]
    sentiment_pred_logits = mt_model(
        input_ids=sentiment_subset["input_ids"],
        attention_mask=sentiment_subset["attention_mask"],
        task="sentiment",
    )
    sentiment_pred_probs = F.sigmoid(sentiment_pred_logits).squeeze()
    sentiment_pred = (sentiment_pred_probs > 0.5).int()
    sentiment_report = classification_report(
        sentiment_subset["label"], sentiment_pred, target_names=("Negative", "Positive")
    )
    print(sentiment_report)
    with open(model_dir / "sentiment-classification-report.txt", "w") as fp:
        fp.write(sentiment_report)


if __name__ == "__main__":
    main(
        max_seq_length=100,
        num_epochs=1,
        embed_dim=384,
        num_attention_heads=4,
        num_transformer_blocks=2,
        batch_size=8,
        device="cpu",
    )

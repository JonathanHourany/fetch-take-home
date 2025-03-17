from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_tools.datasets import AGNewsDataset
from llm_tools.modules.models import TransformerModel
from llm_tools.utils import generate_model_name, plot_losses, plot_tsne

OPENAI_AG_SAMPLES = "https://raw.githubusercontent.com/openai/openai-cookbook/refs/heads/main/examples/data/AG_news_samples.csv"
PLOT_WORDS = [
    "California",
    "Oregon",
    "Arizona",
    "Washington",
    "Texas",
    "Japan",
    "France",
    "England",
    "Europe",
    "America",
    "Asia",
    "Scotland",
    "China",
    "Australia",
    "Germany",
    "Kansas",
    "Idaho",
    "Nebraska",
    "Colorado",
    "Alaska",
    "Hawaii",
    "Italy",
    "Mexico",
    "Turkey",
    "Africa",
]


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

    train_data = AGNewsDataset(
        df=OPENAI_AG_SAMPLES,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)

    model = TransformerModel(
        num_attention_heads=num_attention_heads,
        max_seq_length=max_seq_length,
        num_tokens=tokenizer.vocab_size,
        embed_dim=embed_dim,
        num_transformer_blocks=num_transformer_blocks,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_record: list[float] = []

    # Training loop
    for epoch in range(num_epochs):
        for batch_number, train_batch in enumerate(train_dataloader, start=1):
            input_ids = train_batch["input_ids"]
            attention_masks = train_batch["attention_mask"].type(torch.float16)
            logits = model(input_ids, attention_mask=attention_masks)

            shifted_logits = logits[:, :-1, :]
            shifted_targets = input_ids[:, 1:]

            flatted_logits = shifted_logits.reshape(
                input_ids.shape[0] * (input_ids.shape[1] - 1), tokenizer.vocab_size
            )

            loss = F.cross_entropy(
                flatted_logits,
                shifted_targets.reshape(
                    shifted_targets.shape[0] * shifted_targets.shape[1]
                ),
            )
            loss_record.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Epoch: {epoch} | Batch: {batch_number:>3}/{len(train_dataloader)} "
                f"| Loss: {loss.item():>7.4f}"
            )

    # Save the model
    model_name = generate_model_name(
        model_name="transformer",
        num_transformer_blocks=num_transformer_blocks,
        model=model,
    )
    model_dir = base_output_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = Path(model_name).with_suffix(".pt")
    # TODO: This should be part of the class
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": model.hparams,  # Store hyperparameters
        },
        model_dir / model_file,
    )

    # Plot losses
    loss_fig = plot_losses(loss_record, labels=("Training",))
    loss_fig.savefig(model_dir / "losses.png")

    # Plot the embeddings
    plot_words_ids = [
        tokenizer.encode(word, add_special_tokens=False)[0] for word in PLOT_WORDS
    ]
    word_embeddings = model.token_embeddings(torch.tensor(plot_words_ids))

    tsne_fig = plot_tsne(
        embeddings=word_embeddings.detach().numpy(), annotations=PLOT_WORDS
    )
    tsne_fig.savefig(model_dir / "word_embeddings_tsne.png")


if __name__ == "__main__":
    main(
        max_seq_length=512,
        num_epochs=3,
        embed_dim=384,
        num_attention_heads=4,
        num_transformer_blocks=3,
        device="cpu",
        base_output_path=Path("./models"),
    )

from pathlib import Path
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn

from llm_tools.modules.layers import PositionEmbedding, TransformerBlock
from llm_tools.typing import MULTI_TASK_MODELS
from llm_tools.utils import mean_pooling


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        max_seq_length: int,
        num_tokens: int,
        embed_dim: int,
        dropout: float = 0.1,
        num_transformer_blocks: int = 1,
    ) -> None:
        super().__init__()

        self.hparams = {
            "num_heads": num_attention_heads,
            "max_seq_length": max_seq_length,
            "num_tokens": num_tokens,
            "embed_dim": embed_dim,
            "dropout": dropout,
            "num_transformer_blocks": num_transformer_blocks,
        }
        self.token_embeddings = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embed_dim
        )
        self.position_embeddings = PositionEmbedding(
            max_seq_length=max_seq_length,
            embed_dim=embed_dim,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    num_attention_heads=num_attention_heads,
                    num_tokens=num_tokens,
                    embed_dim=embed_dim,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        self.linear = nn.Linear(embed_dim, num_tokens)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        embeddings = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(embeddings)
        out = self.dropout1(pos_emb)

        for layer in self.transformer:
            out = layer(out, attention_mask)

        self.last_hidden_state = out

        logits = self.linear(out)

        return logits

    @torch.no_grad()
    def generate(
        self, sequence: torch.Tensor, max_length: int = 1, eos_token_id: int = 50256
    ) -> torch.Tensor:
        """
        Given a starting sequence of token IDs, generates the next token until either
        the token predicted is the `eos_token_id` or the generated sequence reaches
        `max_length` size.
        """
        for _ in range(max_length):
            attention_mask = nn.Transformer.generate_square_subsequent_mask(
                len(sequence)
            )
            logits = self.forward(sequence, attention_mask)
            last_sequence = logits[:, -1, :]
            probs = F.softmax(last_sequence, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat((sequence, next_token), dim=1)

            if next_token.item() == eos_token_id:
                break

        return sequence


class SentenceTransformer(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        max_seq_length: int,
        num_tokens: int,
        embed_dim: int,
        num_transformer_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hparams = {
            "num_attention_heads": num_attention_heads,
            "max_seq_length": max_seq_length,
            "num_tokens": num_tokens,
            "embed_dim": embed_dim,
            "num_transformer_blocks": num_transformer_blocks,
            "dropout": dropout,
        }
        self.transformer = TransformerModel(
            num_attention_heads=num_attention_heads,
            max_seq_length=max_seq_length,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            dropout=dropout,
            num_transformer_blocks=num_transformer_blocks,
        )

    def forward(self, input_ids, attention_mask=None):
        _ = self.transformer(input_ids, attention_mask)
        last_hidden_state = self.transformer.last_hidden_state

        return mean_pooling(last_hidden_state, attention_mask, dim=1, keepdim=False)

    @classmethod
    def from_pretrained(cls, model_path: Path | str) -> Self:
        model_state_dict = torch.load(model_path)
        sent_transformer = cls(**model_state_dict["hparams"])
        sent_transformer.transformer.load_state_dict(model_state_dict["state_dict"])

        return sent_transformer


class MultiTaskModel(nn.Module):
    """
    A Multitask Model use for learning topic classification and sentiment analysis.
    """

    def __init__(
        self,
        num_attention_heads: int,
        max_seq_length: int,
        num_tokens: int,
        embed_dim: int,
        num_transformer_blocks: int,
        num_topics: int,
        num_sentiments: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hparams = {
            "num_attention_heads": num_attention_heads,
            "max_seq_length": max_seq_length,
            "num_tokens": num_tokens,
            "embed_dim": embed_dim,
            "num_transformer_blocks": num_transformer_blocks,
            "num_topics": num_topics,
            "num_sentiments": num_sentiments,
            "dropout": dropout,
        }
        self.sent_model = SentenceTransformer(
            num_attention_heads=num_attention_heads,
            max_seq_length=max_seq_length,
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            dropout=dropout,
            num_transformer_blocks=num_transformer_blocks,
        )
        self.topic_clf_head = nn.Linear(embed_dim, num_topics)
        self.sentiment_clf_head = nn.Linear(embed_dim, num_sentiments - 1)

    def forward(
        self,
        input_ids,
        task: MULTI_TASK_MODELS,
        attention_mask=None,
    ):
        sentence_embeddings = self.sent_model(input_ids, attention_mask)

        match task:
            case "topic":
                return self.topic_clf_head(sentence_embeddings)
            case "sentiment":
                return self.sentiment_clf_head(sentence_embeddings)
            case _:
                raise ValueError(
                    f"Unknown task: {task}, must be one of {MULTI_TASK_MODELS}"
                )

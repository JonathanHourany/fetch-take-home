import torch
from torch import nn


class PositionEmbedding(nn.Module):
    """
    Creates a Positional Embedding layer with dimensions (max_seq_length, embed_dim).
    When called, adds the sin(x) for even indices and cos(x) for odd indices, where
    x = position/(k^(2 * index / embed_dim))
    """

    def __init__(self, max_seq_length: int, embed_dim: int, k: int = 10000):
        """
        Creates and pre-computes position vectors for sin and cos with
        dimensions (max_seq_length, embed_dim)
        """
        super().__init__()

        self.position_weights = torch.zeros((max_seq_length, embed_dim))

        pos_range = torch.arange(max_seq_length).reshape(-1, 1)
        i = torch.arange(embed_dim).reshape(1, -1)
        denom = torch.exp(-torch.log(torch.tensor(k)) * (2 * (i // 2) / embed_dim))

        self.position_weights[:, 0::2] = torch.sin(pos_range * denom[:, 0::2])
        self.position_weights[:, 1::2] = torch.cos(pos_range * denom[:, 1::2])

    def forward(self, embeddings):
        """
        Adds positional weight to the passed embeddings. The passed
        embeddings can have a sequence length upto the initialized `max_seq_length`
        but sequences shorter than that are allowed.
        """
        self.position_weights = self.position_weights.to(embeddings.device)
        return embeddings + self.position_weights[: embeddings.shape[1], :]


class TransformerBlock(nn.Module):
    """Standard Pre-LayerNorm Transformer block"""

    def __init__(
        self,
        num_attention_heads: int,
        num_tokens: int,
        embed_dim: int,
        ffwd_expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Sequential(
            nn.Linear(embed_dim, ffwd_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(ffwd_expansion * embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, sequence: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        layer_norm1 = self.layer_norm1(sequence)
        atten_out, _ = self.multihead_attention(
            query=layer_norm1,
            key=layer_norm1,
            value=layer_norm1,
            need_weights=False,
            attn_mask=attention_mask,
            is_causal=True,
        )
        mha_residuals = atten_out + sequence
        layer_norm2 = self.layer_norm2(mha_residuals)
        sequence = self.linear1(layer_norm2) + mha_residuals
        sequence = self.dropout(sequence)

        return sequence

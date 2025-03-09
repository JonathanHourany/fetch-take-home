from __future__ import annotations

from datetime import date
from itertools import cycle
from typing import TYPE_CHECKING, Any, Generator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.manifold import TSNE
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from llm_tools.modules.models import TransformerModel


class TextGenerator:
    def __init__(
        self, model: TransformerModel, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self, sequence: str | Sequence[int], max_length: int = 16, device="cpu"
    ) -> str:
        self.model.eval()
        self.model.to(device)

        if isinstance(sequence, str):
            # A bug in Huggingface Transformers lists the return type of `encode` as list[int] even when
            # # `return_tensors="pt"` is used so we'll have to correct the type here to satisfy mypy
            input_ids: torch.Tensor = self.tokenizer.encode(
                sequence, return_tensors="pt"
            )
            input_ids = input_ids.to(device)

        out = self.model.generate(input_ids, max_length=max_length)

        return self.tokenizer.decode(out.squeeze())


def mean_pooling(in_tensor, mask=None, dim=1, keepdim=False):
    if mask is None:
        return in_tensor.mean(dim=dim, keepdim=keepdim)

    mask = mask.unsqueeze(-1).expand(in_tensor.shape)

    # summed_tensor = in_tensor.sum(dim=dim, keepdim=keepdim)
    masked_summed = (in_tensor * mask).sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim)

    return masked_summed / count


def compute_validation_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    task: Sequence | None = None,
    squeeze_logits: bool = False,
) -> float:
    """
    Computes average loss over batches of a dataset. Useful when running out of GPU memory
    """
    model.eval()
    total_loss: float = 0

    with torch.no_grad():
        for batch in data_loader:
            labels = torch.tensor(batch.pop("label"))

            if task:
                kwargs = {**batch, "task": task}
            else:
                kwargs = batch

            logits = model(**kwargs)

            if squeeze_logits:
                logits = logits.squeeze()

            total_loss += criterion(logits, labels).item()

    model.train()

    average_loss = total_loss / len(data_loader)

    return average_loss


def generate_model_name(
    model_name: str,
    num_transformer_blocks: int,
    model: torch.nn.Module,
    date_str: str | date | None = None,
):
    """
    Generates a model name based on the number of transformer blocks, parameters, and date.
    Name will be in the form `{model_name}-L{num_transformer_blocks}-{params_in_millions}M-{date}`

    Args:
        model_name: String denoting the type of model
        num_transformer_blocks: Int representing the total number of transformer blocks
        date_str: Optional date object or string. If `None`, `date.today()` will be used

    Example:
    >>> generate_model_name(model_name="transformer", num_transformer_blocks=9, params_in_millions=55)
    'transformer-L9-55M-2025-03-08'
    """
    if not date_str:
        date_str = date.today()

    num_params = sum(p.numel() for p in model.parameters())
    params_in_millions = round(num_params / 1_000_000)

    return f"{model_name}-L{num_transformer_blocks}-{params_in_millions}M-{date_str}"


def plot_losses(*losses, labels: Sequence[str]) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    for loss, label in zip(losses, labels):
        ax.plot(loss, label=label)

    ax.set_xlabel("Number of Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()

    return fig


def plot_tsne(
    embeddings: NDArray, labels: Sequence | ArrayLike | None = None, annotations=None
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    tsne = TSNE(n_components=2, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)

    if labels is not None:
        for label in np.unique(labels):
            ax.scatter(
                embeddings_2d[labels == label, 0],
                embeddings_2d[labels == label, 1],
                label=label,
                alpha=0.8,
                edgecolors="k",
            )
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

    if annotations:
        for i, anno in enumerate(annotations):
            ax.annotate(anno, (embeddings_2d[i, 0], embeddings_2d[i, 1]), alpha=0.4)  # type: ignore

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    ax.legend()

    return fig


def track_gradient_norms(model: nn.Module, norms: dict[str, list[float]]) -> None:
    """
    Mutates a `norms` dict by creating a key for each layer in the model and appends
    the gradient norm of the layer to a list when called. If the layer name is missing
    from the dict keys it will be added.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norm = param.grad.norm().item()

            if name not in norms:
                norms[name] = []

            norms[name].append(norm)


def roundrobin_iters(*iterables, tasks: Any) -> Generator[tuple[Any, Any]]:
    """
     Yields items from any number of iterables in a roundrobin style. If `task` is a
     sequence then it is cycled through infinitely until all iterables are exhausted.

     Though, aside from the `task` parameter, this function is generic, it's well suited
     for the job of emitting batches of data from multiple dataloader and their tasks
     for multitask learning.

     Example:
     >>> foo = (0, 2)
     >>> bar = (1, 3)
     >>> tasks = ("foo", "bar")
     >>> for ele, task in roundrobin_iters(foo, bar, tasks=tasks):
    ...:     print(ele, task)
    0 foo
    1 bar
    2 foo
    3 bar
    """
    iterators = [iter(it) for it in iterables]
    num_active = len(iterators)
    tasks = cycle(tasks)

    while num_active:
        for i, it in enumerate(iterators):
            try:
                yield next(it), next(tasks)
            except StopIteration:
                iterators[i] = None
                num_active -= 1
                break
        iterators = [it for it in iterators if it is not None]

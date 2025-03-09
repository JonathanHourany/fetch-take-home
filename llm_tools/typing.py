from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict

if TYPE_CHECKING:
    import torch

MULTI_TASK_MODELS: TypeAlias = Literal["topics", "sentiment"]


class BatchTextEncoding(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: float | list[float]

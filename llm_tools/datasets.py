from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from llm_tools.typing import BatchTextEncoding

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class AGNewsDataset(Dataset):
    def __init__(
        self, df: str | pd.DataFrame, tokenizer: PreTrainedTokenizerBase, max_seq_length
    ):
        self.tokenizer = tokenizer

        if isinstance(df, str):
            df = pd.read_csv(
                df,
                usecols=["description", "label_int"],
            )

        tokens_masks = tokenizer(
            df.description.to_list(),
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        self.news_descriptions = tokens_masks["input_ids"]
        self.attention_masks = tokens_masks["attention_mask"]
        df.label_int = df["label_int"] - 1
        self.labels = df.label_int.to_list()

    def __len__(self):
        return len(self.news_descriptions)

    def __getitem__(self, idx) -> BatchTextEncoding:
        return {
            "input_ids": self.news_descriptions[idx],
            "attention_mask": self.attention_masks[idx].type(torch.float16),
            "label": self.labels[idx],
        }


class TextDataset(Dataset):
    def __init__(
        self,
        text: str | Sequence[str],
        labels: Sequence | NDArray,
        tokenizer,
        max_seq_length: int,
    ):

        batch_encoding = tokenizer(
            text,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

        self.input_ids = batch_encoding["input_ids"]
        self.attention_masks = batch_encoding["attention_mask"].type(torch.float16)
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx) -> BatchTextEncoding:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "label": self.labels[idx],
        }

"""
data.py — Dataset loading, cleaning, and preprocessing pipeline.

Handles:
  - Loading CSV or HuggingFace dataset
  - Text cleaning (lowercasing, emoji handling, URL removal)
  - Tokenization via HuggingFace tokenizers
  - Train / val / test splitting with stratification
  - PyTorch DataLoader creation
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Text cleaning utilities
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise a raw tweet/headline for NLP consumption.

    Steps:
      1. Lowercase
      2. Remove URLs
      3. Remove @mentions
      4. Collapse repeated punctuation (e.g., "!!!!!!" → "!")
      5. Strip leading/trailing whitespace

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"@\w+", "@user", text)                # anonymise mentions
    text = re.sub(r"([!?.])\1+", r"\1", text)            # collapse repeated punct
    text = re.sub(r"\s+", " ", text).strip()             # normalise whitespace
    return text


# ──────────────────────────────────────────────
#  PyTorch Dataset
# ──────────────────────────────────────────────

class SarcasmDataset(Dataset):
    """
    PyTorch Dataset wrapping tokenised sarcasm data.

    Args:
        texts:     List of raw text strings.
        labels:    Corresponding binary labels (0 = not sarcastic, 1 = sarcastic).
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length (truncate/pad to this).
        clean:     Whether to apply text cleaning before tokenisation.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        clean: bool = True,
    ):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Optionally clean text before tokenising
        self.texts = [clean_text(t) if clean else t for t in texts]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          self.labels[idx],
        }


# ──────────────────────────────────────────────
#  Data loading helpers
# ──────────────────────────────────────────────

def load_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load and minimally validate the raw CSV.

    Expects at least two columns: ``text`` and ``label``.
    Drops rows with missing values in either column.

    Args:
        csv_path: Absolute or relative path to the CSV file.

    Returns:
        Cleaned DataFrame with columns [text, label].

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "See data/README.md for download instructions."
        )

    df = pd.read_csv(path)

    # Drop unnamed index columns that Pandas sometimes adds
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed)

    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with null text/label.", dropped)

    df["label"] = df["label"].astype(int)
    logger.info(
        "Loaded %d samples  |  class distribution: %s",
        len(df),
        df["label"].value_counts().to_dict(),
    )
    return df[["text", "label"]]


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.

    Args:
        df:          Full DataFrame with [text, label] columns.
        train_ratio: Fraction of data for training.
        val_ratio:   Fraction of data for validation.
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    X, y = df["text"].values, df["label"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=seed
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), stratify=y_temp, random_state=seed
    )

    logger.info(
        "Split sizes  →  train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return (
        pd.DataFrame({"text": X_train, "label": y_train}),
        pd.DataFrame({"text": X_val,   "label": y_val}),
        pd.DataFrame({"text": X_test,  "label": y_test}),
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap DataFrames in SarcasmDataset and return DataLoaders.

    Args:
        train_df, val_df, test_df: Split DataFrames.
        tokenizer:   HuggingFace tokenizer.
        batch_size:  Mini-batch size.
        max_length:  Sequence length cap.
        num_workers: Parallel data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    def _make(df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = SarcasmDataset(
            texts=df["text"].tolist(),
            labels=df["label"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        _make(train_df, shuffle=True),
        _make(val_df,   shuffle=False),
        _make(test_df,  shuffle=False),
    )

"""
Data Loader Module
==================
Handles loading and merging of ISOT and McIntire fake news datasets.

Author: Hybrid Fake News Detector
"""

import os
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_isot_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Load and merge ISOT Fake News Dataset (Fake.csv + True.csv).

    Args:
        data_dir: Path to the data directory containing Fake.csv and True.csv

    Returns:
        Merged DataFrame with columns: title, text, label (0=Real, 1=Fake)

    Raises:
        FileNotFoundError: If Fake.csv or True.csv are not found
    """
    fake_path = Path(data_dir) / "Fake.csv"
    true_path = Path(data_dir) / "True.csv"

    if not fake_path.exists():
        raise FileNotFoundError(
            f"Fake.csv not found at {fake_path}. "
            "Please download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )
    if not true_path.exists():
        raise FileNotFoundError(
            f"True.csv not found at {true_path}. "
            "Please download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    logger.info("Loading ISOT Fake News Dataset...")

    # Load fake news (label = 1)
    df_fake = pd.read_csv(fake_path)
    df_fake["label"] = 1
    logger.info(f"  Fake articles loaded: {len(df_fake):,}")

    # Load real news (label = 0)
    df_true = pd.read_csv(true_path)
    df_true["label"] = 0
    logger.info(f"  Real articles loaded: {len(df_true):,}")

    # Merge datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)
    logger.info(f"  Total merged: {len(df):,} articles")

    return df


def load_mcintire_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Load McIntire Fake News Dataset for cross-domain evaluation.

    Args:
        data_dir: Path to the data directory containing fake_or_real_news.csv

    Returns:
        DataFrame with columns: title, text, label (0=Real, 1=Fake)

    Raises:
        FileNotFoundError: If fake_or_real_news.csv is not found
    """
    path = Path(data_dir) / "fake_or_real_news.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"fake_or_real_news.csv not found at {path}. "
            "Please download from: https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news"
        )

    logger.info("Loading McIntire Fake News Dataset...")
    df = pd.read_csv(path)
    logger.info(f"  Articles loaded: {len(df):,}")

    # Normalize label column
    if "label" in df.columns:
        # McIntire uses 'FAKE'/'REAL' strings — convert to 0/1
        if df["label"].dtype == object:
            df["label"] = df["label"].map({"FAKE": 1, "REAL": 0, "fake": 1, "real": 0})
    elif "class" in df.columns:
        df["label"] = df["class"].map({"FAKE": 1, "REAL": 0, "fake": 1, "real": 0})
    else:
        raise ValueError("Could not find label column in McIntire dataset. Expected 'label' or 'class'.")

    # Ensure text column exists
    if "text" not in df.columns and "article" in df.columns:
        df.rename(columns={"article": "text"}, inplace=True)

    logger.info(f"  Label distribution:\n{df['label'].value_counts()}")

    return df


def prepare_isot(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full ISOT pipeline: load, validate, and return features + labels.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (DataFrame, Series) — cleaned df and label series
    """
    df = load_isot_dataset(data_dir)

    # Combine title + text for richer features
    df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["full_text"] = df["full_text"].str.strip()

    # Drop rows with empty combined text
    initial_len = len(df)
    df = df[df["full_text"].str.len() > 50].copy()
    logger.info(f"  Removed {initial_len - len(df)} rows with insufficient text")

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df["label"]
    logger.info(
        f"  Final dataset shape: {df.shape}\n"
        f"  Fake (1): {y.sum():,} | Real (0): {(y == 0).sum():,}"
    )

    return df, y


def prepare_mcintire(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full McIntire pipeline: load, validate, and return features + labels.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (DataFrame, Series)
    """
    df = load_mcintire_dataset(data_dir)

    # Combine title + text if title column exists
    if "title" in df.columns:
        df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        df["full_text"] = df["text"].fillna("")

    df["full_text"] = df["full_text"].str.strip()

    # Drop rows with empty text
    df = df[df["full_text"].str.len() > 50].copy()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = df["label"]
    logger.info(
        f"  McIntire Final: {df.shape}\n"
        f"  Fake (1): {y.sum():,} | Real (0): {(y == 0).sum():,}"
    )

    return df, y


def get_data_statistics(df: pd.DataFrame) -> dict:
    """
    Compute basic statistics about the dataset.

    Args:
        df: Input DataFrame with 'full_text' and 'label' columns

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_articles": len(df),
        "fake_count": int((df["label"] == 1).sum()),
        "real_count": int((df["label"] == 0).sum()),
        "fake_percentage": float((df["label"] == 1).mean() * 100),
        "real_percentage": float((df["label"] == 0).mean() * 100),
        "missing_text": int(df["full_text"].isna().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "avg_word_count": float(df["full_text"].str.split().str.len().mean()),
        "median_word_count": float(df["full_text"].str.split().str.len().median()),
        "max_word_count": int(df["full_text"].str.split().str.len().max()),
        "min_word_count": int(df["full_text"].str.split().str.len().min()),
    }
    return stats

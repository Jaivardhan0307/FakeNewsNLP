"""
Feature Engineering Module
===========================
Generates semantic (SBERT) and stylometric features for fake news detection.

Author: Hybrid Fake News Detector
"""

import os
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

import textstat
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.preprocessing import (
    clean_text,
    count_sentences,
    count_words,
    count_characters,
    get_sentences,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 64
RANDOM_SEED = 42

STYLOMETRIC_FEATURE_NAMES = [
    "sentiment_polarity",
    "sentiment_subjectivity",
    "readability_score",
    "exclamation_ratio",
    "capital_ratio",
    "avg_sentence_length",
    "word_count",
    "char_count",
]


# ─────────────────────────────────────────────
# Semantic Features (SBERT)
# ─────────────────────────────────────────────

def load_sbert_model(model_name: str = SBERT_MODEL) -> SentenceTransformer:
    """
    Load or download the Sentence-BERT model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        Loaded SentenceTransformer model
    """
    logger.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_sbert_embeddings(
    texts: List[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = SBERT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate SBERT embeddings for a list of texts.

    Args:
        texts: List of text strings
        model: Pre-loaded SentenceTransformer (loads default if None)
        batch_size: Batch size for encoding
        show_progress: Show tqdm progress bar

    Returns:
        NumPy array of shape (n_samples, embedding_dim)
    """
    if model is None:
        model = load_sbert_model()

    logger.info(f"Generating SBERT embeddings for {len(texts):,} texts...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    logger.info(f"  Embeddings shape: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────
# Stylometric Features
# ─────────────────────────────────────────────

def compute_sentiment(text: str) -> Tuple[float, float]:
    """
    Compute TextBlob sentiment scores.

    Args:
        text: Input text

    Returns:
        Tuple of (polarity, subjectivity)
        polarity: [-1, 1] (negative to positive)
        subjectivity: [0, 1] (objective to subjective)
    """
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return 0.0, 0.0


def compute_readability(text: str) -> float:
    """
    Compute Flesch Reading Ease score.

    Args:
        text: Input text

    Returns:
        Flesch score [0-100], higher = easier to read
    """
    try:
        score = textstat.flesch_reading_ease(text)
        # Clamp to reasonable range
        return float(np.clip(score, -100, 200))
    except Exception:
        return 0.0


def compute_exclamation_ratio(text: str) -> float:
    """
    Compute ratio of exclamation marks to total characters.

    Args:
        text: Input text

    Returns:
        Float ratio [0, 1]
    """
    if not text:
        return 0.0
    exclamation_count = text.count("!")
    return exclamation_count / max(len(text), 1)


def compute_capital_ratio(text: str) -> float:
    """
    Compute ratio of uppercase letters to total letters.

    Args:
        text: Input text (use original, not lowercased)

    Returns:
        Float ratio [0, 1]
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    capitals = sum(1 for c in letters if c.isupper())
    return capitals / len(letters)


def compute_avg_sentence_length(text: str) -> float:
    """
    Compute average number of words per sentence.

    Args:
        text: Input text

    Returns:
        Float average sentence length
    """
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    word_counts = [len(s.split()) for s in sentences]
    return float(np.mean(word_counts))


def _to_clean_float(val, name="feature") -> float:
    if val is None:
        return 0.0
    try:
        if pd.isna(val):
            return 0.0
    except Exception:
        pass
    if isinstance(val, (float, int, np.floating, np.integer)):
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return float(val)
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return 0.0
        return _to_clean_float(val[0], name)
    if isinstance(val, str):
        cleaned_val = val.strip().replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('{', '').replace('}', '')
        if not cleaned_val:
            return 0.0
        try:
            return float(cleaned_val)
        except ValueError as e:
            logger.error(f"Failed to convert feature '{name}' with value '{val}' to float: {e}")
            return 0.0
    try:
        return float(val)
    except Exception as e:
        logger.error(f"Unexpected error converting feature '{name}' with value '{val}' to float: {e}")
        return 0.0

def extract_stylometric_features(text: str, original_text: str = None) -> np.ndarray:
    """
    Extract all stylometric features from a single article.

    Args:
        text: Cleaned text (used for most features)
        original_text: Original text (used for capital ratio)

    Returns:
        NumPy array of shape (8,) with stylometric features
    """
    if original_text is None:
        original_text = text

    polarity, subjectivity = compute_sentiment(text)
    readability = compute_readability(text)
    exclamation_ratio = compute_exclamation_ratio(original_text)
    capital_ratio = compute_capital_ratio(original_text)
    avg_sent_len = compute_avg_sentence_length(text)
    word_count = count_words(text)
    char_count = count_characters(text)

    # Apply clean float conversion
    polarity = _to_clean_float(polarity, "sentiment_polarity")
    subjectivity = _to_clean_float(subjectivity, "sentiment_subjectivity")
    readability = _to_clean_float(readability, "readability_score")
    exclamation_ratio = _to_clean_float(exclamation_ratio, "exclamation_ratio")
    capital_ratio = _to_clean_float(capital_ratio, "capital_ratio")
    avg_sent_len = _to_clean_float(avg_sent_len, "avg_sentence_length")
    word_count = _to_clean_float(word_count, "word_count")
    char_count = _to_clean_float(char_count, "char_count")

    features = np.array([
        polarity,
        subjectivity,
        readability,
        exclamation_ratio,
        capital_ratio,
        avg_sent_len,
        word_count,
        char_count,
    ], dtype=np.float32)

    return features



def generate_stylometric_features(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    original_col: str = "original_text",
) -> np.ndarray:
    """
    Generate stylometric features for all articles in a DataFrame.

    Args:
        df: DataFrame with text columns
        text_col: Column with cleaned text
        original_col: Column with original text

    Returns:
        NumPy array of shape (n_samples, 8)
    """
    logger.info(f"Generating stylometric features for {len(df):,} articles...")

    features_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Stylometric"):
        text = row.get(text_col, "")
        original = row.get(original_col, text)
        features_list.append(extract_stylometric_features(text, original))

    features = np.array(features_list, dtype=np.float32)
    logger.info(f"  Stylometric features shape: {features.shape}")
    return features


def get_stylometric_dataframe(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    original_col: str = "original_text",
) -> pd.DataFrame:
    """
    Return stylometric features as a DataFrame with named columns.

    Args:
        df: Input DataFrame
        text_col: Cleaned text column
        original_col: Original text column

    Returns:
        DataFrame with stylometric feature columns
    """
    features = generate_stylometric_features(df, text_col, original_col)
    return pd.DataFrame(features, columns=STYLOMETRIC_FEATURE_NAMES, index=df.index)


# ─────────────────────────────────────────────
# Feature Fusion
# ─────────────────────────────────────────────

def fuse_features(
    sbert_embeddings: np.ndarray,
    stylometric_features: np.ndarray,
) -> np.ndarray:
    """
    Concatenate SBERT embeddings + stylometric features into hybrid feature matrix.

    Args:
        sbert_embeddings: Shape (n, 384)
        stylometric_features: Shape (n, 8)

    Returns:
        Concatenated array of shape (n, 392)
    """
    assert sbert_embeddings.shape[0] == stylometric_features.shape[0], (
        f"Sample count mismatch: {sbert_embeddings.shape[0]} vs {stylometric_features.shape[0]}"
    )

    hybrid = np.hstack([sbert_embeddings, stylometric_features])
    logger.info(f"  Hybrid feature matrix shape: {hybrid.shape}")
    return hybrid


def get_feature_names(embedding_dim: int = 384) -> List[str]:
    """
    Get feature names for the full hybrid feature matrix.

    Args:
        embedding_dim: Dimensionality of SBERT embeddings

    Returns:
        List of feature name strings
    """
    embedding_names = [f"sbert_{i}" for i in range(embedding_dim)]
    return embedding_names + STYLOMETRIC_FEATURE_NAMES


# ─────────────────────────────────────────────
# Save / Load utilities
# ─────────────────────────────────────────────

def save_features(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str = "outputs/features.npz",
) -> None:
    """
    Save feature matrix and labels to compressed NumPy format.

    Args:
        X: Feature matrix
        y: Label vector
        output_path: Path to output .npz file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X=X, y=y)
    logger.info(f"Features saved to {output_path} (X: {X.shape}, y: {y.shape})")


def load_features(input_path: str = "outputs/features.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load feature matrix and labels from .npz file.

    Args:
        input_path: Path to .npz file

    Returns:
        Tuple of (X, y) arrays
    """
    data = np.load(input_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    logger.info(f"Features loaded from {input_path} (X: {X.shape}, y: {y.shape})")
    return X, y


def save_feature_names(names: List[str], path: str = "outputs/feature_names.pkl") -> None:
    """Save feature names list to pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(names, f)
    logger.info(f"Feature names saved to {path}")


def load_feature_names(path: str = "outputs/feature_names.pkl") -> List[str]:
    """Load feature names from pickle file."""
    with open(path, "rb") as f:
        names = pickle.load(f)
    return names

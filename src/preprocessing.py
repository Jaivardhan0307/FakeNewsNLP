"""
Text Preprocessing Module
=========================
Comprehensive text cleaning pipeline for fake news detection.
Preserves sentence boundaries while removing noise.

Author: Hybrid Fake News Detector
"""

import re
import logging
from typing import Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core cleaning functions
# ─────────────────────────────────────────────

def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_urls(text: str) -> str:
    """Remove URLs (http, https, www patterns)."""
    url_pattern = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE
    )
    return url_pattern.sub(" ", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    html_pattern = re.compile(r"<[^>]+>")
    return html_pattern.sub(" ", text)


def remove_special_characters(text: str) -> str:
    """
    Remove special characters while preserving sentence boundaries.
    Keeps letters, digits, and basic punctuation (., !, ?).
    """
    # Keep alphanumeric, spaces, and sentence-ending punctuation
    clean = re.sub(r"[^a-zA-Z0-9\s\.\!\?]", " ", text)
    return clean


def remove_excessive_whitespace(text: str) -> str:
    """Normalize whitespace — collapse multiple spaces into one."""
    return re.sub(r"\s+", " ", text).strip()


def handle_missing_text(text) -> str:
    """Handle None, NaN, or empty text values."""
    if pd.isna(text) or text is None:
        return ""
    return str(text).strip()


# ─────────────────────────────────────────────
# Full cleaning pipeline
# ─────────────────────────────────────────────

def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Full text cleaning pipeline.

    Steps:
        1. Handle missing values
        2. Convert to lowercase
        3. Remove URLs
        4. Remove HTML tags
        5. Remove special characters
        6. Remove excessive whitespace

    Args:
        text: Raw input text
        lowercase: Whether to convert to lowercase (default: True)

    Returns:
        Cleaned text string
    """
    # Step 1: Handle missing
    text = handle_missing_text(text)
    if not text:
        return ""

    # Step 2: Lowercase
    if lowercase:
        text = to_lowercase(text)

    # Step 3: Remove URLs
    text = remove_urls(text)

    # Step 4: Remove HTML tags
    text = remove_html_tags(text)

    # Step 5: Remove special characters (preserve sentence boundaries)
    text = remove_special_characters(text)

    # Step 6: Normalize whitespace
    text = remove_excessive_whitespace(text)

    return text


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "full_text",
    cleaned_column: str = "cleaned_text"
) -> pd.DataFrame:
    """
    Apply cleaning pipeline to an entire DataFrame column.

    Args:
        df: Input DataFrame
        text_column: Column containing raw text
        cleaned_column: Column to store cleaned text

    Returns:
        DataFrame with new cleaned_text column added
    """
    logger.info(f"Preprocessing text column '{text_column}'...")

    # Preserve original text
    if "original_text" not in df.columns:
        df["original_text"] = df[text_column].copy()

    # Apply cleaning pipeline
    df[cleaned_column] = df[text_column].apply(clean_text)

    # Report empty texts after cleaning
    empty_count = (df[cleaned_column].str.len() == 0).sum()
    if empty_count > 0:
        logger.warning(f"  {empty_count} articles have empty text after cleaning")

    logger.info(f"  Preprocessing complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# Sentence boundary utilities
# ─────────────────────────────────────────────

def get_sentences(text: str) -> List[str]:
    """
    Split text into sentences while preserving boundaries.

    Args:
        text: Cleaned text

    Returns:
        List of sentence strings
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def count_sentences(text: str) -> int:
    """Count number of sentences in text."""
    return len(get_sentences(text))


def count_words(text: str) -> int:
    """Count number of words in text."""
    return len(text.split())


def count_characters(text: str) -> int:
    """Count number of characters (excluding whitespace)."""
    return len(text.replace(" ", ""))


# ─────────────────────────────────────────────
# TF-IDF preprocessing
# ─────────────────────────────────────────────

def clean_for_tfidf(text: str) -> str:
    """
    Aggressive cleaning for TF-IDF vectorization.
    Removes all punctuation and numbers for cleaner token vocabulary.

    Args:
        text: Input text

    Returns:
        Cleaned text suitable for TF-IDF
    """
    text = handle_missing_text(text)
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    # Remove everything except letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    text = remove_excessive_whitespace(text)
    return text

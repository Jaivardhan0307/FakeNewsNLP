"""
Prediction Module
=================
End-to-end inference pipeline for the Streamlit application.
Takes raw article text → returns prediction + explanation.

Author: Hybrid Fake News Detector
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

from src.preprocessing import clean_text, count_words, count_characters, count_sentences
from src.feature_engineering import (
    extract_stylometric_features,
    SBERT_MODEL,
    STYLOMETRIC_FEATURE_NAMES,
    get_feature_names,
)
from src.explainability import get_shap_explainer, explain_single_prediction

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Global model cache (singleton pattern)
# ─────────────────────────────────────────────

_sbert_model = None
_classifier = None
_metadata = None
_shap_explainer = None


def load_all_models(
    model_path: str = "models/fake_news_model.pkl",
    metadata_path: str = "models/model_metadata.pkl",
    sbert_model_name: str = SBERT_MODEL,
) -> Tuple[Any, Any, Any]:
    """
    Load all required models (SBERT + classifier + metadata).
    Uses singleton pattern for efficiency in Streamlit apps.

    Args:
        model_path: Path to trained classifier
        metadata_path: Path to model metadata
        sbert_model_name: SBERT model identifier

    Returns:
        Tuple of (sbert_model, classifier, metadata)
    """
    global _sbert_model, _classifier, _metadata

    try:
        if _sbert_model is None:
            logger.info(f"Loading SBERT model: {sbert_model_name}...")
            _sbert_model = SentenceTransformer(sbert_model_name)
            logger.info("SBERT model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SBERT model: {e}")
        raise RuntimeError(f"SBERT load error: {e}")

    try:
        if _classifier is None:
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Trained model file not found at '{model_path}'. "
                    "Please run the training notebooks/pipeline first."
                )
            logger.info(f"Loading trained classifier from {model_path}...")
            _classifier = joblib.load(model_path)
            logger.info("Classifier loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load trained classifier: {e}")
        raise RuntimeError(f"Classifier load error: {e}")

    try:
        if _metadata is None:
            if Path(metadata_path).exists():
                logger.info(f"Loading model metadata from {metadata_path}...")
                _metadata = joblib.load(metadata_path)
                logger.info("Metadata loaded successfully.")
            else:
                logger.warning(f"Metadata file not found at {metadata_path}. Using empty metadata.")
                _metadata = {}
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        _metadata = {}

    return _sbert_model, _classifier, _metadata


def validate_input_features(x: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Validates and cleans input features before prediction.
    Ensures:
      - Array is float32 or float64.
      - No object types or non-numeric types.
      - Handles NaN, inf, None, and strings.
      - Logs warnings/errors with exact feature names and values.
    """
    logger.info("Starting input feature validation...")
    
    # Check shape and dtypes
    logger.info(f"Input features shape: {x.shape}")
    logger.info(f"Input features dtype: {x.dtype}")
    
    # Ensure 2D shape
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
        
    cleaned_x = np.zeros_like(x, dtype=np.float32)
    
    for r_idx in range(x.shape[0]):
        for c_idx in range(x.shape[1]):
            val = x[r_idx, c_idx]
            feat_name = feature_names[c_idx] if c_idx < len(feature_names) else f"feat_{c_idx}"
            
            cleaned_val = 0.0
            if val is None:
                logger.warning(f"Feature '{feat_name}' at index {c_idx} is None. Coercing to 0.0.")
                cleaned_val = 0.0
            else:
                # Handle pandas/numpy NaN
                try:
                    import pandas as pd
                    if pd.isna(val):
                        logger.warning(f"Feature '{feat_name}' at index {c_idx} is NaN. Coercing to 0.0.")
                        val = np.nan
                except Exception:
                    pass
                
                if isinstance(val, (float, int, np.floating, np.integer)):
                    if np.isnan(val) or np.isinf(val):
                        logger.warning(f"Feature '{feat_name}' at index {c_idx} is non-finite ({val}). Coercing to 0.0.")
                        cleaned_val = 0.0
                    else:
                        cleaned_val = float(val)
                elif isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) == 0:
                        logger.warning(f"Feature '{feat_name}' at index {c_idx} is empty list. Coercing to 0.0.")
                        cleaned_val = 0.0
                    else:
                        logger.warning(f"Feature '{feat_name}' at index {c_idx} is list-like: {val}. Using first element.")
                        first_val = val[0]
                        try:
                            cleaned_val = float(first_val)
                        except Exception:
                            # Strip formatting
                            cleaned_str = str(first_val).strip().replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                            try:
                                cleaned_val = float(cleaned_str)
                            except ValueError:
                                logger.error(f"Failed to convert list element '{first_val}' of '{feat_name}' to float. Coercing to 0.0.")
                                cleaned_val = 0.0
                elif isinstance(val, str):
                    cleaned_str = val.strip().replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('{', '').replace('}', '')
                    if not cleaned_str:
                        logger.warning(f"Feature '{feat_name}' at index {c_idx} is empty string. Coercing to 0.0.")
                        cleaned_val = 0.0
                    else:
                        try:
                            cleaned_val = float(cleaned_str)
                        except ValueError as e:
                            logger.error(f"Conversion failure: Feature '{feat_name}' (idx {c_idx}) has string value '{val}' which could not be converted to float. Error: {e}. Coercing to 0.0.")
                            cleaned_val = 0.0
                else:
                    try:
                        cleaned_val = float(val)
                    except Exception as e:
                        logger.error(f"Unexpected feature type '{type(val)}' for '{feat_name}' (value '{val}'). Error: {e}. Coercing to 0.0.")
                        cleaned_val = 0.0
            
            cleaned_x[r_idx, c_idx] = cleaned_val
            
    # Diagnostics check
    logger.info(f"Cleaned features shape: {cleaned_x.shape}, dtype: {cleaned_x.dtype}")
    logger.info(f"Check for NaN: {np.isnan(cleaned_x).sum()}, Check for Inf: {np.isinf(cleaned_x).sum()}")
    
    # Log sample values
    sample_size = min(5, cleaned_x.shape[1])
    logger.info(f"Diagnostics - First {sample_size} features: {cleaned_x[0, :sample_size].tolist()}")
    logger.info(f"Diagnostics - Last {sample_size} features: {cleaned_x[0, -sample_size:].tolist()}")
    
    return cleaned_x


def predict_article(
    text: str,
    sbert_model: Optional[SentenceTransformer] = None,
    classifier=None,
    metadata: Optional[dict] = None,
    model_path: str = "models/fake_news_model.pkl",
    metadata_path: str = "models/model_metadata.pkl",
) -> Dict[str, Any]:
    """
    Run the complete fake news detection pipeline on a single article.

    Args:
        text: Raw article text
        sbert_model: Pre-loaded SentenceTransformer
        classifier: Pre-loaded classifier
        metadata: Model metadata dict
        model_path: Classifier path
        metadata_path: Metadata path

    Returns:
        Dictionary with full prediction results
    """
    # Validate input
    if not text or len(text.strip()) < 20:
        raise ValueError("Article text is too short. Please provide at least 20 characters.")

    # Load models if not provided
    if sbert_model is None or classifier is None:
        sbert_model, classifier, metadata = load_all_models(model_path, metadata_path)

    # ── Step 1: Preprocess ──────────────────
    cleaned = clean_text(text, lowercase=True)
    original = text.strip()

    # ── Step 2: SBERT Embedding ─────────────
    try:
        embedding = sbert_model.encode(
            [cleaned],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )  # shape: (1, 384)
        if not isinstance(embedding, np.ndarray):
            logger.warning("SBERT embedding is not a numpy array. Converting...")
            embedding = np.array(embedding, dtype=np.float32)
        else:
            embedding = embedding.astype(np.float32)
    except Exception as e:
        logger.error(f"SBERT embedding generation failed: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")

    # ── Step 3: Stylometric Features ────────
    try:
        style_features = extract_stylometric_features(cleaned, original)  # shape: (8,)
        if not isinstance(style_features, np.ndarray):
            style_features = np.array(style_features, dtype=np.float32)
        else:
            style_features = style_features.astype(np.float32)
    except Exception as e:
        logger.error(f"Stylometric feature extraction failed: {e}")
        raise RuntimeError(f"Stylometric extraction failed: {e}")

    # ── Step 4: Fuse Features ───────────────
    try:
        x = np.hstack([embedding[0], style_features]).reshape(1, -1)
    except Exception as e:
        logger.error(f"Feature fusion failed: {e}")
        raise RuntimeError(f"Feature fusion failed: {e}")

    # ── Step 4b: Validate Features ──────────
    feature_names = get_feature_names(embedding_dim=embedding.shape[1])
    try:
        x = validate_input_features(x, feature_names)
    except Exception as e:
        logger.error(f"Feature validation failed: {e}")
        raise RuntimeError(f"Feature validation failed: {e}")

    # ── Step 5: Predict ─────────────────────
    try:
        prediction = int(classifier.predict(x)[0])
        probabilities = classifier.predict_proba(x)[0]
        fake_prob = float(probabilities[1])
        real_prob = float(probabilities[0])
        confidence = max(fake_prob, real_prob)
        label = "FAKE" if prediction == 1 else "REAL"
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise RuntimeError(f"Model prediction failed: {e}")

    # ── Step 6: Compute text statistics ─────
    word_count = count_words(cleaned)
    char_count = count_characters(original)
    sentence_count = count_sentences(original)

    # Build stylometric dict
    style_dict = {
        name: float(val)
        for name, val in zip(STYLOMETRIC_FEATURE_NAMES, style_features)
    }

    result = {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "real_probability": real_prob,
        "sentiment_polarity": style_dict["sentiment_polarity"],
        "sentiment_subjectivity": style_dict["sentiment_subjectivity"],
        "readability_score": style_dict["readability_score"],
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "stylometric_features": style_dict,
        "cleaned_text": cleaned,
        "feature_vector": x,
    }

    return result



def predict_with_shap(
    text: str,
    X_train_sample: Optional[np.ndarray] = None,
    top_n: int = 10,
    model_path: str = "models/fake_news_model.pkl",
    metadata_path: str = "models/model_metadata.pkl",
) -> Dict[str, Any]:
    """
    Predict with SHAP explanation for a single article.

    Args:
        text: Raw article text
        X_train_sample: Background data for SHAP (loads from features.npz if None)
        top_n: Number of top SHAP features to return
        model_path: Classifier path
        metadata_path: Metadata path

    Returns:
        Prediction dict + SHAP explanation
    """
    global _shap_explainer

    # Get base prediction
    sbert_model, classifier, metadata = load_all_models(model_path, metadata_path)
    result = predict_article(text, sbert_model, classifier, metadata)
    x = result["feature_vector"]

    # Build SHAP explainer if needed
    if _shap_explainer is None:
        if X_train_sample is None:
            # Try loading from saved features
            features_path = "outputs/features.npz"
            if Path(features_path).exists():
                data = np.load(features_path, allow_pickle=True)
                X_train_sample = data["X"][:500]  # use subset as background
            else:
                logger.warning("No background data for SHAP. Using feature vector as background.")
                X_train_sample = x

        _shap_explainer = get_shap_explainer(classifier, X_train_sample)

    # Compute explanation
    feature_names = get_feature_names(
        embedding_dim=sbert_model.get_sentence_embedding_dimension()
    )

    try:
        explanation = explain_single_prediction(
            _shap_explainer,
            x[0],
            feature_names,
            top_n=top_n,
        )
        result["shap_explanation"] = explanation
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        result["shap_explanation"] = None

    return result

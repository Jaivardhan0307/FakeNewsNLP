"""
Model Training and Evaluation Utilities
========================================
Handles training, evaluation, saving, and loading of ML models
for the fake news detection pipeline.

Author: Hybrid Fake News Detector
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

RANDOM_SEED = 42


# ─────────────────────────────────────────────
# Data Splitting
# ─────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train-test split.

    Args:
        X: Feature matrix
        y: Labels
        test_size: Fraction for test set
        random_state: Reproducibility seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | "
        f"Fake%: train={y_train.mean():.2%}, test={y_test.mean():.2%}"
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────

def get_logistic_regression(C: float = 1.0) -> LogisticRegression:
    """
    Create Logistic Regression classifier.

    Args:
        C: Regularization parameter

    Returns:
        Configured LogisticRegression model
    """
    return LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
        n_jobs=-1,
    )


def get_xgboost_classifier() -> XGBClassifier:
    """
    Create XGBoost classifier with optimized hyperparameters.

    Returns:
        Configured XGBClassifier model
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )


def get_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Create TF-IDF vectorizer.

    Returns:
        Configured TfidfVectorizer
    """
    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{2,}\b",
    )


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a trained model.

    Args:
        model: Trained sklearn-compatible model
        X_test: Test features
        y_test: True labels
        model_name: Name for logging

    Returns:
        Dictionary with accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    logger.info(
        f"\n{'='*50}\n{model_name} Results:\n"
        f"  Accuracy:  {metrics['accuracy']:.4f}\n"
        f"  Precision: {metrics['precision']:.4f}\n"
        f"  Recall:    {metrics['recall']:.4f}\n"
        f"  F1 Score:  {metrics['f1']:.4f}\n"
        f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n{'='*50}"
    )
    return metrics


def get_classification_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> str:
    """Generate formatted classification report string."""
    y_pred = model.predict(X_test)
    return classification_report(
        y_test, y_pred,
        target_names=["Real News", "Fake News"],
        digits=4,
    )


# ─────────────────────────────────────────────
# Plot utilities
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Best Model",
    output_path: str = "outputs/confusion_matrix.png",
) -> None:
    """
    Plot and save confusion matrix heatmap.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Title for the plot
        output_path: Where to save the figure
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str = "outputs/roc_curve.png",
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        models_dict: {model_name: model} dictionary
        X_test: Test features
        y_test: True labels
        output_path: Where to save
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#2E86AB", "#E84855", "#3BB273", "#F18F01", "#7B2D8B"]

    for (name, model), color in zip(models_dict.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color, linewidth=2.5)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves saved to {output_path}")


def save_model_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: str = "outputs/model_comparison.csv",
) -> pd.DataFrame:
    """
    Create and save model comparison table.

    Args:
        results: {model_name: metrics_dict}
        output_path: CSV output path

    Returns:
        Comparison DataFrame
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update({k.capitalize(): round(v, 4) for k, v in metrics.items()})
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    df.to_csv(output_path, index=False)
    logger.info(f"Model comparison saved to {output_path}")
    return df


def select_best_model(
    results: Dict[str, Dict[str, float]],
    metric: str = "f1",
) -> str:
    """
    Select best model based on a given metric.

    Args:
        results: {model_name: metrics_dict}
        metric: Metric to optimize

    Returns:
        Name of the best-performing model
    """
    best_name = max(results, key=lambda k: results[k].get(metric, 0))
    best_score = results[best_name][metric]
    logger.info(f"Best model: {best_name} ({metric}={best_score:.4f})")
    return best_name


# ─────────────────────────────────────────────
# Model Persistence
# ─────────────────────────────────────────────

def save_model(
    model,
    metadata: dict,
    model_path: str = "models/fake_news_model.pkl",
    metadata_path: str = "models/model_metadata.pkl",
) -> None:
    """
    Save trained model and metadata using joblib.

    Args:
        model: Trained model object
        metadata: Dictionary with model info
        model_path: Path for model file
        metadata_path: Path for metadata file
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    joblib.dump(metadata, metadata_path, compress=3)
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")


def load_model(
    model_path: str = "models/fake_news_model.pkl",
) -> Any:
    """
    Load trained model from disk.

    Args:
        model_path: Path to saved model

    Returns:
        Loaded model object
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def load_metadata(
    metadata_path: str = "models/model_metadata.pkl",
) -> dict:
    """Load model metadata from disk."""
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    return joblib.load(metadata_path)

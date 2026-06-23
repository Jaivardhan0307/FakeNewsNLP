"""
Explainability Module
======================
SHAP-based explainability for the fake news detection model.

Author: Hybrid Fake News Detector
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SHAP Monkey-Patch for XGBoost 2.x/3.x Base Score Incompatibility
# ─────────────────────────────────────────────
try:
    import shap.explainers._tree as shap_tree
    original_decode = shap_tree.decode_ubjson_buffer

    def patched_decode(fd):
        jmodel = original_decode(fd)
        try:
            if "learner" in jmodel and "learner_model_param" in jmodel["learner"]:
                param = jmodel["learner"]["learner_model_param"]
                if "base_score" in param:
                    bs = param["base_score"]
                    if isinstance(bs, str):
                        cleaned = bs.strip().replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                        try:
                            float(cleaned)
                            param["base_score"] = cleaned
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"SHAP UBJSON decode patch error: {e}")
        return jmodel

    shap_tree.decode_ubjson_buffer = patched_decode
    logger.info("Successfully applied SHAP TreeExplainer compatibility patch for XGBoost base_score.")
except Exception as e:
    logger.warning(f"Failed to apply SHAP base_score compatibility patch: {e}")


# ─────────────────────────────────────────────
# SHAP Explainer Setup
# ─────────────────────────────────────────────

def get_shap_explainer(model, X_background: np.ndarray):
    """
    Create the appropriate SHAP explainer based on model type.

    Supports:
        - LinearExplainer for LogisticRegression
        - TreeExplainer for XGBoost / tree-based models

    Args:
        model: Trained sklearn-compatible model
        X_background: Background dataset for kernel approximation

    Returns:
        SHAP explainer object
    """
    model_type = type(model).__name__

    if "LogisticRegression" in model_type:
        logger.info("Using SHAP LinearExplainer for Logistic Regression")
        explainer = shap.LinearExplainer(
            model,
            X_background,
            feature_perturbation="correlation_dependent",
        )
    elif "XGB" in model_type or "GradientBoosting" in model_type or "RandomForest" in model_type:
        logger.info(f"Using SHAP TreeExplainer for {model_type}")
        explainer = shap.TreeExplainer(model)
    else:
        logger.warning(f"Unknown model type {model_type}. Using KernelExplainer (slow).")
        background = shap.sample(X_background, 100)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_shap_values(
    explainer,
    X_test: np.ndarray,
    n_samples: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for a sample of test instances.

    Args:
        explainer: SHAP explainer object
        X_test: Test feature matrix
        n_samples: Number of samples to explain
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (shap_values, X_sample)
    """
    rng = np.random.RandomState(random_state)
    n = min(n_samples, len(X_test))
    sample_idx = rng.choice(len(X_test), size=n, replace=False)
    X_sample = X_test[sample_idx]

    logger.info(f"Computing SHAP values for {n} samples...")
    shap_values = explainer.shap_values(X_sample)

    # Handle multi-class output (take class 1 — Fake)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(f"  SHAP values shape: {np.array(shap_values).shape}")
    return shap_values, X_sample


# ─────────────────────────────────────────────
# SHAP Visualization
# ─────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: np.ndarray,
    feature_names: List[str],
    output_path: str = "outputs/shap_summary.png",
    max_display: int = 20,
) -> None:
    """
    Generate and save SHAP summary (beeswarm) plot.

    Args:
        shap_values: SHAP values array
        X_sample: Feature matrix for samples
        feature_names: Feature name list
        output_path: Where to save the figure
        max_display: Max features to show
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.sca(ax)

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )

    plt.title("SHAP Summary Plot — Feature Impact on Fake News Prediction",
              fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved to {output_path}")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: List[str],
    output_path: str = "outputs/shap_bar.png",
    max_display: int = 20,
) -> None:
    """
    Generate and save SHAP mean absolute bar chart.

    Args:
        shap_values: SHAP values array
        feature_names: Feature name list
        output_path: Where to save the figure
        max_display: Max features to show
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Get top-N features
    top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_names = [feature_names[i] for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_idx)))
    bars = ax.barh(range(len(top_idx))[::-1], top_values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_idx))[::-1])
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("SHAP Feature Importance (Top Features)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.0001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP bar plot saved to {output_path}")


# ─────────────────────────────────────────────
# Single-instance explanation
# ─────────────────────────────────────────────

def explain_single_prediction(
    explainer,
    x: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> Dict:
    """
    Generate SHAP explanation for a single prediction.

    Args:
        explainer: Fitted SHAP explainer
        x: Single feature vector (shape: [n_features])
        feature_names: Feature names
        top_n: Number of top features to return

    Returns:
        Dictionary with:
            - top_positive: Features pushing toward Fake
            - top_negative: Features pushing toward Real
            - shap_values: Full SHAP value array
    """
    x_2d = x.reshape(1, -1)
    sv = explainer.shap_values(x_2d)

    if isinstance(sv, list):
        sv = sv[1]

    sv_flat = np.array(sv).flatten()

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(sv_flat))[::-1]

    top_positive = [
        {"feature": feature_names[i], "shap_value": float(sv_flat[i])}
        for i in sorted_idx[:top_n] if sv_flat[i] > 0
    ]
    top_negative = [
        {"feature": feature_names[i], "shap_value": float(sv_flat[i])}
        for i in sorted_idx[:top_n] if sv_flat[i] < 0
    ]

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "shap_values": sv_flat,
        "all_features": feature_names,
    }

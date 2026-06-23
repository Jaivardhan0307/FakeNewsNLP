"""
Hybrid Fake News Detector — Training Pipeline
==============================================
A script to load data, generate features, and train the classifier.

Usage:
  python run_pipeline.py [--full]
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import prepare_isot
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import (
    generate_sbert_embeddings,
    generate_stylometric_features,
    fuse_features,
    get_feature_names,
    save_features,
    save_feature_names,
)
from src.model_utils import (
    split_data,
    get_xgboost_classifier,
    evaluate_model,
    get_classification_report,
    save_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Fake News Detection training pipeline.")
    parser.add_argument("--full", action="store_true", help="Train on the full dataset instead of a 5,000 sample subset.")
    args = parser.parse_args()

    logger.info("Initializing pipeline...")
    
    # 1. Load data
    try:
        df, y = prepare_isot(data_dir="data")
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        logger.info("Please download the ISOT dataset from Kaggle and place Fake.csv & True.csv in the data/ directory.")
        sys.exit(1)

    # 2. Downsample for fast execution if not --full
    if not args.full:
        logger.info("Downsampling to 5,000 balanced samples for quick local testing (use --full for complete dataset)...")
        # Ensure balanced classes
        df_fake = df[df["label"] == 1].sample(2500, random_state=42)
        df_true = df[df["label"] == 0].sample(2500, random_state=42)
        df = pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
        y = df["label"]

    logger.info(f"Using {len(df):,} samples for training (Fake: {sum(y == 1)}, Real: {sum(y == 0)}).")

    # 3. Preprocess text
    logger.info("Running text cleaning...")
    df = preprocess_dataframe(df, text_column="full_text", cleaned_column="cleaned_text")

    # 4. Generate SBERT semantic embeddings
    logger.info("Generating Sentence-BERT semantic embeddings...")
    sbert_embeddings = generate_sbert_embeddings(
        df["cleaned_text"].tolist(),
        batch_size=64,
        show_progress=True
    )

    # 5. Extract stylometric features
    logger.info("Extracting stylometric features...")
    stylo_features = generate_stylometric_features(
        df,
        text_col="cleaned_text",
        original_col="full_text"
    )

    # 6. Fuse features and save
    logger.info("Fusing features...")
    X_hybrid = fuse_features(sbert_embeddings, stylo_features)
    
    # Save hybrid features
    save_features(X_hybrid, y.values, output_path="outputs/features.npz")
    
    # Save feature names
    feature_names = get_feature_names(embedding_dim=384)
    save_feature_names(feature_names, path="outputs/feature_names.pkl")

    # 7. Split data
    X_train, X_test, y_train, y_test = split_data(X_hybrid, y.values, test_size=0.2, random_state=42)

    # 8. Train Best Model (XGBoost Classifier)
    logger.info("Training XGBoost Classifier...")
    model = get_xgboost_classifier()
    model.fit(X_train, y_train)

    # 9. Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, model_name="Hybrid XGBoost Classifier")
    report = get_classification_report(model, X_test, y_test)
    logger.info(f"\nClassification Report:\n{report}")

    # Write report to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
    logger.info("Saved classification report to outputs/classification_report.txt")

    # 10. Save Model
    metadata = {
        "model_name": "Hybrid XGBoost (SBERT + Stylometric)",
        "metrics": metrics,
        "feature_count": X_hybrid.shape[1],
        "feature_types": "sbert_384 + stylometric_8",
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }
    save_model(model, metadata, model_path="models/fake_news_model.pkl", metadata_path="models/model_metadata.pkl")
    logger.info("Pipeline complete! You can now launch the Streamlit app with: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()

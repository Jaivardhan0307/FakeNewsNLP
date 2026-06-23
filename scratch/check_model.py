import os
import sys
import joblib
import numpy as np

# Change directory to project root
os.chdir(r"e:\FakeNewsNLP")

# Load model and metadata
model = joblib.load("models/fake_news_model.pkl")
metadata = joblib.load("models/model_metadata.pkl")
feature_names = joblib.load("outputs/feature_names.pkl")

print("Model Type:", type(model))
print("Metadata:", metadata)
print("Feature Names Length:", len(feature_names))
print("Last 10 Feature Names:", feature_names[-10:])

# Try predicting a dummy sample
try:
    dummy_x = np.random.rand(1, len(feature_names)).astype(np.float32)
    pred = model.predict(dummy_x)
    prob = model.predict_proba(dummy_x)
    print("Prediction test success!")
    print("Pred:", pred, "Prob:", prob)
except Exception as e:
    print("Prediction test failed:", e)

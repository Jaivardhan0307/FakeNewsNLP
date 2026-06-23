import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import predict_article
import joblib

os.chdir(r"e:\FakeNewsNLP")

text = "This is a test article containing [5E-1] and other text elements to make it longer than twenty characters."
try:
    res = predict_article(text)
    print("Prediction result:", res)
except Exception as e:
    print("Prediction failed:", e)
    import traceback
    traceback.print_exc()

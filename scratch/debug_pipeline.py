import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text
from src.feature_engineering import extract_stylometric_features, STYLOMETRIC_FEATURE_NAMES
import numpy as np

text = "This is a test article! It has some CAPITAL words. And some numbers like 12.3. Also some exclamation marks!!!"
cleaned = clean_text(text)
print("Cleaned text:", cleaned)

style_feats = extract_stylometric_features(cleaned, text)
print("\nStylometric features:")
for name, val in zip(STYLOMETRIC_FEATURE_NAMES, style_feats):
    print(f"{name}: {val} (type: {type(val)})")

print("\nFeatures array dtype:", style_feats.dtype)
print("Features array shape:", style_feats.shape)

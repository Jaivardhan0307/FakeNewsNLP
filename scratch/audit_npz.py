import numpy as np
import os

os.chdir(r"e:\FakeNewsNLP")

features_path = "outputs/features.npz"
if os.path.exists(features_path):
    data = np.load(features_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("y shape:", y.shape)
    print("y dtype:", y.dtype)
    
    # Check if there are any non-numeric values
    non_numeric_count = 0
    for r_idx in range(min(100, X.shape[0])):
        for c_idx in range(X.shape[1]):
            val = X[r_idx, c_idx]
            if isinstance(val, str):
                non_numeric_count += 1
                if non_numeric_count < 10:
                    print(f"Non-numeric at row {r_idx}, col {c_idx}: {val} (type: {type(val)})")
    print("Total non-numeric elements scanned in first 100 rows:", non_numeric_count)
else:
    print("features.npz does not exist.")

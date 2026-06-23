import os
import sys
import joblib
import numpy as np

# Set working directory to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def validate_artifacts():
    print("==================================================")
    print("LOG: AUTOMATED MODEL ARTIFACT VALIDATION")
    print("==================================================")
    
    passed = True
    
    # 1. Check Model File
    model_path = "models/fake_news_model.pkl"
    if not os.path.exists(model_path):
        print("FAIL: Model File: NOT FOUND (models/fake_news_model.pkl)")
        passed = False
    else:
        try:
            model = joblib.load(model_path)
            print(f"SUCCESS: Model File: LOADED SUCCESSFULLY ({type(model).__name__})")
        except Exception as e:
            print(f"FAIL: Model File: FAILED TO LOAD ({e})")
            passed = False
            model = None

    # 2. Check Metadata File
    metadata_path = "models/model_metadata.pkl"
    if not os.path.exists(metadata_path):
        print("FAIL: Metadata File: NOT FOUND (models/model_metadata.pkl)")
        passed = False
    else:
        try:
            meta = joblib.load(metadata_path)
            required_keys = ["model_name", "metrics", "feature_count"]
            missing_keys = [k for k in required_keys if k not in meta]
            if missing_keys:
                print(f"FAIL: Metadata File: LOADED but missing keys: {missing_keys}")
                passed = False
            else:
                print(f"SUCCESS: Metadata File: LOADED & VALIDATED ({meta['model_name']})")
                print(f"   * Feature Count: {meta['feature_count']}")
                print(f"   * F1 Score: {meta['metrics'].get('f1', 'N/A')}")
        except Exception as e:
            print(f"FAIL: Metadata File: FAILED TO LOAD ({e})")
            passed = False
            meta = None

    # 3. Check Feature Names
    feat_names_path = "outputs/feature_names.pkl"
    if not os.path.exists(feat_names_path):
        print("FAIL: Feature Names: NOT FOUND (outputs/feature_names.pkl)")
        passed = False
    else:
        try:
            feat_names = joblib.load(feat_names_path)
            if not isinstance(feat_names, list):
                print(f"FAIL: Feature Names: INVALID TYPE (Expected list, got {type(feat_names).__name__})")
                passed = False
            else:
                print(f"SUCCESS: Feature Names: LOADED & VALIDATED ({len(feat_names)} names)")
        except Exception as e:
            print(f"FAIL: Feature Names: FAILED TO LOAD ({e})")
            passed = False
            feat_names = None

    # 4. Check Dimension Consistency
    if model is not None and meta is not None and feat_names is not None:
        meta_count = meta["feature_count"]
        names_count = len(feat_names)
        
        # Test feature matrix shape with XGBoost
        try:
            dummy_x = np.random.rand(1, meta_count).astype(np.float32)
            model.predict(dummy_x)
            print(f"SUCCESS: Dimension Consistency: Model predicts successfully with {meta_count} features")
        except Exception as e:
            print(f"FAIL: Dimension Consistency: Model prediction failed for size {meta_count}. Error: {e}")
            passed = False
            
        if meta_count != names_count:
            print(f"FAIL: Dimension Consistency: Count mismatch! Metadata specifies {meta_count} but feature_names.pkl has {names_count}")
            passed = False
        else:
            print(f"SUCCESS: Dimension Consistency: Metadata count matches feature_names count ({meta_count})")
            
    # 5. Check Dtype Consistency of Training Features
    features_npz = "outputs/features.npz"
    if os.path.exists(features_npz):
        try:
            data = np.load(features_npz)
            X, y = data["X"], data["y"]
            if not np.issubdtype(X.dtype, np.number):
                print(f"FAIL: Feature Dtype: INVALID training feature matrix dtype ({X.dtype})")
                passed = False
            else:
                print(f"SUCCESS: Feature Dtype: VALID training feature matrix dtype ({X.dtype})")
        except Exception as e:
            print(f"FAIL: Feature Dtype: FAILED TO AUDIT training features ({e})")
            passed = False
    else:
        print("WARN: Feature Dtype: training features.npz not found, skipping validation.")

    print("==================================================")
    if passed:
        print("INFO: ALL CHECKS PASSED: Model artifacts are healthy!")
        sys.exit(0)
    else:
        print("ERROR: VALIDATION FAILED: Please fix identified issues.")
        sys.exit(1)

if __name__ == "__main__":
    validate_artifacts()

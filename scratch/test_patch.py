import os
import sys
import joblib
import numpy as np
import shap

# Apply patch
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
        print("Patch error during decode:", e)
    return jmodel

shap_tree.decode_ubjson_buffer = patched_decode

# Test loading TreeExplainer
os.chdir(r"e:\FakeNewsNLP")
model = joblib.load("models/fake_news_model.pkl")
data = np.load("outputs/features.npz")
X = data["X"][:50]

print("Initializing TreeExplainer with patched decoder...")
explainer = shap.TreeExplainer(model)
print("TreeExplainer created successfully!")

shap_values = explainer.shap_values(X)
print("SHAP values calculated successfully! Shape:", np.array(shap_values).shape)

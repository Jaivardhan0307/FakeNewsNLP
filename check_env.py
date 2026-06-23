import subprocess
import sys
import os

python_exe = r"e:\FakeNewsNLP\envs\fnd_env\python.exe"

checks = [
    "import sys; print('Python:', sys.version[:10])",
    "import sklearn; print('sklearn:', sklearn.__version__)",
    "import xgboost; print('xgboost ok')",
    "import shap; print('shap ok')",
    "import sentence_transformers; print('sentence_transformers ok')",
    "import textblob; print('textblob ok')",
    "import textstat; print('textstat ok')",
    "import kaggle; print('kaggle ok')",
    "import streamlit; print('streamlit ok')",
    "import joblib; print('joblib ok')",
    "import matplotlib; print('matplotlib ok')",
    "import seaborn; print('seaborn ok')",
]

for check in checks:
    try:
        result = subprocess.run(
            [python_exe, "-c", check],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"MISSING: {check.split(';')[0].replace('import ', '')}")
    except Exception as e:
        print(f"ERROR: {e}")

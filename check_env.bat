@echo off
e:\FakeNewsNLP\envs\fnd_env\python.exe --version
e:\FakeNewsNLP\envs\fnd_env\Scripts\pip.exe --version
git --version
e:\FakeNewsNLP\envs\fnd_env\python.exe -c "import kaggle; print('kaggle ok')" 2>nul || echo kaggle NOT installed
e:\FakeNewsNLP\envs\fnd_env\python.exe -c "import sklearn; print('sklearn ok')" 2>nul || echo sklearn NOT installed
e:\FakeNewsNLP\envs\fnd_env\python.exe -c "import xgboost; print('xgboost ok')" 2>nul || echo xgboost NOT installed
e:\FakeNewsNLP\envs\fnd_env\python.exe -c "import shap; print('shap ok')" 2>nul || echo shap NOT installed
e:\FakeNewsNLP\envs\fnd_env\python.exe -c "import sentence_transformers; print('sentence_transformers ok')" 2>nul || echo sentence_transformers NOT installed
e:\FakeNewsNLP\envs\fnd_env\python.exe -c "import textblob; print('textblob ok')" 2>nul || echo textblob NOT installed
e:\FakeNewsNLP\envs\fnd_env\python.exe -c "import textstat; print('textstat ok')" 2>nul || echo textstat NOT installed
echo DONE

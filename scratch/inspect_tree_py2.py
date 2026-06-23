with open(r"E:\FakeNewsNLP\envs\fnd_env\lib\site-packages\shap\explainers\_tree.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
for idx in range(2045, 2085):
    print(f"{idx+1}: {lines[idx]}", end="")

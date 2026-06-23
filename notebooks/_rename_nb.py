import os
import shutil

src = r'e:\FakeNewsNLP\notebooks\02_features.json'
dst = r'e:\FakeNewsNLP\notebooks\02_features.ipynb'

if os.path.exists(src):
    shutil.copy2(src, dst)
    os.remove(src)
    print(f"Renamed: {src} -> {dst}")
    print(f"File size: {os.path.getsize(dst)} bytes")
else:
    print(f"Source file not found: {src}")

# Verify the .ipynb is valid JSON
import json
with open(dst, 'r', encoding='utf-8') as f:
    nb = json.load(f)
print(f"nbformat: {nb.get('nbformat')}")
print(f"Cell count: {len(nb.get('cells', []))}")
print("Notebook JSON is valid!")

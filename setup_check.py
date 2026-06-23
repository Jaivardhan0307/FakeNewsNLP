"""
Script to check what packages are available and install missing ones.
Run with: e:\FakeNewsNLP\envs\fnd_env\python.exe e:\FakeNewsNLP\setup_check.py
"""
import subprocess
import sys

PYTHON = sys.executable
PIP = [PYTHON, "-m", "pip"]

packages_needed = {
    "kaggle": "kaggle",
}

def check_import(pkg):
    result = subprocess.run(
        [PYTHON, "-c", f"import {pkg}; print(f'{pkg} OK')"],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()

print(f"Python: {sys.version}")
print(f"Executable: {PYTHON}")
print()

for import_name, pip_name in packages_needed.items():
    ok, out, err = check_import(import_name)
    if ok:
        print(f"[OK] {import_name}: {out}")
    else:
        print(f"[MISSING] {import_name} - installing {pip_name}...")
        res = subprocess.run(PIP + ["install", pip_name], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  Installed {pip_name} successfully")
        else:
            print(f"  Failed: {res.stderr[:200]}")

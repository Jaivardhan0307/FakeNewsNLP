import os

def search_files(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py') or file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '5E-1' in content or '5e-1' in content:
                            print(f"Found in {file_path}")
                except Exception as e:
                    pass

search_files("e:\\FakeNewsNLP")
print("Search complete.")

import os

def search_files(dir_path):
    target_dirs = ['src', 'app', 'notebooks']
    for t_dir in target_dirs:
        full_t_dir = os.path.join(dir_path, t_dir)
        if not os.path.exists(full_t_dir):
            continue
        for root, dirs, files in os.walk(full_t_dir):
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

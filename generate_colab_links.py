import os

# Replace with your GitHub username and repository name
GITHUB_USERNAME = "AnnieFiB"
GITHUB_REPO = "my_projects"

# Base Colab URL
COLAB_URL = "https://colab.research.google.com/github"

# Root directory where notebooks are stored
NOTEBOOK_DIR = "DataAnalysis/notebooks"

# Generate Colab links for all notebooks
for root, _, files in os.walk(NOTEBOOK_DIR):
    # Skip any .ipynb_checkpoints folders
    if ".ipynb_checkpoints" in root:
        continue
    
    for file in files:
        if file.endswith(".ipynb"):  # Only process Jupyter notebooks
            notebook_path = os.path.join(root, file).replace("\\", "/")  # Convert to UNIX path format
            colab_link = f"{COLAB_URL}/{GITHUB_USERNAME}/{GITHUB_REPO}/blob/main/{notebook_path}"
            print(f"[📗 {file}](<{colab_link}>)")

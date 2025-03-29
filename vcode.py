import os
import json
from pathlib import Path

# Set project root and .vscode folder
###project_root = Path.cwd()
#vscode_dir = project_root / ".vscode"
#vscode_dir.mkdir(exist_ok=True)

# Update `.vscode/settings.json` to auto-activate virtual environment and enhance Streamlit workflow

vscode_dir = Path(".vscode")
settings_path = vscode_dir / "settings.json"

# Load existing settings if they exist
if settings_path.exists():
    with open(settings_path, "r", encoding="utf-8") as f:
        current_settings = json.load(f)
else:
    current_settings = {}

# Update or add new settings
current_settings.update({
    "python.defaultInterpreterPath": "myenv/bin/python",  # Adjust path for Windows if needed
    "python.envFile": "${workspaceFolder}/.env",
    "python.terminal.activateEnvironment": True,
    "python.terminal.activateEnvInCurrentTerminal": True,
    "terminal.integrated.defaultProfile.windows": "Git Bash",
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": ["DataAnalysis/scripts"],
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "remote.containers.enabled": False,
    "files.autoSave": "onFocusChange"
})

# Save updated settings
with open(settings_path, "w", encoding="utf-8") as f:
    json.dump(current_settings, f, indent=4)

settings_path

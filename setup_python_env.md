# Setting Up a Python Virtual Environment

## Creating and Activating a Virtual Environment

```sh
python -m venv myenv  # Create virtual environment
# Verify Creation:
ls -la venv/bin
```

Activate the virtual environment:

```sh
source myenv/bin/activate  # On macOS/Linux
```

## Installing Dependencies

```sh
pip install -r requirements.txt
pip list  # Verify installed packages
```

## Creating a Kernel for Your Virtual Environment

```sh
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
```

### Verify Kernel Installation

```sh
jupyter kernelspec list
```

## Selecting the Virtual Environment Interpreter

Open the Command Palette (`Ctrl + Shift + P`) and:

- Select the interpreter from the virtual environment.

## Creating a `.gitignore` File

Ensure your virtual environment is ignored in version control:

```
myenv/
```

## Creating and Running a Test Script

Create `test.py`:

```python
import requests

response = requests.get('https://api.github.com')
print(response.status_code)
```

Run the script:

```sh
python test.py
```

## Deactivating the Virtual Environment

Use the following command when finished:

```sh
deactivate
```

## Using Pylance in VS Code

### Setting Up Pylance

1. Open the Command Palette (`Ctrl + Shift + P`).
2. Select **Python: Select Interpreter** and choose the interpreter from your virtual environment.
3. Open the Command Palette (`Ctrl + Shift + P`) and select **Preferences: Open Settings (JSON)**.
4. Add or update the following configuration:

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.typeCheckingMode": "basic",  // Set to "basic" or "strict" as needed.
    "python.analysis.autoImportCompletions": true,
    "python.analysis.extraPaths": ["./src"]  // Adjust this based on your project structure.
}
```

## Updating and Pushing Code to Git

```sh
git status
git add .
git commit -m "initial commit"
git push
```

### Removing Git Lock (if needed)

```sh
rm -f .git/index.lock
```

## Summary

- **Ignore the virtual environment** in `.gitignore`
- **Store dependencies** in `requirements.txt`
- **Set the Python interpreter** in `.vscode/settings.json`
- **Document setup steps** in `README.md`

#!/bin/bash
# Cleanup Script for Data Projects (Bash Version)

# Exit on errors and unset variables
set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Initialize counters
dir_count=0
file_count=0

echo -e "${GREEN}############################################"
echo "#          Project Cleanup Utility         #"
echo -e "############################################${NC}"
echo

clean_directories() {
    echo "[1/6] Cleaning cache directories..."
    while IFS= read -r -d $'\0' dir; do
        rm -rf "$dir" && ((dir_count++)) || {
            echo -e "${RED}Failed to remove: $dir${NC}"
        }
    done < <(find . -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" -o -name ".pytest_cache" \) -print0)
}

clean_python_files() {
    echo "[2/6] Removing Python cache files..."
    while IFS= read -r -d $'\0' file; do
        rm -f "$file" && ((file_count++)) || {
            echo -e "${RED}Failed to delete: $file${NC}"
        }
    done < <(find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -print0)
}

clean_temp_files() {
    echo "[3/6] Cleaning temporary files..."
    find . -type f \( -name "*.tmp" -o -name "*.temp" -o -name "~$*" \) -delete -exec echo "Deleted: {}" \; | wc -l | read -r count
    file_count=$((file_count + count))
}

clean_jupyter_checkpoints() {
    echo "[4/6] Clearing Jupyter checkpoints..."
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + -exec echo "Removed: {}" \; | wc -l | read -r count
    dir_count=$((dir_count + count))
}

clean_pip_cache() {
    echo "[5/6] Purging pip cache..."
    if command -v pip &> /dev/null; then
        pip cache purge || echo -e "${RED}Pip cache purge failed${NC}"
    else
        echo -e "${RED}Pip not found - skipping cache purge${NC}"
    fi
}

main() {
    clean_directories
    clean_python_files
    clean_temp_files
    clean_jupyter_checkpoints
    clean_pip_cache
    
    echo "[6/6] Finalizing cleanup..."
    echo
    echo -e "${GREEN}#################################"
    echo "#          Summary              #"
    echo -e "#################################${NC}"
    echo -e "Removed directories: ${dir_count}"
    echo -e "Deleted files: ${file_count}"
    echo
    echo -e "${GREEN}Cleanup operation completed successfully${NC}"
}

main
#!/bin/bash
# Simple Project Cleanup Script

echo "Starting cleanup..."

# Remove Python cache directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Delete Python compiled files
find . -type f -name "*.py[co]" -delete

# Clear Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# Remove temporary files
find . -type f \( -name "*.tmp" -o -name "*.temp" \) -delete 2>/dev/null

# Clean pip cache if available
command -v pip >/dev/null && pip cache purge

echo "Cleanup complete!"
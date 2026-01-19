#!/bin/bash
# Training script for VAE on CelebA
# Reference: AAIT Assignment 3, Task 1

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

echo "=============================================="
echo "VAE CelebA Training"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for package management"

    # Sync dependencies
    echo "Syncing dependencies..."
    uv sync

    # Run training with uv
    echo ""
    echo "Starting training..."
    uv run python src/train.py --config configs/config.yaml "$@"
else
    echo "uv not found, using pip/python directly"

    # Activate virtual environment if it exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Run training
    echo ""
    echo "Starting training..."
    python src/train.py --config configs/config.yaml "$@"
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="

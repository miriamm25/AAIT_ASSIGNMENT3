#!/bin/bash
# Train attribute classifier for Task 2.2 - Label Guidance
# Reference: AAIT_Assignment_3.pdf Task 2.2

# Change to project directory
cd "$(dirname "$0")/.."

# Default configuration
CONFIG="configs/editing_config.yaml"
EPOCHS=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Training Attribute Classifier for Task 2.2"
echo "=============================================="
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo ""

# Run training
python scripts/train_classifier.py \
    --config "$CONFIG" \
    --epochs "$EPOCHS"

echo ""
echo "Training complete!"

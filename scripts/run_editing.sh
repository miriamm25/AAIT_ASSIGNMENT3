#!/bin/bash
# Run Task 2: Editing Pictures with Latent Space Manipulation
# Reference: AAIT_Assignment_3.pdf Task 2

# Change to project directory
cd "$(dirname "$0")/.."

# Default configuration
CONFIG="configs/editing_config.yaml"
TASK="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --feature_amplification)
            TASK="feature_amplification"
            shift
            ;;
        --label_guidance)
            TASK="label_guidance"
            shift
            ;;
        --identity_transfer)
            TASK="identity_transfer"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config PATH         Path to config file (default: configs/editing_config.yaml)"
            echo "  --task TASK           Task to run: all, feature_amplification, label_guidance, identity_transfer"
            echo "  --feature_amplification  Run only feature amplification"
            echo "  --label_guidance        Run only label guidance"
            echo "  --identity_transfer     Run only identity transfer"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Task 2: Editing Pictures with Latent Space"
echo "=============================================="
echo "Config: $CONFIG"
echo "Task: $TASK"
echo ""

# Check if classifier exists for label guidance
if [[ "$TASK" == "all" || "$TASK" == "label_guidance" ]]; then
    CLASSIFIER_PATH=$(grep "checkpoint_path" "$CONFIG" | grep "classifier" | awk '{print $2}')
    if [[ ! -f "$CLASSIFIER_PATH" && "$CLASSIFIER_PATH" != "" ]]; then
        echo "Warning: Classifier checkpoint not found at $CLASSIFIER_PATH"
        echo "Please run train_classifier.sh first for label guidance."
        echo ""
        if [[ "$TASK" == "label_guidance" ]]; then
            exit 1
        fi
    fi
fi

# Run editing tasks
python scripts/run_editing.py \
    --config "$CONFIG" \
    --task "$TASK"

echo ""
echo "Task 2 complete!"

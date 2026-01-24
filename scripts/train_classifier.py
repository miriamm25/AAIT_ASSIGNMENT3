"""
Component: Attribute Classifier Training Script
Reference: AAIT_Assignment_3.pdf Task 2.2 - Label Guidance

Purpose:
    Train a multi-label attribute classifier on CelebA.
    This classifier is used for label guidance in Task 2.2.

Usage:
    python scripts/train_classifier.py --config configs/editing_config.yaml
    python scripts/train_classifier.py --config configs/editing_config.yaml --epochs 20

Teacher's BIG HINT:
    "Train classifier f(x) = label"
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

from data.celeba_with_attributes import (
    get_celeba_attribute_dataloaders,
    CELEBA_ATTRIBUTES,
)
from editing.classifier import (
    AttributeClassifier,
    train_classifier,
    save_classifier,
    evaluate_per_attribute,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train attribute classifier for Task 2.2 - Label Guidance"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/editing_config.yaml",
        help="Path to editing config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate existing checkpoint",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    cl_config = config["classifier"]
    print("Loaded configuration from", args.config)

    # Apply command line overrides
    if args.epochs:
        cl_config["epochs"] = args.epochs
    if args.batch_size:
        cl_config["batch_size"] = args.batch_size
    if args.lr:
        cl_config["lr"] = args.lr

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Print configuration
    print("\nClassifier Configuration:")
    print(f"  Backbone: {cl_config['backbone']}")
    print(f"  Pretrained: {cl_config['pretrained']}")
    print(f"  Number of attributes: {cl_config['n_attributes']}")
    print(f"  Batch size: {cl_config['batch_size']}")
    print(f"  Learning rate: {cl_config['lr']}")
    print(f"  Epochs: {cl_config['epochs']}")
    print(f"  Checkpoint path: {cl_config['checkpoint_path']}")

    # Create output directory
    checkpoint_path = Path(cl_config["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print("\nLoading CelebA dataset with attributes...")
    train_loader, val_loader, test_loader = get_celeba_attribute_dataloaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=cl_config["batch_size"],
        num_workers=cl_config["num_workers"],
        download=config["data"]["download"],
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    if args.evaluate_only:
        # Load and evaluate existing checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        print(f"\nLoading classifier from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = AttributeClassifier(
            n_attributes=cl_config["n_attributes"],
            pretrained=False,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    else:
        # Create model
        print("\nCreating AttributeClassifier...")
        model = AttributeClassifier(
            n_attributes=cl_config["n_attributes"],
            pretrained=cl_config["pretrained"],
            freeze_backbone=False,
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Train
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        history = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cl_config["epochs"],
            lr=cl_config["lr"],
            device=device,
            save_path=checkpoint_path,
            log_interval=100,
        )

        # Save final model
        final_path = checkpoint_path.parent / "classifier_final.pt"
        save_classifier(model, final_path, history, cl_config["epochs"])
        print(f"\nSaved final model to {final_path}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    per_attr_results = evaluate_per_attribute(
        model=model,
        dataloader=test_loader,
        device=device,
        attribute_names=CELEBA_ATTRIBUTES,
    )

    # Print results for target attributes
    target_attrs = config["label_guidance"]["target_attributes"]
    print("\nResults for target attributes:")
    print("-" * 50)
    for attr_info in target_attrs:
        attr_idx = attr_info["index"]
        attr_name = attr_info["name"]
        results = per_attr_results[attr_idx]
        print(f"  {attr_name} (idx {attr_idx}):")
        print(f"    Accuracy:  {results['accuracy']:.4f}")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Recall:    {results['recall']:.4f}")

    # Print overall statistics
    print("\nOverall attribute statistics:")
    print("-" * 50)
    accuracies = [r["accuracy"] for r in per_attr_results.values()]
    print(f"  Mean accuracy: {sum(accuracies) / len(accuracies):.4f}")
    print(f"  Min accuracy:  {min(accuracies):.4f}")
    print(f"  Max accuracy:  {max(accuracies):.4f}")

    # Save detailed results
    import json
    results_path = checkpoint_path.parent / "classifier_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(per_attr_results, f, indent=2)
    print(f"\nSaved detailed evaluation results to {results_path}")

    # Print top and bottom 5 attributes by accuracy
    sorted_attrs = sorted(per_attr_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)

    print("\nTop 5 attributes by accuracy:")
    for idx, results in sorted_attrs[:5]:
        print(f"  {results['name']}: {results['accuracy']:.4f}")

    print("\nBottom 5 attributes by accuracy:")
    for idx, results in sorted_attrs[-5:]:
        print(f"  {results['name']}: {results['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("Classifier training/evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Component: Main Editing Script for Task 2
Reference: AAIT_Assignment_3.pdf Task 2 - Editing Pictures with Latent Space Manipulation

Purpose:
    Run all three editing tasks:
    1. Feature Amplification (Task 2.1)
    2. Label Guidance (Task 2.2)
    3. Identity Transfer (Task 2.3)

Usage:
    python scripts/run_editing.py --config configs/editing_config.yaml
    python scripts/run_editing.py --config configs/editing_config.yaml --task feature_amplification
    python scripts/run_editing.py --config configs/editing_config.yaml --task label_guidance
    python scripts/run_editing.py --config configs/editing_config.yaml --task identity_transfer

Prerequisites:
    - Trained VAE from Task 1 (checkpoint required)
    - For Task 2.2: Trained attribute classifier (run train_classifier.py first)
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src directory to path
script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

from models.vae import VAE
from data.celeba import get_celeba_dataloaders
from data.celeba_with_attributes import (
    get_celeba_attribute_dataloaders,
    CelebAWithAttributes,
    CELEBA_ATTRIBUTES,
)
from editing.feature_amplification import (
    find_meaningful_dimensions,
    create_amplification_grid,
    visualize_amplification_all_samples,
)
from editing.classifier import load_classifier
from editing.label_guidance import (
    create_label_guidance_grid,
    visualize_label_guidance_detailed,
)
from editing.identity_transfer import (
    compute_anchor_latent,
    fit_latent_pca,
    create_identity_grid,
    visualize_identity_transfer_all,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_vae(config: dict, device: torch.device) -> VAE:
    """Load trained VAE from checkpoint."""
    # Create model
    vae = VAE(
        in_channels=config["vae"]["in_channels"],
        base_channels=config["vae"]["base_channels"],
        latent_dim=config["vae"]["latent_dim"],
        blocks_per_level=config["vae"]["blocks_per_level"],
        image_size=config["vae"]["image_size"],
        predict_variance=config["vae"]["predict_variance"],
    ).to(device)

    # Load checkpoint
    checkpoint_path = config["vae"]["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"VAE checkpoint not found at {checkpoint_path}. "
            "Please train the VAE first (Task 1)."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    print(f"Loaded VAE from {checkpoint_path}")
    return vae


def get_sample_images(dataloader, n_samples: int, device: torch.device) -> torch.Tensor:
    """Get sample images from dataloader."""
    images = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        images.append(batch)
        if sum(x.shape[0] for x in images) >= n_samples:
            break

    images = torch.cat(images, dim=0)[:n_samples]
    return images.to(device)


def run_feature_amplification(config: dict, vae: VAE, device: torch.device):
    """
    Run Task 2.1: Feature Amplification.

    Reference: AAIT_Assignment_3.pdf Task 2.1
        "4 meaningful components × 10 alpha values × 8 samples"
    """
    print("\n" + "=" * 60)
    print("Task 2.1: Feature Amplification")
    print("=" * 60)

    fa_config = config["feature_amplification"]
    output_dir = Path(fa_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloader
    train_loader, _, _ = get_celeba_dataloaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=32,
        num_workers=4,
        download=config["data"]["download"],
    )

    # Find meaningful dimensions
    print("\nFinding meaningful latent dimensions...")
    dimensions, variance = find_meaningful_dimensions(
        vae=vae,
        dataloader=train_loader,
        n_dims=fa_config["n_dimensions"],
        n_images=fa_config["n_images_for_variance"],
        device=device,
    )
    print(f"Selected dimensions: {dimensions}")
    print(f"Variance of selected dimensions: {variance[dimensions].tolist()}")

    # Save dimension info
    import json
    dim_info = {
        "dimensions": dimensions,
        "variances": variance[dimensions].tolist(),
        "all_variances": variance.tolist(),
    }
    with open(output_dir / "dimension_info.json", "w") as f:
        json.dump(dim_info, f, indent=2)

    # Get sample images
    print(f"\nGetting {fa_config['n_samples']} sample images...")
    sample_images = get_sample_images(train_loader, fa_config["n_samples"], device)

    # Generate alpha values
    alpha_min, alpha_max = fa_config["alpha_range"]
    alphas = np.linspace(alpha_min, alpha_max, fa_config["n_alphas"]).tolist()
    print(f"Alpha values: {alphas}")

    # Create amplification grid
    print("\nGenerating amplification grid...")
    grids = create_amplification_grid(
        vae=vae,
        images=sample_images,
        dimensions=dimensions,
        alphas=alphas,
        device=device,
    )

    # Visualize and save
    print("\nSaving visualizations...")
    dimension_names = [f"Dim_{d}" for d in dimensions]
    visualize_amplification_all_samples(
        grids=grids,
        dimensions=dimensions,
        alphas=alphas,
        output_dir=output_dir,
        dimension_names=dimension_names,
    )

    print(f"\nFeature amplification complete! Results saved to {output_dir}")


def run_label_guidance(config: dict, vae: VAE, device: torch.device):
    """
    Run Task 2.2: Label Guidance.

    Reference: AAIT_Assignment_3.pdf Task 2.2
        "4 different label changes × 8 samples"
    """
    print("\n" + "=" * 60)
    print("Task 2.2: Label Guidance")
    print("=" * 60)

    lg_config = config["label_guidance"]
    cl_config = config["classifier"]
    output_dir = Path(lg_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classifier
    classifier_path = cl_config["checkpoint_path"]
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            f"Classifier checkpoint not found at {classifier_path}. "
            "Please run train_classifier.py first."
        )

    print(f"\nLoading classifier from {classifier_path}...")
    classifier = load_classifier(classifier_path, device)

    # Get dataloader
    train_loader, _, _ = get_celeba_dataloaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=32,
        num_workers=4,
        download=config["data"]["download"],
    )

    # Get sample images
    print(f"\nGetting {lg_config['n_samples']} sample images...")
    sample_images = get_sample_images(train_loader, lg_config["n_samples"], device)

    # Create label guidance grid
    print("\nOptimizing latents for target attributes...")
    target_attributes = lg_config["target_attributes"]
    print(f"Target attributes: {[a['name'] for a in target_attributes]}")

    results = create_label_guidance_grid(
        vae=vae,
        classifier=classifier,
        images=sample_images,
        target_attributes=target_attributes,
        n_steps=lg_config["optimization"]["n_steps"],
        lr=lg_config["optimization"]["lr"],
        regularization_weight=lg_config["optimization"]["regularization_weight"],
        device=device,
    )

    # Visualize and save
    print("\nSaving visualizations...")
    attribute_names = [a["name"] for a in target_attributes]
    visualize_label_guidance_detailed(
        results=results,
        attribute_names=attribute_names,
        output_dir=output_dir,
    )

    print(f"\nLabel guidance complete! Results saved to {output_dir}")


def run_identity_transfer(config: dict, vae: VAE, device: torch.device):
    """
    Run Task 2.3: Identity Transfer.

    Reference: AAIT_Assignment_3.pdf Task 2.3
        "3 anchor people × 8 different subjects"
    """
    print("\n" + "=" * 60)
    print("Task 2.3: Identity Transfer")
    print("=" * 60)

    it_config = config["identity_transfer"]
    output_dir = Path(it_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloader with identity labels
    print("\nLoading CelebA dataset with identity labels...")
    train_loader, _, _ = get_celeba_attribute_dataloaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=32,
        num_workers=4,
        download=config["data"]["download"],
        return_identity=True,
    )

    # Get sample subject images (different from anchors)
    print(f"\nGetting {it_config['n_subjects']} subject images...")
    subject_images = get_sample_images(train_loader, it_config["n_subjects"] + 50, device)
    # Use later images as subjects (first ones might be used as anchors)
    subject_images = subject_images[50:50 + it_config["n_subjects"]]

    # Select anchor identities
    # For simplicity, we'll use the first few images as anchor representatives
    # In practice, you'd want to select specific identities with multiple images
    print(f"\nSelecting {it_config['n_anchors']} anchor identities...")

    anchor_latents = []
    anchor_images_list = []

    # Get first n_anchors*n_images_per_anchor images for anchors
    anchor_source_images = get_sample_images(
        train_loader,
        it_config["n_anchors"] * it_config["n_images_per_anchor"],
        device
    )

    for anchor_idx in range(it_config["n_anchors"]):
        start_idx = anchor_idx * it_config["n_images_per_anchor"]
        end_idx = start_idx + it_config["n_images_per_anchor"]
        anchor_imgs = anchor_source_images[start_idx:end_idx]

        # Compute anchor latent
        anchor_latent = compute_anchor_latent(
            vae=vae,
            anchor_images=[anchor_imgs[i:i+1] for i in range(anchor_imgs.shape[0])],
            device=device,
        )
        anchor_latents.append(anchor_latent)
        anchor_images_list.append(anchor_imgs[0:1])  # Representative image

        print(f"  Anchor {anchor_idx + 1}: latent computed from {anchor_imgs.shape[0]} images")

    # Fit PCA if using PCA method
    method = it_config["method"]
    pca = None

    if method == "pca":
        print("\nFitting PCA on latent space...")
        pca, _ = fit_latent_pca(
            vae=vae,
            dataloader=train_loader,
            n_latents=it_config["pca"]["n_latents_for_pca"],
            device=device,
        )

    # Create identity transfer grid
    print(f"\nPerforming identity transfer using {method} method...")
    results = create_identity_grid(
        vae=vae,
        subject_images=subject_images,
        anchor_latents=anchor_latents,
        method=method,
        pca=pca,
        n_identity_components=it_config["pca"]["n_identity_components"] if method == "pca" else None,
        alphas=it_config["alphas"] if method == "simple" else None,
        device=device,
    )

    # Visualize and save
    print("\nSaving visualizations...")
    visualize_identity_transfer_all(
        results=results,
        n_anchors=it_config["n_anchors"],
        output_dir=output_dir,
        method=method,
        alphas=it_config["alphas"] if method == "simple" else None,
        anchor_images=[img.cpu() for img in anchor_images_list],
    )

    # Save anchor representative images
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, it_config["n_anchors"], figsize=(it_config["n_anchors"] * 2, 2))
    if it_config["n_anchors"] == 1:
        axes = [axes]

    for i, anchor_img in enumerate(anchor_images_list):
        img = anchor_img[0].cpu().permute(1, 2, 0).numpy()
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"Anchor {i + 1}")

    plt.suptitle("Anchor Identities")
    plt.tight_layout()
    plt.savefig(output_dir / "anchor_identities.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nIdentity transfer complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Task 2: Editing Pictures with Latent Space Manipulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/editing_config.yaml",
        help="Path to editing config file",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["all", "feature_amplification", "label_guidance", "identity_transfer"],
        default="all",
        help="Which task to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to config value.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print("Loaded configuration from", args.config)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(config.get("seed", 42))
    if device.type == "cuda":
        torch.cuda.manual_seed(config.get("seed", 42))

    # Load VAE
    print("\nLoading trained VAE...")
    vae = load_vae(config, device)

    # Run tasks
    if args.task == "all" or args.task == "feature_amplification":
        run_feature_amplification(config, vae, device)

    if args.task == "all" or args.task == "label_guidance":
        run_label_guidance(config, vae, device)

    if args.task == "all" or args.task == "identity_transfer":
        run_identity_transfer(config, vae, device)

    print("\n" + "=" * 60)
    print("Task 2 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Component: Improved Identity Transfer Script
Reference: Task 2.3 Improvement Plan

Purpose:
    Fix the identity transfer bug where random people were averaged
    instead of multiple images of the SAME person.

The Problem (Original Implementation):
    Anchor "essence" = average of [Random Person 1, Random Person 2, ...]
    Result: Generic average face, not a specific identity

The Fix (This Implementation):
    Anchor "essence" = average of [Person A pose 1, Person A pose 2, ...]
    Result: Person A's unique identity features

From the assignment (AAIT_Assignment_3.pdf Task 2.3):
    "Encode multiple images of each anchor -> average latent = essence"
    This REQUIRES multiple images of the SAME person.

Usage:
    python scripts/experiments/run_identity_transfer_improved.py
    python scripts/experiments/run_identity_transfer_improved.py --explore  # Find identities
    python scripts/experiments/run_identity_transfer_improved.py --run      # Run transfer
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src directory to path
script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

from models.vae import VAE
from data.celeba_with_attributes import (
    get_celeba_attribute_dataloaders,
    CelebAWithAttributes,
)
from editing.identity_transfer import (
    transfer_identity_simple,
    transfer_identity_pca,
    fit_latent_pca,
)
from editing.experiments.identity_utils import (
    find_rich_identities,
    get_images_by_identity_id,
    compute_true_anchor_latent,
    visualize_identity_samples,
    explore_identities,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "vae": {
        "checkpoint_path": "outputs/checkpoints/best_model.pt",
        "in_channels": 3,
        "base_channels": 64,
        "latent_dim": 256,
        "blocks_per_level": 2,
        "image_size": 64,
        "predict_variance": False,
    },
    "data": {
        "root": "./data",
        "image_size": 64,
        "batch_size": 32,
        "num_workers": 4,
        "download": False,
    },
    "identity_transfer": {
        "output_dir": "outputs/experiments/identity_transfer",
        "n_anchors": 3,
        "n_subjects": 8,
        "n_images_per_anchor": 10,  # Average this many images per anchor
        "min_images_per_identity": 10,  # Minimum images required for anchor
        "method": "pca",  # "simple" or "pca"
        "pca": {
            "n_latents_for_pca": 5000,
            "n_identity_components": 64,
        },
        "simple": {
            "alpha": 0.5,  # Interpolation factor
        },
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def load_vae(config: dict, device: torch.device) -> VAE:
    """Load trained VAE from checkpoint."""
    vae = VAE(
        in_channels=config["vae"]["in_channels"],
        base_channels=config["vae"]["base_channels"],
        latent_dim=config["vae"]["latent_dim"],
        blocks_per_level=config["vae"]["blocks_per_level"],
        image_size=config["vae"]["image_size"],
        predict_variance=config["vae"]["predict_variance"],
    ).to(device)

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


def get_dataset_with_identity(config: dict) -> CelebAWithAttributes:
    """Get CelebA dataset with identity labels."""
    return CelebAWithAttributes(
        root=config["data"]["root"],
        split="train",
        image_size=config["data"]["image_size"],
        download=config["data"]["download"],
        return_identity=True,
    )


# =============================================================================
# Phase 1: Identity Exploration
# =============================================================================

def run_identity_exploration(config: dict):
    """
    Explore available identities to help select good anchors.

    This step helps you find identities with many images and visually
    inspect them before selecting your final 3 anchors.
    """
    print("\n" + "=" * 60)
    print("Phase 1: Identity Exploration")
    print("=" * 60)

    output_dir = Path(config["identity_transfer"]["output_dir"]) / "identity_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\nLoading CelebA dataset with identity labels...")
    dataset = get_dataset_with_identity(config)
    print(f"Dataset size: {len(dataset)}")

    # Explore identities
    min_images = config["identity_transfer"]["min_images_per_identity"]
    results = explore_identities(
        dataset=dataset,
        min_images=min_images,
        n_to_show=20,
        output_dir=output_dir,
    )

    # Print summary
    print(f"\n" + "=" * 60)
    print("Identity Exploration Complete!")
    print("=" * 60)
    print(f"\nFound {results['total_rich_identities']} identities with >= {min_images} images")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review identity sample images in {output_dir}")
    print(f"  2. Select 3 diverse anchors (e.g., different genders, ages)")
    print(f"  3. Note their identity IDs")
    print(f"  4. Run: python {__file__} --run --anchors ID1 ID2 ID3")

    return results


# =============================================================================
# Phase 2: Improved Identity Transfer
# =============================================================================

def run_improved_identity_transfer(
    config: dict,
    vae: VAE,
    dataset: CelebAWithAttributes,
    anchor_ids: list,
    device: torch.device,
):
    """
    Run identity transfer with CORRECT anchor computation.

    The FIX: Average multiple images of the SAME person, not random people.
    """
    print("\n" + "=" * 60)
    print("Phase 2: Improved Identity Transfer")
    print("=" * 60)

    it_config = config["identity_transfer"]
    output_dir = Path(it_config["output_dir"]) / "improved_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_anchors = len(anchor_ids)
    n_subjects = it_config["n_subjects"]
    n_images_per_anchor = it_config["n_images_per_anchor"]
    method = it_config["method"]

    # ======================
    # Step 1: Compute TRUE anchor latents
    # ======================
    print(f"\nStep 1: Computing TRUE anchor latents...")
    print(f"  Method: Average {n_images_per_anchor} images of SAME person")

    anchor_latents = []
    anchor_images_list = []

    for i, anchor_id in enumerate(anchor_ids):
        print(f"\n  Anchor {i+1}: Identity {anchor_id}")

        # Compute true essence from same person's images
        essence, source_images = compute_true_anchor_latent(
            vae=vae,
            dataset=dataset,
            identity_id=anchor_id,
            max_images=n_images_per_anchor,
            device=device,
        )

        anchor_latents.append(essence)
        anchor_images_list.append(source_images)

        # Save anchor source images
        visualize_identity_samples(
            identity_id=anchor_id,
            images=source_images,
            output_path=output_dir / f"anchor_{i+1}_identity_{anchor_id}_sources.png",
            title=f"Anchor {i+1} (ID: {anchor_id}) - {len(source_images)} Source Images",
        )

    # ======================
    # Step 2: Get subject images
    # ======================
    print(f"\nStep 2: Getting {n_subjects} subject images...")

    # Get subjects that are NOT the anchor identities
    used_indices = set()
    for anchor_id in anchor_ids:
        _, indices = get_images_by_identity_id(dataset, anchor_id)
        used_indices.update(indices)

    subject_images = []
    subject_idx = 0
    while len(subject_images) < n_subjects:
        if subject_idx not in used_indices:
            image, _, _ = dataset[subject_idx]
            subject_images.append(image)
        subject_idx += 1

    subject_images = torch.stack(subject_images).to(device)
    print(f"  Got {len(subject_images)} subject images")

    # ======================
    # Step 3: Fit PCA if needed
    # ======================
    pca = None
    if method == "pca":
        print(f"\nStep 3: Fitting PCA on latent space...")

        # Create dataloader for PCA fitting
        train_loader, _, _ = get_celeba_attribute_dataloaders(
            root=config["data"]["root"],
            image_size=config["data"]["image_size"],
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            download=config["data"]["download"],
            return_identity=False,  # Don't need identity for PCA
        )

        pca, _ = fit_latent_pca(
            vae=vae,
            dataloader=train_loader,
            n_latents=it_config["pca"]["n_latents_for_pca"],
            device=device,
        )

    # ======================
    # Step 4: Perform identity transfer
    # ======================
    print(f"\nStep 4: Performing identity transfer using '{method}' method...")

    results = {
        "subjects": subject_images.cpu(),
    }

    for anchor_idx, anchor_latent in enumerate(tqdm(anchor_latents, desc="Anchors")):
        morphed_list = []

        for subj_idx in range(n_subjects):
            subject_image = subject_images[subj_idx:subj_idx+1]

            if method == "pca":
                morphed = transfer_identity_pca(
                    vae=vae,
                    subject_image=subject_image,
                    anchor_latent=anchor_latent,
                    pca=pca,
                    n_identity_components=it_config["pca"]["n_identity_components"],
                    device=device,
                )
            else:
                morphed = transfer_identity_simple(
                    vae=vae,
                    subject_image=subject_image,
                    anchor_latent=anchor_latent,
                    alpha=it_config["simple"]["alpha"],
                    device=device,
                )

            morphed_list.append(morphed.cpu())

        results[f"morphed_anchor_{anchor_idx}"] = torch.cat(morphed_list, dim=0)

    # ======================
    # Step 5: Visualize results
    # ======================
    print(f"\nStep 5: Saving visualizations...")

    # Main result grid
    n_cols = 1 + n_anchors
    fig, axes = plt.subplots(n_subjects, n_cols, figsize=(n_cols * 1.5, n_subjects * 1.5))

    for i in range(n_subjects):
        # Subject original
        img = results["subjects"][i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Subject", fontsize=9)

        # Morphed toward each anchor
        for j in range(n_anchors):
            morphed = results[f"morphed_anchor_{j}"][i].permute(1, 2, 0).numpy()
            axes[i, j + 1].imshow(np.clip(morphed, 0, 1))
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f"→ Anchor {j + 1}", fontsize=9)

    plt.suptitle(f"Improved Identity Transfer ({method.upper()})", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "identity_transfer_improved.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save anchor reference grid
    fig, axes = plt.subplots(1, n_anchors, figsize=(n_anchors * 2, 2))
    if n_anchors == 1:
        axes = [axes]

    for i in range(n_anchors):
        # Use first source image as representative
        img = anchor_images_list[i][0].permute(1, 2, 0).numpy()
        axes[i].imshow(np.clip(img, 0, 1))
        axes[i].axis('off')
        axes[i].set_title(f"Anchor {i + 1}\n(ID: {anchor_ids[i]})", fontsize=9)

    plt.suptitle("Anchor Identities (Representative Images)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "anchor_identities.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save metadata
    metadata = {
        "anchor_ids": anchor_ids,
        "n_images_per_anchor": n_images_per_anchor,
        "n_subjects": n_subjects,
        "method": method,
        "method_params": it_config[method] if method in it_config else {},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n" + "=" * 60)
    print("Improved Identity Transfer Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - identity_transfer_improved.png: Main result grid")
    print(f"  - anchor_identities.png: Anchor reference")
    for i in range(n_anchors):
        print(f"  - anchor_{i+1}_identity_{anchor_ids[i]}_sources.png: Anchor source images")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Improved Identity Transfer (fixes the averaging bug)"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Run identity exploration to find suitable anchors",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run identity transfer with specified anchors",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=None,
        help="Identity IDs to use as anchors (e.g., --anchors 1234 5678 9012)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Default: auto-detect.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to VAE checkpoint.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["simple", "pca"],
        default=None,
        help="Transfer method. Default: pca",
    )
    args = parser.parse_args()

    # Setup
    config = DEFAULT_CONFIG.copy()

    if args.checkpoint:
        config["vae"]["checkpoint_path"] = args.checkpoint

    if args.method:
        config["identity_transfer"]["method"] = args.method

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)

    # Run exploration or transfer
    if args.explore:
        run_identity_exploration(config)

    elif args.run:
        # Load VAE
        print("\nLoading trained VAE...")
        vae = load_vae(config, device)

        # Load dataset
        print("\nLoading CelebA dataset with identity labels...")
        dataset = get_dataset_with_identity(config)

        # Get anchor IDs
        if args.anchors:
            anchor_ids = args.anchors
            print(f"Using specified anchors: {anchor_ids}")
        else:
            # Auto-select from rich identities
            print("\nNo anchors specified. Auto-selecting from rich identities...")
            rich_ids, counts = find_rich_identities(
                dataset,
                min_images=config["identity_transfer"]["min_images_per_identity"],
                max_identities=20,
            )

            if len(rich_ids) < 3:
                raise ValueError(
                    f"Not enough identities with sufficient images. "
                    f"Found {len(rich_ids)}, need at least 3."
                )

            # Select evenly spaced for diversity
            indices = [0, len(rich_ids)//2, len(rich_ids)-1]
            anchor_ids = [rich_ids[i] for i in indices[:3]]
            print(f"Auto-selected anchors: {anchor_ids}")
            print("  (Run with --explore first to manually select better anchors)")

        # Run transfer
        run_improved_identity_transfer(config, vae, dataset, anchor_ids, device)

    else:
        print("Please specify --explore or --run")
        print("\nExample workflow:")
        print(f"  1. python {__file__} --explore")
        print(f"  2. Review images in outputs/experiments/identity_transfer/identity_exploration/")
        print(f"  3. python {__file__} --run --anchors 1234 5678 9012")


if __name__ == "__main__":
    main()

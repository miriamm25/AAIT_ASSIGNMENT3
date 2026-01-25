"""
Component: Identity Transfer Utilities
Reference: Task 2.3 Improvement Plan

Purpose:
    Helper functions for proper identity-based sample selection.
    Fixes the bug where random different people were averaged instead of
    multiple images of the SAME person.

Key implementation notes:
    The original bug:
        Anchor "essence" = average of [Random Person 1, Random Person 2, ...]
        Result: Generic average face, not a specific identity

    The fix:
        Anchor "essence" = average of [Person A pose 1, Person A pose 2, ...]
        Result: Person A's unique identity features

From the assignment (AAIT_Assignment_3.pdf Task 2.3):
    "Encode multiple images of each anchor -> average latent = essence"
    This requires multiple images of the SAME person.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def find_rich_identities(
    dataset,
    min_images: int = 10,
    max_identities: Optional[int] = None,
) -> Tuple[List[int], Dict[int, int]]:
    """
    Find CelebA identities that have at least min_images photos.

    Reference: Task 2.3 Improvement Plan
        "Find identities with multiple images so we can compute true essence"

    Args:
        dataset: CelebA dataset with return_identity=True
        min_images: Minimum number of images required per identity
        max_identities: Maximum number of identities to return (None = all)

    Returns:
        Tuple of:
            - List of identity IDs with >= min_images samples
            - Dictionary mapping identity ID to image count
    """
    print(f"Scanning dataset for identities with >= {min_images} images...")

    identity_counts = Counter()

    for idx in tqdm(range(len(dataset)), desc="Counting identities"):
        # Handle different dataset formats
        sample = dataset[idx]
        if len(sample) == 3:
            _, _, identity = sample
        else:
            raise ValueError(f"Expected 3 elements (image, attrs, identity), got {len(sample)}")

        # Identity can be a tensor or int
        if hasattr(identity, 'item'):
            identity = identity.item()
        identity_counts[identity] += 1

    # Filter to identities with enough images
    rich_identities = [
        id_ for id_, count in identity_counts.items()
        if count >= min_images
    ]

    # Sort by count (descending)
    rich_identities.sort(key=lambda x: identity_counts[x], reverse=True)

    if max_identities is not None:
        rich_identities = rich_identities[:max_identities]

    print(f"Found {len(rich_identities)} identities with >= {min_images} images")

    # Create filtered counts dict
    rich_counts = {id_: identity_counts[id_] for id_ in rich_identities}

    return rich_identities, rich_counts


def get_images_by_identity_id(
    dataset,
    identity_id: int,
    max_images: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Get all images of a specific person from the dataset.

    Reference: Task 2.3 Improvement Plan
        "Get all images of this person for computing true identity essence"

    Args:
        dataset: CelebA dataset with return_identity=True
        identity_id: Identity ID to search for
        max_images: Maximum number of images to return (None = all)

    Returns:
        Tuple of:
            - List of image tensors for that identity
            - List of dataset indices for those images
    """
    images = []
    indices = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        if len(sample) == 3:
            image, _, identity = sample
        else:
            raise ValueError(f"Expected 3 elements, got {len(sample)}")

        if hasattr(identity, 'item'):
            identity = identity.item()

        if identity == identity_id:
            images.append(image)
            indices.append(idx)

            if max_images is not None and len(images) >= max_images:
                break

    return images, indices


def compute_true_anchor_latent(
    vae: nn.Module,
    dataset,
    identity_id: int,
    max_images: int = 10,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute the TRUE identity essence by averaging multiple images of the SAME person.

    Reference: AAIT_Assignment_3.pdf Task 2.3
        "Encode multiple images of each anchor -> average latent = essence"

    This is the CORRECT implementation. The bug was averaging random people.

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        dataset: CelebA dataset with return_identity=True
        identity_id: Identity ID of the anchor person
        max_images: Maximum number of images to use for averaging
        device: Computation device

    Returns:
        Tuple of:
            - Anchor latent vector (1, latent_dim) - the identity "essence"
            - List of source images used for this anchor
    """
    vae.eval()

    # Get all images of this person
    person_images, indices = get_images_by_identity_id(dataset, identity_id, max_images)

    if len(person_images) == 0:
        raise ValueError(f"No images found for identity {identity_id}")

    print(f"  Computing essence from {len(person_images)} images of identity {identity_id}")

    # Stack and encode
    images_tensor = torch.stack(person_images).to(device)

    with torch.no_grad():
        latents = []
        for i in range(len(images_tensor)):
            img = images_tensor[i:i+1]
            mu, _ = vae.encode(img)
            latents.append(mu)

        # Average = identity essence (pose/expression averaged out)
        latents = torch.cat(latents, dim=0)  # (N, latent_dim)
        essence = latents.mean(dim=0, keepdim=True)  # (1, latent_dim)

    return essence, person_images


def visualize_identity_samples(
    identity_id: int,
    images: List[torch.Tensor],
    output_path: Path,
    title: Optional[str] = None,
):
    """
    Visualize all images of a specific identity.

    Args:
        identity_id: Identity ID
        images: List of image tensors
        output_path: Path to save visualization
        title: Optional custom title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, img in enumerate(images):
        row, col = i // n_cols, i % n_cols
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        axes[row, col].imshow(img_np)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Image {i+1}', fontsize=8)

    # Hide empty subplots
    for i in range(n_images, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    title = title or f"Identity {identity_id} - {n_images} Images"
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def explore_identities(
    dataset,
    min_images: int = 10,
    n_to_show: int = 10,
    output_dir: Path = None,
) -> Dict:
    """
    Explore available identities and optionally save sample visualizations.

    Reference: Task 2.3 Improvement Plan
        "Run identity exploration to find rich identities"

    Args:
        dataset: CelebA dataset with return_identity=True
        min_images: Minimum images per identity
        n_to_show: Number of identities to show samples for
        output_dir: Directory to save visualizations (optional)

    Returns:
        Dictionary with identity exploration results
    """
    # Find rich identities
    rich_ids, counts = find_rich_identities(dataset, min_images)

    print(f"\nTop {min(10, len(rich_ids))} identities by image count:")
    for i, id_ in enumerate(rich_ids[:10]):
        print(f"  {i+1}. Identity {id_}: {counts[id_]} images")

    results = {
        "rich_identities": rich_ids[:n_to_show],
        "counts": {str(id_): counts[id_] for id_ in rich_ids[:n_to_show]},
        "total_rich_identities": len(rich_ids),
        "min_images_threshold": min_images,
    }

    # Save visualizations if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        with open(output_dir / "rich_identities.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save sample visualizations
        print(f"\nSaving sample visualizations for top {n_to_show} identities...")
        for i, id_ in enumerate(tqdm(rich_ids[:n_to_show], desc="Visualizing identities")):
            images, _ = get_images_by_identity_id(dataset, id_, max_images=10)
            visualize_identity_samples(
                identity_id=id_,
                images=images,
                output_path=output_dir / f"identity_{id_}_samples.png",
                title=f"Identity {id_} ({counts[id_]} total images)"
            )

        print(f"Saved identity exploration to {output_dir}")

    return results


def select_diverse_anchors(
    dataset,
    n_anchors: int = 3,
    min_images: int = 10,
    manual_ids: Optional[List[int]] = None,
    device: torch.device = torch.device("cuda"),
) -> List[Dict]:
    """
    Select diverse anchor identities for identity transfer.

    Reference: Task 2.3 Improvement Plan
        "Manually select 3 diverse anchors"

    Args:
        dataset: CelebA dataset with return_identity=True
        n_anchors: Number of anchors to select
        min_images: Minimum images required per identity
        manual_ids: Optional list of specific identity IDs to use
        device: Computation device (for future use)

    Returns:
        List of anchor dictionaries, each containing:
            - "id": Identity ID
            - "n_images": Number of available images
            - "description": Placeholder for manual labeling
    """
    if manual_ids is not None:
        # Use manually specified IDs
        selected_ids = manual_ids[:n_anchors]
    else:
        # Automatically select from top identities
        rich_ids, counts = find_rich_identities(dataset, min_images, max_identities=50)

        # Select evenly spaced identities for diversity
        # (In practice, you'd want to manually inspect and select)
        indices = np.linspace(0, min(len(rich_ids)-1, 20), n_anchors, dtype=int)
        selected_ids = [rich_ids[i] for i in indices]

    anchors = []
    for id_ in selected_ids:
        images, _ = get_images_by_identity_id(dataset, id_, max_images=1)
        anchors.append({
            "id": id_,
            "n_images": len(images) if images else 0,
            "description": f"Identity_{id_} (manually label this)",
        })

    return anchors


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test identity utilities.
    """
    print("Testing Identity Utilities...")
    print("-" * 50)

    # We can't easily test without the actual dataset, so just test the logic

    # Test Counter logic
    print("Testing Counter logic...")
    test_identities = [1, 1, 1, 2, 2, 3, 1, 1, 2]
    counts = Counter(test_identities)
    print(f"  Counts: {dict(counts)}")

    rich = [id_ for id_, count in counts.items() if count >= 3]
    print(f"  Identities with >= 3 images: {rich}")

    assert 1 in rich and 2 not in rich and 3 not in rich
    print("  Counter logic passed!")

    print("\n" + "-" * 50)
    print("Identity utility tests passed!")
    print("\nNote: Full testing requires CelebA dataset with identity labels.")
    print("Run the improved identity transfer script to test with real data.")

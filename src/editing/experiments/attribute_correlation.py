"""
Component: Attribute-Correlated Dimension Discovery
Reference: Task 2.1 Improvement - Experiment 1

Purpose:
    Find latent dimensions that correlate with specific CelebA attributes
    instead of just using high variance dimensions.

Key implementation notes:
    - For each dimension, compute correlation with each attribute
    - Select dimensions with highest absolute correlation for target attributes
    - This provides semantic interpretation of dimensions

Expected attributes to target:
    - Smiling (index 31)
    - Eyeglasses (index 15)
    - Male (index 20)
    - Young (index 39)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from pathlib import Path


# CelebA attribute names for reference
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def compute_all_correlations(
    vae: nn.Module,
    dataloader: DataLoader,
    n_samples: int = 5000,
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute correlation between every latent dimension and every attribute.

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        dataloader: DataLoader returning (images, attributes) or (images, attrs, identity)
        n_samples: Number of samples to use for correlation computation
        device: Computation device

    Returns:
        Tuple of:
            - correlations: (n_dims, n_attrs) correlation matrix
            - latents: (n_samples, n_dims) all collected latents
            - attributes: (n_samples, n_attrs) all collected attributes
    """
    vae.eval()
    all_latents = []
    all_attributes = []

    with torch.no_grad():
        samples_collected = 0
        for batch in tqdm(dataloader, desc="Collecting latents and attributes"):
            if samples_collected >= n_samples:
                break

            # Handle different dataloader formats
            if len(batch) == 2:
                images, attrs = batch
            elif len(batch) == 3:
                images, attrs, _ = batch  # Ignore identity
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            images = images.to(device)

            # Encode images
            mu, _ = vae.encode(images)
            all_latents.append(mu.cpu())
            all_attributes.append(attrs)

            samples_collected += images.shape[0]

    # Concatenate and trim to exact n_samples
    latents = torch.cat(all_latents, dim=0)[:n_samples].numpy()  # (N, latent_dim)
    attributes = torch.cat(all_attributes, dim=0)[:n_samples].numpy()  # (N, 40)

    n_dims = latents.shape[1]
    n_attrs = attributes.shape[1]

    # Compute correlation matrix
    correlations = np.zeros((n_dims, n_attrs))

    for dim in tqdm(range(n_dims), desc="Computing correlations"):
        z_values = latents[:, dim]

        for attr_idx in range(n_attrs):
            attr_values = attributes[:, attr_idx]

            # Pearson correlation
            if np.std(z_values) > 0 and np.std(attr_values) > 0:
                corr = np.corrcoef(z_values, attr_values)[0, 1]
                correlations[dim, attr_idx] = corr
            else:
                correlations[dim, attr_idx] = 0.0

    return correlations, latents, attributes


def find_attribute_correlated_dimensions(
    vae: nn.Module,
    dataloader: DataLoader,
    target_attributes: List[str] = None,
    n_samples: int = 5000,
    device: torch.device = torch.device("cuda"),
    save_path: Optional[Path] = None,
) -> Dict[str, Tuple[int, float]]:
    """
    Find the latent dimension most correlated with each target attribute.

    Reference: Task 2.1 Improvement Plan
        "Find dimensions that correlate with specific CelebA attributes
        instead of just high variance."

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        dataloader: DataLoader returning (images, attributes)
        target_attributes: List of attribute names to find correlations for
                          Default: ["Smiling", "Eyeglasses", "Male", "Young"]
        n_samples: Number of samples to use
        device: Computation device
        save_path: Optional path to save full correlation report

    Returns:
        Dictionary mapping attribute name to (best_dimension, correlation_value)
        Example: {"Smiling": (42, 0.35), "Eyeglasses": (128, 0.28)}
    """
    if target_attributes is None:
        target_attributes = ["Smiling", "Eyeglasses", "Male", "Young"]

    # Compute all correlations
    correlations, latents, attributes = compute_all_correlations(
        vae, dataloader, n_samples, device
    )

    # Find best dimension for each target attribute
    results = {}
    all_results = {}

    for attr_name in target_attributes:
        attr_idx = CELEBA_ATTRIBUTES.index(attr_name)

        # Get correlations for this attribute across all dimensions
        attr_correlations = correlations[:, attr_idx]

        # Find dimension with highest absolute correlation
        best_dim = np.argmax(np.abs(attr_correlations))
        best_corr = attr_correlations[best_dim]

        results[attr_name] = (int(best_dim), float(best_corr))

        # Also store top-5 for analysis
        top_5_dims = np.argsort(np.abs(attr_correlations))[::-1][:5]
        all_results[attr_name] = {
            "best_dim": int(best_dim),
            "best_corr": float(best_corr),
            "top_5_dims": [int(d) for d in top_5_dims],
            "top_5_corrs": [float(attr_correlations[d]) for d in top_5_dims],
        }

    # Save full report if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "target_attributes": target_attributes,
            "results": all_results,
            "full_correlation_matrix_shape": list(correlations.shape),
            "n_samples_used": n_samples,
        }

        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        # Also save the full correlation matrix as numpy
        np.save(save_path.parent / "correlation_matrix.npy", correlations)

        print(f"Saved correlation report to {save_path}")

    # Print summary
    print("\nAttribute-Correlated Dimensions Found:")
    print("-" * 50)
    for attr_name, (dim, corr) in results.items():
        sign = "+" if corr > 0 else "-"
        print(f"  {attr_name:20s}: Dimension {dim:3d} (r = {sign}{abs(corr):.3f})")

    return results


def get_correlation_ranking(
    correlations: np.ndarray,
    attr_name: str,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Get top-k dimensions most correlated with a specific attribute.

    Args:
        correlations: Full correlation matrix (n_dims, n_attrs)
        attr_name: Name of the attribute
        top_k: Number of top dimensions to return

    Returns:
        List of (dimension, correlation) tuples, sorted by absolute correlation
    """
    attr_idx = CELEBA_ATTRIBUTES.index(attr_name)
    attr_correlations = correlations[:, attr_idx]

    # Sort by absolute value
    sorted_dims = np.argsort(np.abs(attr_correlations))[::-1]

    return [(int(dim), float(attr_correlations[dim])) for dim in sorted_dims[:top_k]]


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test correlation computation with dummy data.
    """
    print("Testing Attribute Correlation...")
    print("-" * 50)

    # Create dummy data
    n_samples = 100
    latent_dim = 256
    n_attrs = 40

    # Simulate latents and attributes
    latents = np.random.randn(n_samples, latent_dim)
    attributes = np.random.randint(0, 2, (n_samples, n_attrs)).astype(float)

    # Make dimension 42 correlate with attribute 31 (Smiling)
    latents[:, 42] = attributes[:, 31] * 2 + np.random.randn(n_samples) * 0.5

    # Compute correlations manually
    corr = np.corrcoef(latents[:, 42], attributes[:, 31])[0, 1]
    print(f"Expected high correlation for dim 42 with Smiling: {corr:.3f}")

    # Test full correlation matrix computation
    correlations = np.zeros((latent_dim, n_attrs))
    for dim in range(latent_dim):
        for attr in range(n_attrs):
            if np.std(latents[:, dim]) > 0 and np.std(attributes[:, attr]) > 0:
                correlations[dim, attr] = np.corrcoef(latents[:, dim], attributes[:, attr])[0, 1]

    # Find best dim for Smiling
    smiling_correlations = correlations[:, 31]
    best_dim = np.argmax(np.abs(smiling_correlations))
    print(f"Best dimension for Smiling: {best_dim} (expected: 42)")
    print(f"Correlation: {smiling_correlations[best_dim]:.3f}")

    assert best_dim == 42, f"Expected dim 42, got {best_dim}"
    print("\nAll tests passed!")

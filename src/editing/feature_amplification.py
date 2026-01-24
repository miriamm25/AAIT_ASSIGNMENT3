"""
Component: Feature Amplification (Task 2.1)
Reference: AAIT_Assignment_3.pdf Task 2 - Feature Amplification

Purpose:
    Find latent dimensions that control meaningful visual features and
    visualize the effect of amplifying each dimension.

Key implementation notes:
    - "For component z_i, compute z'_i = z_i + α where α is scalar"
    - 4 meaningful components discovered
    - 10 values of α per component
    - 8 different samples per component
    - Total: 4 × 10 × 8 grid visualization
    - Cherry-picking is allowed

Method:
    1. Compute variance of each latent dimension across many samples
    2. Select dimensions with highest variance (most variation = most meaningful)
    3. For each selected dimension, vary α and observe visual changes
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm


def find_meaningful_dimensions(
    vae: nn.Module,
    dataloader: DataLoader,
    n_dims: int = 4,
    n_images: int = 1000,
    device: torch.device = torch.device("cuda"),
) -> Tuple[List[int], torch.Tensor]:
    """
    Find latent dimensions with highest variance across the dataset.

    Reference: AAIT_Assignment_3.pdf Task 2.1
        High variance dimensions encode more information and control
        more meaningful visual features.

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        dataloader: DataLoader for images
        n_dims: Number of dimensions to return
        n_images: Number of images to use for variance computation
        device: Computation device

    Returns:
        Tuple of (dimension_indices, variance_per_dimension)
    """
    vae.eval()
    latents = []

    with torch.no_grad():
        images_processed = 0
        for batch in tqdm(dataloader, desc="Computing latent statistics"):
            if images_processed >= n_images:
                break

            # Handle both (image,) and (image, attrs) formats
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            batch = batch.to(device)

            # Encode to get mean latent (ignore variance)
            mu, _ = vae.encode(batch)
            latents.append(mu.cpu())

            images_processed += batch.shape[0]

    # Concatenate all latents
    latents = torch.cat(latents, dim=0)[:n_images]  # (N, latent_dim)

    # Compute variance per dimension
    variance = latents.var(dim=0)  # (latent_dim,)

    # Get indices of top-k highest variance dimensions
    top_indices = torch.argsort(variance, descending=True)[:n_dims].tolist()

    return top_indices, variance


def amplify_dimension(
    vae: nn.Module,
    image: torch.Tensor,
    dim_idx: int,
    alphas: List[float],
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Generate images with varying amplification of a single latent dimension.

    Reference: AAIT_Assignment_3.pdf Task 2.1
        "For component z_i, compute z'_i = z_i + α where α is scalar"

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        image: Input image tensor (1, C, H, W)
        dim_idx: Index of latent dimension to amplify
        alphas: List of alpha values for amplification
        device: Computation device

    Returns:
        Tensor of generated images (len(alphas), C, H, W)
    """
    vae.eval()
    image = image.to(device)

    with torch.no_grad():
        # Encode image to get mean latent
        mu, _ = vae.encode(image)  # (1, latent_dim)

        results = []
        for alpha in alphas:
            # Copy latent and amplify single dimension
            z_modified = mu.clone()
            z_modified[0, dim_idx] = z_modified[0, dim_idx] + alpha

            # Decode
            if hasattr(vae, 'predict_variance') and vae.predict_variance:
                recon, _ = vae.decode(z_modified)
            else:
                recon = vae.decode(z_modified)

            results.append(recon)

    # Stack results
    results = torch.cat(results, dim=0)  # (len(alphas), C, H, W)
    return results


def create_amplification_grid(
    vae: nn.Module,
    images: torch.Tensor,
    dimensions: List[int],
    alphas: List[float],
    device: torch.device = torch.device("cuda"),
    dimension_names: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Create full visualization grid for feature amplification.

    Reference: AAIT_Assignment_3.pdf Task 2.1
        "4 dimensions × 10 alpha values × 8 samples"

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        images: Input images tensor (N, C, H, W) where N = number of samples
        dimensions: List of dimension indices to amplify
        alphas: List of alpha values
        device: Computation device
        dimension_names: Optional names for each dimension

    Returns:
        Grid tensor of shape (N * len(dimensions), len(alphas), C, H, W)
    """
    vae.eval()
    n_samples = images.shape[0]
    n_dims = len(dimensions)
    n_alphas = len(alphas)

    # Results: for each sample and dimension, generate alpha variations
    all_grids = []

    for sample_idx in tqdm(range(n_samples), desc="Generating amplification grids"):
        image = images[sample_idx:sample_idx+1]  # (1, C, H, W)

        sample_grids = []
        for dim_idx in dimensions:
            # Generate images with varying alpha for this dimension
            dim_results = amplify_dimension(vae, image, dim_idx, alphas, device)
            sample_grids.append(dim_results)  # (n_alphas, C, H, W)

        # Stack dimensions for this sample
        sample_grids = torch.stack(sample_grids, dim=0)  # (n_dims, n_alphas, C, H, W)
        all_grids.append(sample_grids)

    # Stack all samples
    all_grids = torch.stack(all_grids, dim=0)  # (n_samples, n_dims, n_alphas, C, H, W)

    return all_grids


def visualize_amplification_single_sample(
    grid: torch.Tensor,
    dimensions: List[int],
    alphas: List[float],
    save_path: Path,
    dimension_names: Optional[List[str]] = None,
    title: str = "Feature Amplification",
):
    """
    Visualize feature amplification for a single sample.

    Args:
        grid: Tensor of shape (n_dims, n_alphas, C, H, W)
        dimensions: List of dimension indices
        alphas: List of alpha values
        save_path: Path to save the figure
        dimension_names: Optional names for each dimension
        title: Plot title
    """
    n_dims, n_alphas = grid.shape[:2]

    fig, axes = plt.subplots(n_dims, n_alphas, figsize=(n_alphas * 1.5, n_dims * 1.5))

    if n_dims == 1:
        axes = axes.reshape(1, -1)

    for i, dim_idx in enumerate(dimensions):
        for j, alpha in enumerate(alphas):
            img = grid[i, j].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)

            axes[i, j].imshow(img)
            axes[i, j].axis('off')

            # Add alpha label on top row
            if i == 0:
                axes[i, j].set_title(f'α={alpha:.1f}', fontsize=8)

        # Add dimension label on left
        dim_name = dimension_names[i] if dimension_names else f'Dim {dim_idx}'
        axes[i, 0].set_ylabel(dim_name, fontsize=10, rotation=0, ha='right', va='center')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_amplification_all_samples(
    grids: torch.Tensor,
    dimensions: List[int],
    alphas: List[float],
    output_dir: Path,
    dimension_names: Optional[List[str]] = None,
):
    """
    Save amplification visualizations for all samples and all dimensions.

    Args:
        grids: Tensor of shape (n_samples, n_dims, n_alphas, C, H, W)
        dimensions: List of dimension indices
        alphas: List of alpha values
        output_dir: Directory to save figures
        dimension_names: Optional names for each dimension
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = grids.shape[0]

    # Save individual sample visualizations
    for sample_idx in range(n_samples):
        visualize_amplification_single_sample(
            grids[sample_idx],
            dimensions,
            alphas,
            output_dir / f"sample_{sample_idx:02d}.png",
            dimension_names,
            f"Feature Amplification - Sample {sample_idx + 1}",
        )

    # Create combined grid for each dimension
    for dim_i, dim_idx in enumerate(dimensions):
        dim_name = dimension_names[dim_i] if dimension_names else f"Dim_{dim_idx}"

        # Create grid: samples (rows) × alphas (cols)
        n_alphas = len(alphas)
        fig, axes = plt.subplots(n_samples, n_alphas, figsize=(n_alphas * 1.2, n_samples * 1.2))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for sample_idx in range(n_samples):
            for alpha_idx, alpha in enumerate(alphas):
                img = grids[sample_idx, dim_i, alpha_idx].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)

                axes[sample_idx, alpha_idx].imshow(img)
                axes[sample_idx, alpha_idx].axis('off')

                if sample_idx == 0:
                    axes[sample_idx, alpha_idx].set_title(f'α={alpha:.1f}', fontsize=8)

        plt.suptitle(f"Feature Amplification - {dim_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"dimension_{dim_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved amplification visualizations to {output_dir}")


def discover_feature_meanings(
    vae: nn.Module,
    dataloader: DataLoader,
    dimensions: List[int],
    alphas: List[float],
    n_test_images: int = 4,
    device: torch.device = torch.device("cuda"),
) -> Dict[int, str]:
    """
    Interactive helper to discover what each dimension controls.

    This generates small test grids to help manually identify
    what visual feature each dimension corresponds to.

    Args:
        vae: Trained VAE model
        dataloader: DataLoader for images
        dimensions: Dimension indices to test
        alphas: Alpha values to test
        n_test_images: Number of test images
        device: Computation device

    Returns:
        Dictionary mapping dimension index to suggested name
    """
    # Get test images
    test_images = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        test_images.append(batch)
        if sum(x.shape[0] for x in test_images) >= n_test_images:
            break

    test_images = torch.cat(test_images, dim=0)[:n_test_images]

    # Generate grids for inspection
    grids = create_amplification_grid(
        vae, test_images, dimensions, alphas, device
    )

    # Return placeholder names - user should manually identify
    suggested_names = {}
    for i, dim_idx in enumerate(dimensions):
        suggested_names[dim_idx] = f"Feature_{i+1}_dim{dim_idx}"

    return suggested_names


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test feature amplification with dummy data.
    """
    print("Testing Feature Amplification...")
    print("-" * 50)

    # Create dummy VAE for testing
    class DummyVAE(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.predict_variance = False

        def encode(self, x):
            batch_size = x.shape[0]
            mu = torch.randn(batch_size, self.latent_dim)
            log_var = torch.zeros(batch_size, self.latent_dim)
            return mu, log_var

        def decode(self, z):
            batch_size = z.shape[0]
            return torch.rand(batch_size, 3, 64, 64)

    device = torch.device("cpu")
    vae = DummyVAE().to(device)

    # Test amplify_dimension
    print("Testing amplify_dimension...")
    image = torch.rand(1, 3, 64, 64)
    alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    results = amplify_dimension(vae, image, dim_idx=0, alphas=alphas, device=device)
    print(f"  Input shape: {image.shape}")
    print(f"  Output shape: {results.shape}")
    assert results.shape == (len(alphas), 3, 64, 64)
    print("  amplify_dimension passed!")

    # Test create_amplification_grid
    print("\nTesting create_amplification_grid...")
    images = torch.rand(4, 3, 64, 64)  # 4 samples
    dimensions = [0, 10, 50, 100]  # 4 dimensions
    grids = create_amplification_grid(vae, images, dimensions, alphas, device)
    print(f"  Images shape: {images.shape}")
    print(f"  Grid shape: {grids.shape}")
    expected_shape = (4, 4, 5, 3, 64, 64)  # (n_samples, n_dims, n_alphas, C, H, W)
    assert grids.shape == expected_shape, f"Expected {expected_shape}, got {grids.shape}"
    print("  create_amplification_grid passed!")

    print("\n" + "-" * 50)
    print("All feature amplification tests passed!")

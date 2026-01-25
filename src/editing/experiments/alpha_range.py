"""
Component: Wider Alpha Range Testing
Reference: Task 2.1 Improvement - Experiment 2

Purpose:
    Test if larger alpha values [-5, 5] reveal clearer effects compared to [-3, 3].

Key implementation notes:
    - The latent prior is N(0, 1), so |alpha| > 3 is unusual
    - May produce more dramatic visual changes
    - Risk of artifacts when going "out of distribution"
    - Compare side-by-side with original range

From the plan:
    "The assignment does NOT specify the alpha range. You can use any range."
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm


def amplify_dimension(
    vae: nn.Module,
    image: torch.Tensor,
    dim_idx: int,
    alphas: List[float],
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Generate images with varying amplification of a single latent dimension.

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


def compare_alpha_ranges(
    vae: nn.Module,
    images: torch.Tensor,
    dimensions: List[int],
    alpha_ranges: Dict[str, Tuple[float, float]] = None,
    n_alphas: int = 10,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, torch.Tensor]:
    """
    Compare feature amplification with different alpha ranges.

    Reference: Task 2.1 Improvement Plan - Experiment 2
        "Test if larger alpha values reveal clearer effects or cause artifacts"

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        images: Input images tensor (N, C, H, W)
        dimensions: List of dimension indices to amplify
        alpha_ranges: Dict mapping range name to (min, max) tuple
                     Default: {"narrow": (-3, 3), "wide": (-5, 5)}
        n_alphas: Number of alpha values per range
        device: Computation device

    Returns:
        Dictionary with grids for each alpha range:
            - "narrow": tensor of shape (n_samples, n_dims, n_alphas, C, H, W)
            - "wide": tensor of shape (n_samples, n_dims, n_alphas, C, H, W)
    """
    if alpha_ranges is None:
        alpha_ranges = {
            "narrow": (-3, 3),
            "wide": (-5, 5),
        }

    vae.eval()
    n_samples = images.shape[0]
    n_dims = len(dimensions)

    results = {}

    for range_name, (alpha_min, alpha_max) in alpha_ranges.items():
        alphas = np.linspace(alpha_min, alpha_max, n_alphas).tolist()
        print(f"\nProcessing alpha range '{range_name}': [{alpha_min}, {alpha_max}]")

        all_grids = []

        for sample_idx in tqdm(range(n_samples), desc=f"Samples ({range_name})"):
            image = images[sample_idx:sample_idx+1]

            sample_grids = []
            for dim_idx in dimensions:
                dim_results = amplify_dimension(vae, image, dim_idx, alphas, device)
                sample_grids.append(dim_results)  # (n_alphas, C, H, W)

            sample_grids = torch.stack(sample_grids, dim=0)  # (n_dims, n_alphas, C, H, W)
            all_grids.append(sample_grids)

        all_grids = torch.stack(all_grids, dim=0)  # (n_samples, n_dims, n_alphas, C, H, W)
        results[range_name] = all_grids

    return results


def generate_comparison_grid(
    results: Dict[str, torch.Tensor],
    dimensions: List[int],
    alpha_ranges: Dict[str, Tuple[float, float]],
    n_alphas: int,
    sample_idx: int,
    dim_idx: int,
    output_path: Path,
    dimension_name: Optional[str] = None,
):
    """
    Generate side-by-side comparison of different alpha ranges for one dimension.

    Args:
        results: Dictionary from compare_alpha_ranges
        dimensions: List of dimension indices
        alpha_ranges: Dict of alpha range definitions
        n_alphas: Number of alpha values
        sample_idx: Which sample to visualize
        dim_idx: Which dimension index (0, 1, 2, ...)
        output_path: Path to save figure
        dimension_name: Optional name for the dimension
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    range_names = list(alpha_ranges.keys())
    n_ranges = len(range_names)

    fig, axes = plt.subplots(n_ranges, n_alphas, figsize=(n_alphas * 1.5, n_ranges * 1.8))

    if n_ranges == 1:
        axes = axes.reshape(1, -1)

    for row_idx, range_name in enumerate(range_names):
        alpha_min, alpha_max = alpha_ranges[range_name]
        alphas = np.linspace(alpha_min, alpha_max, n_alphas)

        grid = results[range_name]  # (n_samples, n_dims, n_alphas, C, H, W)

        for col_idx, alpha in enumerate(alphas):
            img = grid[sample_idx, dim_idx, col_idx].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)

            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis('off')

            # Add alpha label on top row
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f'{alpha:.1f}', fontsize=8)

        # Add range label on left
        axes[row_idx, 0].set_ylabel(f'{range_name}\n[{alpha_min},{alpha_max}]',
                                     fontsize=9, rotation=0, ha='right', va='center')

    dim_name = dimension_name or f"Dim {dimensions[dim_idx]}"
    plt.suptitle(f"Alpha Range Comparison - {dim_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_alpha_range_comparison(
    results: Dict[str, torch.Tensor],
    dimensions: List[int],
    alpha_ranges: Dict[str, Tuple[float, float]],
    n_alphas: int,
    output_dir: Path,
    n_samples_to_show: int = 4,
    dimension_names: Optional[List[str]] = None,
):
    """
    Save comparison visualizations for all dimensions and samples.

    Args:
        results: Dictionary from compare_alpha_ranges
        dimensions: List of dimension indices
        alpha_ranges: Dict of alpha range definitions
        n_alphas: Number of alpha values
        output_dir: Directory to save figures
        n_samples_to_show: Number of samples to visualize
        dimension_names: Optional names for each dimension
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_dims = len(dimensions)
    n_samples = min(n_samples_to_show, results[list(results.keys())[0]].shape[0])

    for dim_idx in range(n_dims):
        dim_name = dimension_names[dim_idx] if dimension_names else f"dim_{dimensions[dim_idx]}"

        for sample_idx in range(n_samples):
            filename = f"{dim_name}_sample_{sample_idx:02d}_comparison.png"
            generate_comparison_grid(
                results=results,
                dimensions=dimensions,
                alpha_ranges=alpha_ranges,
                n_alphas=n_alphas,
                sample_idx=sample_idx,
                dim_idx=dim_idx,
                output_path=output_dir / filename,
                dimension_name=dim_name,
            )

    # Also create a summary grid showing one sample per dimension
    range_names = list(alpha_ranges.keys())
    n_ranges = len(range_names)

    for range_name in range_names:
        alpha_min, alpha_max = alpha_ranges[range_name]
        alphas = np.linspace(alpha_min, alpha_max, n_alphas)

        fig, axes = plt.subplots(n_dims, n_alphas, figsize=(n_alphas * 1.2, n_dims * 1.2))

        if n_dims == 1:
            axes = axes.reshape(1, -1)

        grid = results[range_name]

        for dim_idx in range(n_dims):
            for col_idx, alpha in enumerate(alphas):
                img = grid[0, dim_idx, col_idx].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)

                axes[dim_idx, col_idx].imshow(img)
                axes[dim_idx, col_idx].axis('off')

                if dim_idx == 0:
                    axes[dim_idx, col_idx].set_title(f'{alpha:.1f}', fontsize=8)

            dim_name = dimension_names[dim_idx] if dimension_names else f"Dim {dimensions[dim_idx]}"
            axes[dim_idx, 0].set_ylabel(dim_name, fontsize=9, rotation=0, ha='right', va='center')

        plt.suptitle(f"Feature Amplification - Alpha Range [{alpha_min}, {alpha_max}]", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / f"summary_{range_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved alpha range comparison to {output_dir}")


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test alpha range comparison with dummy VAE.
    """
    print("Testing Alpha Range Comparison...")
    print("-" * 50)

    # Create dummy VAE
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
    print("Testing amplify_dimension with different ranges...")
    image = torch.rand(1, 3, 64, 64)

    alphas_narrow = np.linspace(-3, 3, 10).tolist()
    alphas_wide = np.linspace(-5, 5, 10).tolist()

    results_narrow = amplify_dimension(vae, image, dim_idx=0, alphas=alphas_narrow, device=device)
    results_wide = amplify_dimension(vae, image, dim_idx=0, alphas=alphas_wide, device=device)

    print(f"  Narrow range results shape: {results_narrow.shape}")
    print(f"  Wide range results shape: {results_wide.shape}")

    assert results_narrow.shape == (10, 3, 64, 64)
    assert results_wide.shape == (10, 3, 64, 64)
    print("  amplify_dimension passed!")

    # Test compare_alpha_ranges
    print("\nTesting compare_alpha_ranges...")
    images = torch.rand(4, 3, 64, 64)
    dimensions = [0, 10, 50]

    results = compare_alpha_ranges(
        vae=vae,
        images=images,
        dimensions=dimensions,
        device=device,
    )

    print(f"  Result keys: {list(results.keys())}")
    print(f"  Narrow shape: {results['narrow'].shape}")
    print(f"  Wide shape: {results['wide'].shape}")

    expected_shape = (4, 3, 10, 3, 64, 64)  # (n_samples, n_dims, n_alphas, C, H, W)
    assert results['narrow'].shape == expected_shape
    assert results['wide'].shape == expected_shape
    print("  compare_alpha_ranges passed!")

    print("\n" + "-" * 50)
    print("All alpha range tests passed!")

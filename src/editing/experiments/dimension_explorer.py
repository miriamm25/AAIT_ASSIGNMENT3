"""
Component: Manual Dimension Explorer
Reference: Task 2.1 Improvement - Experiment 3

Purpose:
    Explore top-N dimensions (e.g., top 20 by variance) and generate
    visualizations for manual inspection and cherry-picking.

Key implementation notes:
    - The assignment allows cherry-picking: "Feel free to cherry pick your best results"
    - Generate grids for many dimensions
    - Visually inspect to find semantically meaningful ones
    - Document what each dimension appears to control

From the plan:
    "Instead of just top-4, explore top-20 and manually select the best."
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


def find_top_variance_dimensions(
    vae: nn.Module,
    dataloader: DataLoader,
    n_dims: int = 20,
    n_images: int = 1000,
    device: torch.device = torch.device("cuda"),
) -> Tuple[List[int], torch.Tensor]:
    """
    Find latent dimensions with highest variance across the dataset.

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

    Args:
        vae: Trained VAE model
        image: Input image tensor (1, C, H, W)
        dim_idx: Index of latent dimension to amplify
        alphas: List of alpha values
        device: Computation device

    Returns:
        Tensor of generated images (len(alphas), C, H, W)
    """
    vae.eval()
    image = image.to(device)

    with torch.no_grad():
        mu, _ = vae.encode(image)

        results = []
        for alpha in alphas:
            z_modified = mu.clone()
            z_modified[0, dim_idx] = z_modified[0, dim_idx] + alpha

            if hasattr(vae, 'predict_variance') and vae.predict_variance:
                recon, _ = vae.decode(z_modified)
            else:
                recon = vae.decode(z_modified)

            results.append(recon)

    return torch.cat(results, dim=0)


def explore_top_n_dimensions(
    vae: nn.Module,
    dataloader: DataLoader,
    n_dims: int = 20,
    n_samples: int = 4,
    alphas: List[float] = None,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, torch.Tensor]:
    """
    Generate amplification grids for top-N variance dimensions.

    Reference: Task 2.1 Improvement Plan - Experiment 3
        "Explore top-20 dimensions and manually select the best"

    Args:
        vae: Trained VAE model
        dataloader: DataLoader for images
        n_dims: Number of dimensions to explore
        n_samples: Number of sample images per dimension
        alphas: Alpha values (default: linspace(-3, 3, 10))
        device: Computation device

    Returns:
        Dictionary with:
            - "dimensions": list of dimension indices
            - "variances": list of variance values
            - "grids": tensor of shape (n_dims, n_samples, n_alphas, C, H, W)
    """
    if alphas is None:
        alphas = np.linspace(-3, 3, 10).tolist()

    # Find top variance dimensions
    print("Finding top variance dimensions...")
    dimensions, variance = find_top_variance_dimensions(
        vae, dataloader, n_dims=n_dims, device=device
    )

    print(f"\nTop {n_dims} dimensions by variance:")
    for i, dim in enumerate(dimensions):
        print(f"  {i+1:2d}. Dimension {dim:3d}: variance = {variance[dim]:.2f}")

    # Get sample images
    print(f"\nCollecting {n_samples} sample images...")
    sample_images = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        sample_images.append(batch)
        if sum(x.shape[0] for x in sample_images) >= n_samples:
            break

    sample_images = torch.cat(sample_images, dim=0)[:n_samples].to(device)

    # Generate grids for all dimensions
    print(f"\nGenerating amplification grids for {n_dims} dimensions...")
    all_grids = []

    for dim_idx, dim in enumerate(tqdm(dimensions, desc="Dimensions")):
        dim_grids = []
        for sample_idx in range(n_samples):
            image = sample_images[sample_idx:sample_idx+1]
            grid = amplify_dimension(vae, image, dim, alphas, device)
            dim_grids.append(grid)  # (n_alphas, C, H, W)

        dim_grids = torch.stack(dim_grids, dim=0)  # (n_samples, n_alphas, C, H, W)
        all_grids.append(dim_grids)

    all_grids = torch.stack(all_grids, dim=0)  # (n_dims, n_samples, n_alphas, C, H, W)

    return {
        "dimensions": dimensions,
        "variances": variance[dimensions].tolist(),
        "grids": all_grids.cpu(),
        "alphas": alphas,
        "sample_images": sample_images.cpu(),
    }


def generate_dimension_report(
    results: Dict,
    output_dir: Path,
    create_individual_plots: bool = True,
):
    """
    Generate visualizations and report for dimension exploration.

    Args:
        results: Dictionary from explore_top_n_dimensions
        output_dir: Directory to save outputs
        create_individual_plots: Whether to create per-dimension plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions = results["dimensions"]
    variances = results["variances"]
    grids = results["grids"]
    alphas = results["alphas"]
    n_dims = len(dimensions)
    n_samples = grids.shape[1]
    n_alphas = len(alphas)

    # Save dimension ranking as JSON
    ranking = {
        "dimensions": [
            {
                "rank": i + 1,
                "dimension": dim,
                "variance": var,
                "suggested_label": f"Feature_{i+1}_to_investigate"
            }
            for i, (dim, var) in enumerate(zip(dimensions, variances))
        ],
        "total_dimensions_explored": n_dims,
        "alphas_used": alphas,
        "n_samples_per_dimension": n_samples,
    }

    with open(output_dir / "dimension_ranking.json", "w") as f:
        json.dump(ranking, f, indent=2)

    # Create individual dimension plots
    if create_individual_plots:
        print(f"\nGenerating individual plots for {n_dims} dimensions...")

        for dim_idx, dim in enumerate(tqdm(dimensions, desc="Creating plots")):
            fig, axes = plt.subplots(n_samples, n_alphas, figsize=(n_alphas * 1.2, n_samples * 1.2))

            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for sample_idx in range(n_samples):
                for alpha_idx, alpha in enumerate(alphas):
                    img = grids[dim_idx, sample_idx, alpha_idx].permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)

                    axes[sample_idx, alpha_idx].imshow(img)
                    axes[sample_idx, alpha_idx].axis('off')

                    if sample_idx == 0:
                        axes[sample_idx, alpha_idx].set_title(f'{alpha:.1f}', fontsize=8)

            var = variances[dim_idx]
            plt.suptitle(f"Dimension {dim} (Rank {dim_idx+1}, Var={var:.1f})", fontsize=11)
            plt.tight_layout()
            plt.savefig(output_dir / f"dim_{dim:03d}_rank_{dim_idx+1:02d}.png",
                       dpi=150, bbox_inches='tight')
            plt.close()

    # Create overview grid (first sample of each dimension)
    print("\nGenerating overview grid...")
    n_cols = min(10, n_alphas)
    fig, axes = plt.subplots(n_dims, n_cols, figsize=(n_cols * 1.0, n_dims * 0.8))

    for dim_idx in range(n_dims):
        # Select evenly spaced alphas for overview
        alpha_indices = np.linspace(0, n_alphas - 1, n_cols, dtype=int)

        for col_idx, alpha_idx in enumerate(alpha_indices):
            img = grids[dim_idx, 0, alpha_idx].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            axes[dim_idx, col_idx].imshow(img)
            axes[dim_idx, col_idx].axis('off')

            if dim_idx == 0:
                axes[dim_idx, col_idx].set_title(f'{alphas[alpha_idx]:.1f}', fontsize=7)

        axes[dim_idx, 0].set_ylabel(f'D{dimensions[dim_idx]}', fontsize=7,
                                     rotation=0, ha='right', va='center')

    plt.suptitle("Dimension Explorer - Overview (First Sample per Dimension)", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "overview_all_dimensions.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Create variance plot
    print("Generating variance plot...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(n_dims), variances)
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f'D{d}' for d in dimensions], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Variance')
    ax.set_title(f'Variance of Top {n_dims} Latent Dimensions')
    plt.tight_layout()
    plt.savefig(output_dir / "variance_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nDimension exploration report saved to {output_dir}")
    print(f"  - dimension_ranking.json: Full ranking data")
    print(f"  - overview_all_dimensions.png: Quick comparison")
    print(f"  - variance_ranking.png: Variance distribution")
    if create_individual_plots:
        print(f"  - dim_XXX_rank_YY.png: Individual dimension grids")


def create_selection_template(
    results: Dict,
    output_path: Path,
):
    """
    Create a template file for manual dimension labeling.

    Args:
        results: Dictionary from explore_top_n_dimensions
        output_path: Path to save the template
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dimensions = results["dimensions"]
    variances = results["variances"]

    template = """# Dimension Selection Template
# Fill in your observations for each dimension after visual inspection.
# Mark your final 4 selections with [SELECTED] at the end.

"""

    for i, (dim, var) in enumerate(zip(dimensions, variances)):
        template += f"""
## Rank {i+1}: Dimension {dim} (Variance: {var:.2f})
- **Observed Effect**: [Describe what changes when alpha varies]
- **Semantic Interpretation**: [What facial feature does this control?]
- **Quality**: [Good/Medium/Poor] - [Explain why]
- **Notes**: [Any other observations]
"""

    template += """

---
# FINAL SELECTION

Based on the visual inspection above, select the 4 best dimensions:

1. Dimension ___: [Reason]
2. Dimension ___: [Reason]
3. Dimension ___: [Reason]
4. Dimension ___: [Reason]

# Observations

[Add any general notes about the latent space structure]
"""

    with open(output_path, "w") as f:
        f.write(template)

    print(f"Selection template saved to {output_path}")


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test dimension explorer with dummy VAE.
    """
    print("Testing Dimension Explorer...")
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

    from torch.utils.data import TensorDataset

    device = torch.device("cpu")
    vae = DummyVAE().to(device)

    # Create dummy dataloader
    dummy_images = torch.rand(100, 3, 64, 64)
    dummy_dataset = TensorDataset(dummy_images)
    dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16)

    # Test find_top_variance_dimensions
    print("Testing find_top_variance_dimensions...")
    dimensions, variance = find_top_variance_dimensions(
        vae, dataloader, n_dims=10, n_images=50, device=device
    )
    print(f"  Found {len(dimensions)} dimensions")
    print(f"  First 5 dimensions: {dimensions[:5]}")

    # Test explore_top_n_dimensions
    print("\nTesting explore_top_n_dimensions...")
    results = explore_top_n_dimensions(
        vae=vae,
        dataloader=dataloader,
        n_dims=5,
        n_samples=2,
        device=device,
    )

    print(f"  Dimensions: {results['dimensions']}")
    print(f"  Variances: {[f'{v:.2f}' for v in results['variances']]}")
    print(f"  Grids shape: {results['grids'].shape}")

    expected_shape = (5, 2, 10, 3, 64, 64)  # (n_dims, n_samples, n_alphas, C, H, W)
    assert results['grids'].shape == expected_shape

    print("\nAll dimension explorer tests passed!")

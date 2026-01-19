"""
Component: Visualization Utilities
Reference: Task1_VAE_Guide.md, AAIT_Assignment_3.pdf

Purpose:
    Generate visualizations for VAE evaluation:
    - Latent space interpolation
    - Temperature-based sampling
    - Reconstruction comparisons
    - Loss curves

Key implementation notes:
    - Interpolation: z_i = z_1 * α + (1 - α) * z_2
    - Temperature sampling: z ~ N(0, τ²I) where τ is temperature
    - Reference: AAIT_Assignment_3.pdf visualization requirements

Required outputs (from assignment):
    1. Interpolation grid: 2 images → interpolated latents → reconstructions
    2. Temperature sampling grid: temps [0.2, 0.5, 1.0, 1.5] × 8 samples
    3. Reconstruction examples: Original vs Reconstructed
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.vae import VAE


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for plotting."""
    # Ensure on CPU, detach from graph, convert to numpy
    x = x.detach().cpu()

    # Handle different shapes
    if x.dim() == 4:
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        # (C, H, W) -> (H, W, C)
        x = x.permute(1, 2, 0)

    # Clamp to [0, 1] and convert
    x = x.clamp(0, 1).numpy()

    return x


def plot_interpolation(
    vae: "VAE",
    img1: torch.Tensor,
    img2: torch.Tensor,
    n_steps: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Latent Space Interpolation",
) -> plt.Figure:
    """
    Plot interpolation between two images in latent space.

    Reference: AAIT_Assignment_3.pdf
        "z_i = z_1 · α + (1 - α) · z_2"

    Args:
        vae: Trained VAE model
        img1: First image, shape (1, C, H, W)
        img2: Second image, shape (1, C, H, W)
        n_steps: Number of interpolation steps
        save_path: Optional path to save the figure
        title: Figure title

    Returns:
        Matplotlib figure
    """
    vae.eval()
    device = next(vae.parameters()).device

    # Move images to device
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Get interpolations
    with torch.no_grad():
        interpolations = vae.interpolate(img1, img2, num_steps=n_steps)

    # Convert to numpy
    interpolations = tensor_to_numpy(interpolations)
    img1_np = tensor_to_numpy(img1)[0]
    img2_np = tensor_to_numpy(img2)[0]

    # Create figure
    fig, axes = plt.subplots(1, n_steps + 2, figsize=(2 * (n_steps + 2), 2))

    # Plot original images
    axes[0].imshow(img1_np)
    axes[0].set_title("Image 1")
    axes[0].axis("off")

    axes[-1].imshow(img2_np)
    axes[-1].set_title("Image 2")
    axes[-1].axis("off")

    # Plot interpolations
    for i in range(n_steps):
        axes[i + 1].imshow(interpolations[i])
        axes[i + 1].set_title(f"α={i / (n_steps - 1):.1f}")
        axes[i + 1].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved interpolation plot to {save_path}")

    return fig


def plot_temperature_samples(
    vae: "VAE",
    temperatures: List[float] = [0.2, 0.5, 1.0, 1.5],
    n_samples: int = 8,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Temperature Sampling",
) -> plt.Figure:
    """
    Plot samples at different temperatures.

    Reference: Task1_VAE_Guide.md
        "Sample z ~ N(0, τ²I) where τ is the temperature"
        "temps [0.2, 0.5, 1.0, 1.5]"

    Lower temperature -> more conservative samples (closer to mean)
    Higher temperature -> more diverse samples (more variance)

    Args:
        vae: Trained VAE model
        temperatures: List of temperatures to sample at
        n_samples: Number of samples per temperature
        save_path: Optional path to save the figure
        title: Figure title

    Returns:
        Matplotlib figure
    """
    vae.eval()
    device = next(vae.parameters()).device

    # Generate samples at each temperature
    all_samples = []
    for temp in temperatures:
        with torch.no_grad():
            samples = vae.sample(n_samples, temperature=temp, device=device)
        all_samples.append(tensor_to_numpy(samples))

    # Create figure
    fig, axes = plt.subplots(
        len(temperatures),
        n_samples,
        figsize=(2 * n_samples, 2 * len(temperatures)),
    )

    for i, (temp, samples) in enumerate(zip(temperatures, all_samples)):
        for j in range(n_samples):
            ax = axes[i, j] if len(temperatures) > 1 else axes[j]
            ax.imshow(samples[j])
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(f"τ={temp}", fontsize=12)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved temperature samples to {save_path}")

    return fig


def plot_reconstructions(
    vae: "VAE",
    images: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Reconstructions",
) -> plt.Figure:
    """
    Plot original images and their reconstructions side by side.

    Reference: AAIT_Assignment_3.pdf
        "Reconstruction examples: Original vs Reconstructed"

    Args:
        vae: Trained VAE model
        images: Batch of images, shape (B, C, H, W)
        save_path: Optional path to save the figure
        title: Figure title

    Returns:
        Matplotlib figure
    """
    vae.eval()
    device = next(vae.parameters()).device

    # Move images to device and get reconstructions
    images = images.to(device)
    with torch.no_grad():
        recons = vae.reconstruct(images)

    # Convert to numpy
    images_np = tensor_to_numpy(images)
    recons_np = tensor_to_numpy(recons)

    n_images = images.shape[0]

    # Create figure
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))

    for i in range(n_images):
        # Original
        axes[0, i].imshow(images_np[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12)

        # Reconstruction
        axes[1, i].imshow(recons_np[i])
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Recon", fontsize=12)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved reconstructions to {save_path}")

    return fig


def plot_loss_curves(
    train_losses: dict,
    val_losses: dict,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Reference: AAIT_Assignment_3.pdf
        "Loss plots: Train/Val Gaussian NLL over epochs"
        "KL plot: Train/Val KL Divergence over epochs"

    Args:
        train_losses: Dictionary with keys 'recon', 'kl', 'total'
            Each value is a list of losses per epoch
        val_losses: Same structure as train_losses
        save_path: Optional path to save the figure
        title: Figure title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(train_losses.get("total", [])) + 1)

    # Plot reconstruction loss
    if "recon" in train_losses:
        axes[0].plot(epochs, train_losses["recon"], label="Train", marker="o", markersize=3)
        if "recon" in val_losses:
            axes[0].plot(epochs, val_losses["recon"], label="Val", marker="s", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Reconstruction Loss")
        axes[0].set_title("Reconstruction Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Plot KL loss
    if "kl" in train_losses:
        axes[1].plot(epochs, train_losses["kl"], label="Train", marker="o", markersize=3)
        if "kl" in val_losses:
            axes[1].plot(epochs, val_losses["kl"], label="Val", marker="s", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("KL Divergence")
        axes[1].set_title("KL Divergence")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot total loss
    if "total" in train_losses:
        axes[2].plot(epochs, train_losses["total"], label="Train", marker="o", markersize=3)
        if "total" in val_losses:
            axes[2].plot(epochs, val_losses["total"], label="Val", marker="s", markersize=3)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Total Loss (ELBO)")
        axes[2].set_title("Total Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved loss curves to {save_path}")

    return fig


def save_sample_grid(
    vae: "VAE",
    n_samples: int = 64,
    temperature: float = 1.0,
    save_path: Optional[Union[str, Path]] = None,
    nrow: int = 8,
) -> torch.Tensor:
    """
    Generate and save a grid of samples.

    Args:
        vae: Trained VAE model
        n_samples: Total number of samples
        temperature: Sampling temperature
        save_path: Optional path to save the image
        nrow: Number of images per row

    Returns:
        Grid tensor
    """
    from torchvision.utils import make_grid, save_image

    vae.eval()
    device = next(vae.parameters()).device

    with torch.no_grad():
        samples = vae.sample(n_samples, temperature=temperature, device=device)

    # Make grid
    grid = make_grid(samples, nrow=nrow, padding=2, normalize=False)

    if save_path:
        save_image(grid, save_path)
        print(f"Saved sample grid to {save_path}")

    return grid


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test visualization functions with a dummy VAE.
    """
    print("Testing Visualization Utilities...")
    print("-" * 50)
    print("Note: This requires a trained VAE model.")
    print("Visualization functions will be tested during training.")
    print("-" * 50)
    print("✓ Visualization module loaded successfully!")

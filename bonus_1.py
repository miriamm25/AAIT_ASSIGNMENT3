"""
Bonus 1: BPD Optimization Metric
Reference: AAIT_Assignment_3.pdf - Bonus Structure

Purpose:
    Compute the harmonic mean between normalized NLL and KL Divergence on the test set.
    This metric balances reconstruction quality (NLL) with latent space regularization (KL).

Normalization:
    - Normalized NLL = NLL / number_of_pixels (D = H × W × C)
    - Normalized KL = KL / number_of_latent_components (d = latent_dim)

Harmonic Mean:
    H = 2 * (norm_NLL * norm_KL) / (norm_NLL + norm_KL)

Usage:
    python bonus_1.py --checkpoint outputs/checkpoints/best_model.pt
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json

# Add src directory to path
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir / "src"
sys.path.insert(0, str(src_dir))

from models.vae import VAE
from data.celeba import get_celeba_dataloaders
from losses.kl_divergence import kl_divergence_loss
from losses.reconstruction import mse_loss, gaussian_nll_loss


def compute_normalized_metrics(
    vae: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    predict_variance: bool = False,
) -> dict:
    """
    Compute normalized NLL and KL divergence on a dataset.

    Reference: AAIT_Assignment_3.pdf - Bonus Structure
        "Normalized scores will divide the KL Div by the number of latent components,
        and the NLL by the number of pixels."

    Args:
        vae: Trained VAE model
        dataloader: DataLoader for evaluation
        device: Computation device
        predict_variance: Whether VAE predicts variance

    Returns:
        Dictionary with metrics
    """
    vae.eval()

    total_nll = 0.0
    total_kl = 0.0
    total_samples = 0

    # Get dimensions
    latent_dim = vae.latent_dim
    image_size = vae.image_size
    num_pixels = 3 * image_size * image_size  # D = H × W × C

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing metrics"):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Forward pass
            output = vae(batch)

            # Compute reconstruction loss (NLL)
            if predict_variance:
                # Gaussian NLL with predicted variance
                nll = gaussian_nll_loss(
                    batch, output["mu_x"], output["log_var_x"], reduction="sum"
                )
            else:
                # MSE (equivalent to Gaussian NLL with σ²=1, ignoring constants)
                # For proper NLL: NLL = 0.5 * MSE + 0.5 * D * log(2π)
                mse = ((batch - output["mu_x"]) ** 2).sum()
                # Add the constant term for proper Gaussian NLL
                nll = 0.5 * mse + 0.5 * batch_size * num_pixels * torch.log(
                    torch.tensor(2 * 3.14159265359)
                )

            # Compute KL divergence (sum over batch and latent dims)
            # KL = 0.5 * sum(σ² + μ² - 1 - log(σ²))
            mu = output["mu_z"]
            log_var = output["log_var_z"]
            kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            total_nll += nll.item()
            total_kl += kl.item()
            total_samples += batch_size

    # Compute averages per sample
    avg_nll = total_nll / total_samples
    avg_kl = total_kl / total_samples

    # Normalize
    # Normalized NLL = NLL / num_pixels
    norm_nll = avg_nll / num_pixels

    # Normalized KL = KL / latent_dim
    norm_kl = avg_kl / latent_dim

    # Compute harmonic mean
    # H = 2 * (a * b) / (a + b)
    if norm_nll + norm_kl > 0:
        harmonic_mean = 2 * (norm_nll * norm_kl) / (norm_nll + norm_kl)
    else:
        harmonic_mean = 0.0

    # Also compute BPD for reference
    # BPD = -log p(x) / (D * log(2)) ≈ (NLL + KL) / (D * log(2))
    elbo = avg_nll + avg_kl
    bpd = elbo / (num_pixels * 0.693147)  # log(2) ≈ 0.693147

    return {
        "avg_nll": avg_nll,
        "avg_kl": avg_kl,
        "norm_nll": norm_nll,
        "norm_kl": norm_kl,
        "harmonic_mean": harmonic_mean,
        "bpd": bpd,
        "total_samples": total_samples,
        "num_pixels": num_pixels,
        "latent_dim": latent_dim,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute bonus BPD metric (harmonic mean of normalized NLL & KL)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Path to CelebA data root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/bonus_1_results.json",
        help="Path to save results",
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    # Get model config
    model_config = config.get("model", {})
    latent_dim = model_config.get("latent_dim", 256)
    base_channels = model_config.get("base_channels", 64)
    blocks_per_level = model_config.get("blocks_per_level", 2)
    image_size = model_config.get("image_size", 64)
    predict_variance = model_config.get("predict_variance", False)

    print(f"Model config:")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Image size: {image_size}")
    print(f"  Predict variance: {predict_variance}")

    # Create model
    vae = VAE(
        in_channels=3,
        base_channels=base_channels,
        latent_dim=latent_dim,
        blocks_per_level=blocks_per_level,
        image_size=image_size,
        predict_variance=predict_variance,
    ).to(device)

    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()
    print("Model loaded successfully!")

    # Load test dataloader
    print(f"\nLoading CelebA test set...")
    _, _, test_loader = get_celeba_dataloaders(
        root=args.data_root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=4,
        download=False,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing Bonus 1 Metrics on Test Set")
    print("=" * 60)

    results = compute_normalized_metrics(
        vae=vae,
        dataloader=test_loader,
        device=device,
        predict_variance=predict_variance,
    )

    # Print results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Total test samples: {results['total_samples']}")
    print(f"Number of pixels (D): {results['num_pixels']}")
    print(f"Latent dimension (d): {results['latent_dim']}")
    print()
    print(f"Average NLL (per sample): {results['avg_nll']:.4f}")
    print(f"Average KL (per sample): {results['avg_kl']:.4f}")
    print()
    print(f"Normalized NLL (NLL/D): {results['norm_nll']:.6f}")
    print(f"Normalized KL (KL/d): {results['norm_kl']:.6f}")
    print()
    print(f">>> HARMONIC MEAN: {results['harmonic_mean']:.6f} <<<")
    print()
    print(f"BPD (for reference): {results['bpd']:.4f}")
    print("-" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Bonus 1 metric computation complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()

#!/bin/bash
# Evaluation script for VAE on CelebA
# Generates all required visualizations and metrics
# Reference: AAIT Assignment 3, Task 1

# Exit on error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Default checkpoint path
CHECKPOINT="${1:-outputs/checkpoints/best_model.pt}"

echo "=============================================="
echo "VAE CelebA Evaluation"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please provide a valid checkpoint path as argument"
    echo "Usage: ./evaluate.sh [checkpoint_path]"
    exit 1
fi

# Python evaluation script
EVAL_SCRIPT=$(cat << 'EOF'
import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vae import VAE
from src.data.celeba import get_celeba_dataloaders
from src.losses.kl_divergence import kl_divergence_loss
from src.losses.reconstruction import mse_loss, gaussian_nll_loss
from src.utils.metrics import compute_bpd
from src.utils.visualization import (
    plot_reconstructions,
    plot_temperature_samples,
    plot_interpolation,
    save_sample_grid,
)


def main():
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/best_model.pt"

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = VAE(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
        latent_dim=config["model"]["latent_dim"],
        blocks_per_level=config["model"]["blocks_per_level"],
        image_size=config["model"]["image_size"],
        predict_variance=config["model"]["predict_variance"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_celeba_dataloaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        download=False,
    )

    # Create output directory
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get test batch
    test_batch = next(iter(test_loader))[:16].to(device)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Reconstructions
    plot_reconstructions(
        model,
        test_batch[:8],
        save_path=output_dir / "reconstructions.png",
        title="Test Set Reconstructions",
    )

    # 2. Temperature sampling
    plot_temperature_samples(
        model,
        temperatures=config["visualization"]["temperatures"],
        n_samples=config["visualization"]["n_samples_per_temp"],
        save_path=output_dir / "temperature_samples.png",
        title="Temperature Sampling",
    )

    # 3. Interpolation
    plot_interpolation(
        model,
        test_batch[:1],
        test_batch[1:2],
        n_steps=config["visualization"]["n_interpolation_steps"],
        save_path=output_dir / "interpolation.png",
        title="Latent Space Interpolation",
    )

    # 4. Sample grid
    save_sample_grid(
        model,
        n_samples=64,
        temperature=1.0,
        save_path=output_dir / "samples.png",
    )

    # Compute test metrics
    print("\nComputing test metrics...")
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)

            if config["model"]["predict_variance"]:
                recon = gaussian_nll_loss(batch, output["mu_x"], output["log_var_x"])
            else:
                recon = mse_loss(batch, output["mu_x"])

            kl = kl_divergence_loss(output["mu_z"], output["log_var_z"])

            total_recon += recon.item()
            total_kl += kl.item()
            num_batches += 1

    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    avg_total = avg_recon + avg_kl

    # Compute BPD
    num_dims = 3 * config["model"]["image_size"] ** 2
    bpd = compute_bpd(torch.tensor(avg_total), num_dims).item()

    print("\n" + "=" * 40)
    print("Test Set Results:")
    print("=" * 40)
    print(f"  Reconstruction Loss: {avg_recon:.4f}")
    print(f"  KL Divergence: {avg_kl:.4f}")
    print(f"  Total Loss: {avg_total:.4f}")
    print(f"  BPD: {bpd:.4f}")
    print("=" * 40)

    print(f"\nVisualizations saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
EOF
)

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv for package management"
    uv run python -c "$EVAL_SCRIPT" "$CHECKPOINT"
else
    echo "Using python directly"
    python -c "$EVAL_SCRIPT" "$CHECKPOINT"
fi

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "=============================================="

"""
Component: Training Script
Reference: AAIT_Assignment_3.pdf, Task1_VAE_Guide.md

Purpose:
    Main training loop for the VAE model.
    Handles training, validation, checkpointing, logging, and visualization.

Key implementation notes:
    - Use Adam optimizer with lr=1e-4 (Task1_VAE_Guide.md Step 6)
    - KL annealing to prevent posterior collapse
    - Optional perceptual loss for sharper reconstructions
    - Mixed precision training for efficiency
    - Reference: Task1_VAE_Guide.md Steps 6-9

Teacher's advice incorporated:
    - "Use the Adam optimizer with a learning rate of 1e-4"
    - "batch_size 16-32 can help prevent mode collapse"
    - "KL annealing can help prevent posterior collapse"
    - "Adding a perceptual loss can make training converge much faster"
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

# Add src directory to path for imports
import os.path as osp
_src_dir = osp.dirname(osp.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from models.vae import VAE
from data.celeba import get_celeba_dataloaders
from losses.kl_divergence import kl_divergence_loss
from losses.reconstruction import mse_loss, gaussian_nll_loss
from losses.perceptual import PerceptualLoss
from utils.kl_annealing import KLAnnealer
from utils.metrics import compute_bpd, compute_elbo
from utils.visualization import (
    plot_loss_curves,
    plot_reconstructions,
    plot_temperature_samples,
    plot_interpolation,
    save_sample_grid,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_device() -> torch.device:
    """Set up the compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_model(config: dict, device: torch.device) -> VAE:
    """Create and initialize the VAE model."""
    model = VAE(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
        latent_dim=config["model"]["latent_dim"],
        blocks_per_level=config["model"]["blocks_per_level"],
        image_size=config["model"]["image_size"],
        predict_variance=config["model"]["predict_variance"],
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    return model


def create_optimizer(model: VAE, config: dict) -> optim.Optimizer:
    """Create optimizer."""
    # Reference: Task1_VAE_Guide.md Step 6
    # "Use the Adam optimizer with a learning rate of 1e-4"
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    return optimizer


def train_epoch(
    model: VAE,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    perceptual_loss_fn: Optional[PerceptualLoss],
    kl_weight: float,
    config: dict,
    device: torch.device,
    scaler: Optional[GradScaler],
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Returns:
        Dictionary of average losses for the epoch
    """
    model.train()
    predict_variance = config["model"]["predict_variance"]

    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_perceptual_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")

    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Mixed precision forward pass
        use_amp = config["training"]["use_amp"] and device.type == "cuda"
        amp_dtype = getattr(torch, config["training"]["amp_dtype"])

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            # Forward pass
            output = model(batch)

            # Reconstruction loss
            if predict_variance:
                recon_loss = gaussian_nll_loss(batch, output["mu_x"], output["log_var_x"])
            else:
                recon_loss = mse_loss(batch, output["mu_x"])

            # KL divergence loss
            kl_loss = kl_divergence_loss(output["mu_z"], output["log_var_z"])

            # Total loss with KL annealing
            loss = recon_loss + kl_weight * kl_loss

            # Optional perceptual loss
            perceptual_loss = torch.tensor(0.0, device=device)
            if perceptual_loss_fn is not None and config["loss"]["use_perceptual"]:
                perceptual_loss = perceptual_loss_fn(batch, output["mu_x"])
                loss = loss + config["loss"]["perceptual_weight"] * perceptual_loss

        # Backward pass with gradient scaling
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if config["training"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config["training"]["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()

        # Accumulate losses
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_perceptual_loss += perceptual_loss.item()
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "recon": f"{recon_loss.item():.2f}",
            "kl": f"{kl_loss.item():.2f}",
            "kl_w": f"{kl_weight:.2f}",
        })

    # Average losses
    return {
        "recon": total_recon_loss / num_batches,
        "kl": total_kl_loss / num_batches,
        "perceptual": total_perceptual_loss / num_batches,
        "total": total_loss / num_batches,
    }


@torch.no_grad()
def validate_epoch(
    model: VAE,
    val_loader: DataLoader,
    perceptual_loss_fn: Optional[PerceptualLoss],
    kl_weight: float,
    config: dict,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Returns:
        Dictionary of average losses for the epoch
    """
    model.eval()
    predict_variance = config["model"]["predict_variance"]

    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_perceptual_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]")

    for batch in pbar:
        batch = batch.to(device)

        # Forward pass
        output = model(batch)

        # Reconstruction loss
        if predict_variance:
            recon_loss = gaussian_nll_loss(batch, output["mu_x"], output["log_var_x"])
        else:
            recon_loss = mse_loss(batch, output["mu_x"])

        # KL divergence loss
        kl_loss = kl_divergence_loss(output["mu_z"], output["log_var_z"])

        # Total loss
        loss = recon_loss + kl_weight * kl_loss

        # Optional perceptual loss
        perceptual_loss = torch.tensor(0.0, device=device)
        if perceptual_loss_fn is not None and config["loss"]["use_perceptual"]:
            perceptual_loss = perceptual_loss_fn(batch, output["mu_x"])
            loss = loss + config["loss"]["perceptual_weight"] * perceptual_loss

        # Accumulate losses
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_perceptual_loss += perceptual_loss.item()
        total_loss += loss.item()
        num_batches += 1

    # Average losses
    return {
        "recon": total_recon_loss / num_batches,
        "kl": total_kl_loss / num_batches,
        "perceptual": total_perceptual_loss / num_batches,
        "total": total_loss / num_batches,
    }


def save_checkpoint(
    model: VAE,
    optimizer: optim.Optimizer,
    epoch: int,
    train_losses: dict,
    val_losses: dict,
    config: dict,
    save_path: str,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": config,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model: VAE,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[int, dict, dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        checkpoint["epoch"],
        checkpoint["train_losses"],
        checkpoint["val_losses"],
    )


def generate_visualizations(
    model: VAE,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    epoch: int,
    output_dir: Path,
):
    """Generate and save visualizations."""
    model.eval()

    # Get a batch of validation images
    val_batch = next(iter(val_loader))[:8].to(device)

    # Reconstructions
    plot_reconstructions(
        model,
        val_batch,
        save_path=output_dir / f"reconstructions_epoch_{epoch + 1}.png",
        title=f"Reconstructions - Epoch {epoch + 1}",
    )

    # Temperature sampling
    plot_temperature_samples(
        model,
        temperatures=config["visualization"]["temperatures"],
        n_samples=config["visualization"]["n_samples_per_temp"],
        save_path=output_dir / f"temperature_samples_epoch_{epoch + 1}.png",
        title=f"Temperature Sampling - Epoch {epoch + 1}",
    )

    # Interpolation (using first two images)
    if val_batch.shape[0] >= 2:
        plot_interpolation(
            model,
            val_batch[:1],
            val_batch[1:2],
            n_steps=config["visualization"]["n_interpolation_steps"],
            save_path=output_dir / f"interpolation_epoch_{epoch + 1}.png",
            title=f"Latent Space Interpolation - Epoch {epoch + 1}",
        )

    # Sample grid
    save_sample_grid(
        model,
        n_samples=64,
        temperature=1.0,
        save_path=output_dir / f"samples_epoch_{epoch + 1}.png",
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train VAE on CelebA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print("Loaded configuration:")
    print(f"  Latent dim: {config['model']['latent_dim']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['lr']}")
    print(f"  KL warmup epochs: {config['training']['kl_warmup_epochs']}")

    # Set seed for reproducibility
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])

    # Setup device
    device = setup_device()

    # Create output directories
    checkpoint_dir = Path(config["checkpoint"]["save_dir"])
    plot_dir = Path(config["logging"]["plot_dir"])
    log_dir = Path(config["logging"]["log_dir"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create data loaders
    print("\nLoading CelebA dataset...")
    train_loader, val_loader, test_loader = get_celeba_dataloaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        download=config["data"]["download"],
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print("\nCreating VAE model...")
    model = create_model(config, device)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Create perceptual loss (if enabled)
    perceptual_loss_fn = None
    if config["loss"]["use_perceptual"]:
        print("Loading perceptual loss (VGG16)...")
        perceptual_loss_fn = PerceptualLoss().to(device)
        perceptual_loss_fn.eval()

    # Create KL annealer
    kl_annealer = KLAnnealer(
        warmup_epochs=config["training"]["kl_warmup_epochs"],
    )
    print(f"KL Annealer: {kl_annealer}")

    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config["training"]["use_amp"] and device.type == "cuda" else None

    # Initialize loss history
    train_history = {"recon": [], "kl": [], "perceptual": [], "total": []}
    val_history = {"recon": [], "kl": [], "perceptual": [], "total": []}
    start_epoch = 0
    best_val_loss = float("inf")

    # Resume from checkpoint if specified
    if args.resume or config["checkpoint"]["resume"]:
        checkpoint_path = args.resume or config["checkpoint"]["resume"]
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        start_epoch, train_history, val_history = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )
        print(f"Resuming from epoch {start_epoch + 1}")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Get KL weight for this epoch
        kl_weight = kl_annealer.get_weight(epoch)

        # Train
        train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            perceptual_loss_fn=perceptual_loss_fn,
            kl_weight=kl_weight,
            config=config,
            device=device,
            scaler=scaler,
            epoch=epoch,
        )

        # Validate
        val_losses = validate_epoch(
            model=model,
            val_loader=val_loader,
            perceptual_loss_fn=perceptual_loss_fn,
            kl_weight=kl_weight,
            config=config,
            device=device,
            epoch=epoch,
        )

        # Update history
        for key in train_history:
            train_history[key].append(train_losses[key])
            val_history[key].append(val_losses[key])

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print(f"  Train - Recon: {train_losses['recon']:.2f}, KL: {train_losses['kl']:.2f}, Total: {train_losses['total']:.2f}")
        print(f"  Val   - Recon: {val_losses['recon']:.2f}, KL: {val_losses['kl']:.2f}, Total: {val_losses['total']:.2f}")
        print(f"  KL weight: {kl_weight:.3f}")

        # Calculate and print BPD
        num_dims = 3 * config["model"]["image_size"] ** 2
        train_bpd = compute_bpd(torch.tensor(train_losses["total"]), num_dims)
        val_bpd = compute_bpd(torch.tensor(val_losses["total"]), num_dims)
        print(f"  BPD - Train: {train_bpd:.4f}, Val: {val_bpd:.4f}")

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_checkpoint(
                model, optimizer, epoch, train_history, val_history, config,
                checkpoint_dir / "best_model.pt"
            )

        # Save periodic checkpoint
        if (epoch + 1) % config["checkpoint"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, train_history, val_history, config,
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            )

        # Generate visualizations
        if (epoch + 1) % config["logging"]["visualize_every"] == 0:
            print("Generating visualizations...")
            generate_visualizations(
                model, val_loader, config, device, epoch, plot_dir
            )

            # Save loss curves
            plot_loss_curves(
                train_history, val_history,
                save_path=plot_dir / "loss_curves.png",
                title="Training Progress"
            )

    # Final save
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    save_checkpoint(
        model, optimizer, config["training"]["epochs"] - 1,
        train_history, val_history, config,
        checkpoint_dir / "final_model.pt"
    )

    # Generate final visualizations
    print("\nGenerating final visualizations...")
    generate_visualizations(
        model, val_loader, config, device,
        config["training"]["epochs"] - 1, plot_dir
    )

    plot_loss_curves(
        train_history, val_history,
        save_path=plot_dir / "final_loss_curves.png",
        title="Final Training Progress"
    )

    # Save training history
    history = {
        "train": train_history,
        "val": val_history,
        "config": config,
    }
    with open(log_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {log_dir / 'training_history.json'}")

    print("\nDone!")


if __name__ == "__main__":
    main()

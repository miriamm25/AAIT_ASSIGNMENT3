"""
Component: VAE Decoder
Reference: Task1_VAE_Guide.md Step 5

Purpose:
    Decode latent vectors back to image space.
    Input: latent vector of size latent_dim
    Output: Reconstructed image (and optionally log_var for variance-predicting decoder)

Key implementation notes:
    - Use nn.Linear to map from latent to 8×8×(base_channels*8)
    - Upsample 3 times: 8 -> 16 -> 32 -> 64
    - Channels decrease: 8*base -> 4*base -> 2*base -> base
    - Two modes: isotropic (MSE loss) or variance-predicting (Gaussian NLL)
    - Reference: Task1_VAE_Guide.md Step 5

Teacher's advice incorporated:
    - "Isotropic decoder is simpler and recommended for learning"
    - "If predicting variance, clamp log_var to [-10, 10] to prevent instability"

Architecture (for base_channels=64, latent_dim=256):
    Input: (B, 256)
    -> Linear: (B, 32768)
    -> Reshape: (B, 512, 8, 8)
    -> Upsample + blocks: (B, 256, 16, 16)
    -> Upsample + blocks: (B, 128, 32, 32)
    -> Upsample + blocks: (B, 64, 64, 64)
    -> Final conv: (B, 3, 64, 64) or (B, 6, 64, 64) if predicting variance
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from .blocks import ConvNormAct, ResidualBottleneck, Upsample


class Decoder(nn.Module):
    """
    VAE Decoder that maps latent vectors back to image space.

    Reference: Task1_VAE_Guide.md Step 5
        "The decoder reconstructs images from latent vectors"

    Args:
        out_channels: Number of output channels (default 3 for RGB)
        base_channels: Base number of channels (default 64)
        latent_dim: Dimension of latent space (default 256)
        blocks_per_level: Number of residual blocks per resolution level (default 2)
        image_size: Output image size (default 64)
        predict_variance: Whether to predict per-pixel variance (default False)
            If True, outputs (mu, log_var). If False, outputs mu only (isotropic).
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 256,
        blocks_per_level: int = 2,
        image_size: int = 64,
        predict_variance: bool = False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size
        self.predict_variance = predict_variance

        # Channel progression (reverse of encoder): 8*base -> 4*base -> 2*base -> base
        channels = [
            base_channels * 8,  # Level 0: 8x8
            base_channels * 4,  # Level 1: 16x16
            base_channels * 2,  # Level 2: 32x32
            base_channels,      # Level 3: 64x64
        ]

        # Calculate initial spatial size (after 3 upsamples to reach image_size)
        # image_size = initial_size * 2^3 => initial_size = image_size / 8
        self.initial_spatial = image_size // 8  # 8 for 64x64 images
        self.initial_channels = channels[0]  # 512

        # Linear layer to map latent to initial feature map
        self.fc = nn.Linear(
            latent_dim,
            self.initial_channels * self.initial_spatial * self.initial_spatial,
        )

        # Initial normalization and activation after linear
        self.initial_norm = nn.GroupNorm(32, self.initial_channels)
        self.initial_act = nn.SiLU()

        # Build decoder levels
        self.levels = nn.ModuleList()

        for i in range(len(channels) - 1):
            level = nn.Sequential(
                # Residual blocks at this resolution
                *[ResidualBottleneck(channels[i]) for _ in range(blocks_per_level)],
                # Upsample spatial dimensions by 2
                Upsample(channels[i], channels[i + 1]),
            )
            self.levels.append(level)

        # Final residual blocks at full resolution
        self.final_blocks = nn.Sequential(
            *[ResidualBottleneck(channels[-1]) for _ in range(blocks_per_level)],
        )

        # Output projection
        # Reference: Task1_VAE_Guide.md Step 5
        # Isotropic: output 3 channels (mu only)
        # Variance-predicting: output 6 channels (mu + log_var)
        output_channels = out_channels * 2 if predict_variance else out_channels
        self.output_conv = nn.Conv2d(
            channels[-1],
            output_channels,
            kernel_size=3,
            padding=1,
        )

        self.out_channels = out_channels

    def forward(
        self, z: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            z: Latent vector of shape (B, latent_dim)

        Returns:
            If predict_variance=False (isotropic):
                mu: Reconstructed image mean of shape (B, C, H, W)
            If predict_variance=True:
                Tuple of (mu, log_var), each of shape (B, C, H, W)
        """
        batch_size = z.size(0)

        # Linear projection
        x = self.fc(z)

        # Reshape to spatial
        x = x.view(
            batch_size,
            self.initial_channels,
            self.initial_spatial,
            self.initial_spatial,
        )

        # Normalize and activate
        x = self.initial_act(self.initial_norm(x))

        # Upsample through levels
        for level in self.levels:
            x = level(x)

        # Final blocks
        x = self.final_blocks(x)

        # Output projection
        x = self.output_conv(x)

        if self.predict_variance:
            # Split into mu and log_var
            mu = x[:, :self.out_channels]
            log_var = x[:, self.out_channels:]

            # Clamp log_var for numerical stability
            # Reference: Task1_VAE_Guide.md Step 5
            # "clamp log_var to [-10, 10] to prevent instability"
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)

            return mu, log_var
        else:
            # Isotropic decoder: return mu only
            # Apply sigmoid to constrain output to [0, 1]
            mu = torch.sigmoid(x)
            return mu


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test decoder with dummy input.
    """
    print("Testing Decoder...")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test isotropic decoder
    print("\n1. Isotropic Decoder (predict_variance=False):")
    decoder_iso = Decoder(
        out_channels=3,
        base_channels=64,
        latent_dim=256,
        blocks_per_level=2,
        image_size=64,
        predict_variance=False,
    ).to(device)

    num_params_iso = sum(p.numel() for p in decoder_iso.parameters())
    print(f"   Parameters: {num_params_iso:,}")

    z = torch.randn(4, 256, device=device)
    mu = decoder_iso(z)

    print(f"   Input shape: {z.shape}")
    print(f"   Output (mu) shape: {mu.shape}")
    print(f"   Output range: [{mu.min():.3f}, {mu.max():.3f}]")

    assert mu.shape == (4, 3, 64, 64), f"Unexpected mu shape: {mu.shape}"
    assert mu.min() >= 0.0, f"mu below 0: {mu.min()}"
    assert mu.max() <= 1.0, f"mu above 1: {mu.max()}"
    print("   ✓ Passed")

    # Test variance-predicting decoder
    print("\n2. Variance-Predicting Decoder (predict_variance=True):")
    decoder_var = Decoder(
        out_channels=3,
        base_channels=64,
        latent_dim=256,
        blocks_per_level=2,
        image_size=64,
        predict_variance=True,
    ).to(device)

    num_params_var = sum(p.numel() for p in decoder_var.parameters())
    print(f"   Parameters: {num_params_var:,}")

    z = torch.randn(4, 256, device=device)
    mu, log_var = decoder_var(z)

    print(f"   Input shape: {z.shape}")
    print(f"   Output (mu) shape: {mu.shape}")
    print(f"   Output (log_var) shape: {log_var.shape}")
    print(f"   log_var range: [{log_var.min():.3f}, {log_var.max():.3f}]")

    assert mu.shape == (4, 3, 64, 64), f"Unexpected mu shape: {mu.shape}"
    assert log_var.shape == (4, 3, 64, 64), f"Unexpected log_var shape: {log_var.shape}"
    assert log_var.min() >= -10.0, f"log_var below -10: {log_var.min()}"
    assert log_var.max() <= 10.0, f"log_var above 10: {log_var.max()}"
    print("   ✓ Passed")

    print("\n" + "-" * 50)
    print("✓ All decoder tests passed!")

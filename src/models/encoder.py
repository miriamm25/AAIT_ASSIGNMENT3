"""
Component: VAE Encoder
Reference: Task1_VAE_Guide.md Step 4

Purpose:
    Encode input images into latent space parameters (mu, log_var).
    Input: 64×64×3 images
    Output: mu and log_var vectors of size latent_dim

Key implementation notes:
    - Downsample 3 times: 64 -> 32 -> 16 -> 8
    - Channels increase: base -> 2*base -> 4*base -> 8*base
    - Use nn.Linear at the end, NOT GlobalAvgPool
    - Reference: Task1_VAE_Guide.md Step 4

Teacher's advice incorporated:
    - "If you use GlobalAveragePooling you are going to lose A LOT of information"
    - "Instead, use nn.Linear to map the flattened features to the latent space"

Architecture (for base_channels=64, latent_dim=256):
    Input: (B, 3, 64, 64)
    -> Initial conv: (B, 64, 64, 64)
    -> Downsample + blocks: (B, 128, 32, 32)
    -> Downsample + blocks: (B, 256, 16, 16)
    -> Downsample + blocks: (B, 512, 8, 8)
    -> Flatten: (B, 512*8*8) = (B, 32768)
    -> Linear: (B, 512)  # mu and log_var each of size 256
"""

import torch
import torch.nn as nn
from typing import Tuple

from .blocks import ConvNormAct, ResidualBottleneck, Downsample


class Encoder(nn.Module):
    """
    VAE Encoder that maps images to latent space parameters.

    Reference: Task1_VAE_Guide.md Step 4
        "The encoder should produce mu and log_var for the latent distribution"

    Args:
        in_channels: Number of input channels (default 3 for RGB)
        base_channels: Base number of channels (default 64)
        latent_dim: Dimension of latent space (default 256)
        blocks_per_level: Number of residual blocks per resolution level (default 2)
        image_size: Input image size (default 64)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 256,
        blocks_per_level: int = 2,
        image_size: int = 64,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size

        # Channel progression: base -> 2*base -> 4*base -> 8*base
        channels = [
            base_channels,      # Level 0: 64x64
            base_channels * 2,  # Level 1: 32x32
            base_channels * 4,  # Level 2: 16x16
            base_channels * 8,  # Level 3: 8x8
        ]

        # Initial convolution to go from RGB to base_channels
        self.initial_conv = ConvNormAct(in_channels, channels[0], kernel_size=3)

        # Build encoder levels
        self.levels = nn.ModuleList()

        for i in range(len(channels) - 1):
            level = nn.Sequential(
                # Downsample spatial dimensions by 2
                Downsample(channels[i], channels[i + 1]),
                # Residual blocks at this resolution
                *[ResidualBottleneck(channels[i + 1]) for _ in range(blocks_per_level)],
            )
            self.levels.append(level)

        # Calculate flattened size after all downsampling
        # 64 -> 32 -> 16 -> 8 (3 downsamples)
        final_spatial = image_size // (2 ** len(self.levels))  # 64 // 8 = 8
        final_channels = channels[-1]  # 512
        self.flattened_size = final_channels * final_spatial * final_spatial

        # Final linear layers to produce mu and log_var
        # Reference: Task1_VAE_Guide.md - "use nn.Linear, NOT GlobalAvgPool"
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (mu, log_var), each of shape (B, latent_dim)
        """
        # Initial convolution
        x = self.initial_conv(x)

        # Downsample through levels
        for level in self.levels:
            x = level(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Produce mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test encoder with dummy input.
    """
    print("Testing Encoder...")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create encoder with default config
    encoder = Encoder(
        in_channels=3,
        base_channels=64,
        latent_dim=256,
        blocks_per_level=2,
        image_size=64,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(4, 3, 64, 64, device=device)
    mu, log_var = encoder(x)

    print(f"\nInput shape: {x.shape}")
    print(f"mu shape: {mu.shape}")
    print(f"log_var shape: {log_var.shape}")

    # Verify shapes
    assert mu.shape == (4, 256), f"Unexpected mu shape: {mu.shape}"
    assert log_var.shape == (4, 256), f"Unexpected log_var shape: {log_var.shape}"

    print("\n" + "-" * 50)
    print("✓ Encoder test passed!")

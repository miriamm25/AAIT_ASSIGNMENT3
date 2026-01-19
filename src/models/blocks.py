"""
Component: Model Building Blocks
Reference: Task1_VAE_Guide.md Step 3

Purpose:
    Implement the basic building blocks for the VAE architecture:
    - ConvNormAct: Convolution + GroupNorm + SiLU activation
    - SqueezeExcitation: Channel attention mechanism
    - ResidualBottleneck: Main residual block with SE and LayerScale
    - Downsample: Spatial downsampling for encoder
    - Upsample: Spatial upsampling for decoder

Key implementation notes:
    - Use GroupNorm instead of BatchNorm (better for small batches)
    - Use SiLU (Swish) activation
    - SE block reduces channels by ratio 4 (or 16 for large models)
    - LayerScale initialized to small value (1e-6) for stability
    - Reference: Task1_VAE_Guide.md Step 3.1-3.4

Teacher's advice incorporated:
    - "Use a modern convolutional block: ConvNormAct with GroupNorm and SiLU"
    - "The SE block provides channel attention"
    - "LayerScale helps with training stability"
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvNormAct(nn.Module):
    """
    Convolution + GroupNorm + SiLU activation block.

    Reference: Task1_VAE_Guide.md Step 3.1
        "ConvNormAct: A convolutional layer followed by GroupNorm and SiLU activation"

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default 3)
        stride: Convolution stride (default 1)
        groups: Number of groups for GroupNorm (default 32, or in_channels if smaller)
        use_act: Whether to apply activation (default True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 32,
        use_act: bool = True,
    ):
        super().__init__()

        # Padding to maintain spatial dimensions (for stride=1)
        padding = kernel_size // 2

        # Convolution
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # No bias needed before normalization
        )

        # GroupNorm - use min(groups, out_channels) to handle small channel counts
        # Reference: Task1_VAE_Guide.md recommends GroupNorm over BatchNorm
        num_groups = min(groups, out_channels)
        # Ensure out_channels is divisible by num_groups
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_channels)

        # SiLU activation (Swish)
        # Reference: Task1_VAE_Guide.md Step 3.1 - "SiLU activation"
        self.act = nn.SiLU() if use_act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W)

        Returns:
            Output tensor of shape (B, C_out, H', W')
            where H', W' depend on stride
        """
        return self.act(self.norm(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Reference: Task1_VAE_Guide.md Step 3.2
        "SE block: Squeeze-and-Excitation for channel attention"
        "Reduces channels by ratio (typically 4 or 16)"

    The SE block:
    1. Global average pools to (B, C, 1, 1)
    2. Reduces channels by ratio
    3. Expands back to original channels
    4. Applies sigmoid to get attention weights
    5. Multiplies input by attention weights

    Args:
        channels: Number of input/output channels
        reduction_ratio: Channel reduction ratio (default 4)
    """

    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()

        # Reduced channel count
        reduced_channels = max(channels // reduction_ratio, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention-weighted tensor of shape (B, C, H, W)
        """
        b, c, _, _ = x.shape

        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)

        # Excitation: FC -> SiLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)

        # Scale input by attention weights
        return x * y


class ResidualBottleneck(nn.Module):
    """
    Residual bottleneck block with SE attention and LayerScale.

    Reference: Task1_VAE_Guide.md Step 3.3
        "ResidualBottleneck: A bottleneck residual block"
        "Structure: ConvNormAct -> ConvNormAct -> ConvNormAct -> SE -> LayerScale"

    The bottleneck structure:
    1. 1x1 conv to reduce channels (bottleneck)
    2. 3x3 conv for spatial processing
    3. 1x1 conv to expand back
    4. SE attention
    5. LayerScale for stability
    6. Residual connection

    Args:
        channels: Number of input/output channels
        bottleneck_ratio: Ratio for bottleneck reduction (default 4)
        se_reduction: SE block reduction ratio (default 4)
        layerscale_init: Initial value for LayerScale (default 1e-6)
    """

    def __init__(
        self,
        channels: int,
        bottleneck_ratio: int = 4,
        se_reduction: int = 4,
        layerscale_init: float = 1e-6,
    ):
        super().__init__()

        # Bottleneck channels
        bottleneck_channels = max(channels // bottleneck_ratio, 1)

        # Main path: 1x1 -> 3x3 -> 1x1
        self.conv1 = ConvNormAct(channels, bottleneck_channels, kernel_size=1)
        self.conv2 = ConvNormAct(bottleneck_channels, bottleneck_channels, kernel_size=3)
        self.conv3 = ConvNormAct(bottleneck_channels, channels, kernel_size=1, use_act=False)

        # SE attention
        # Reference: Task1_VAE_Guide.md - "SE block provides channel attention"
        self.se = SqueezeExcitation(channels, se_reduction)

        # LayerScale for training stability
        # Reference: Task1_VAE_Guide.md - "LayerScale helps with training stability"
        # Initialize to small value to start with mostly residual
        self.layerscale = nn.Parameter(
            layerscale_init * torch.ones(channels, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Main path
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)

        # LayerScale and residual connection
        x = residual + self.layerscale * x

        return x


class Downsample(nn.Module):
    """
    Spatial downsampling block (2x reduction).

    Reference: Task1_VAE_Guide.md Step 3.4
        "Downsample: Reduces spatial dimensions by factor of 2"
        "Uses strided convolution with average pooling residual"

    Structure:
    - Main path: Strided 3x3 conv
    - Residual: Average pool + 1x1 conv (if channel change)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Main path: strided convolution
        self.conv = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
        )

        # Residual path: average pool + 1x1 conv
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W)

        Returns:
            Output tensor of shape (B, C_out, H/2, W/2)
        """
        return self.conv(x) + self.residual(x)


class Upsample(nn.Module):
    """
    Spatial upsampling block (2x increase).

    Reference: Task1_VAE_Guide.md Step 3.4
        "Upsample: Increases spatial dimensions by factor of 2"
        "Uses bilinear interpolation followed by convolution"

    Structure:
    - Main path: Bilinear upsample + 3x3 conv
    - Residual: Bilinear upsample + 1x1 conv (if channel change)

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Main path: bilinear upsample + conv
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvNormAct(in_channels, out_channels, kernel_size=3)

        # Residual path: upsample + 1x1 conv
        self.residual = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, H, W)

        Returns:
            Output tensor of shape (B, C_out, H*2, W*2)
        """
        return self.conv(self.up(x)) + self.residual(x)


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test all blocks with dummy tensors.
    """
    print("Testing model blocks...")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test ConvNormAct
    print("\n1. ConvNormAct:")
    conv_block = ConvNormAct(64, 128).to(device)
    x = torch.randn(2, 64, 32, 32, device=device)
    y = conv_block(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (2, 128, 32, 32), f"Unexpected shape: {y.shape}"
    print("   ✓ Passed")

    # Test SqueezeExcitation
    print("\n2. SqueezeExcitation:")
    se_block = SqueezeExcitation(128).to(device)
    x = torch.randn(2, 128, 32, 32, device=device)
    y = se_block(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == x.shape, f"Unexpected shape: {y.shape}"
    print("   ✓ Passed")

    # Test ResidualBottleneck
    print("\n3. ResidualBottleneck:")
    res_block = ResidualBottleneck(128).to(device)
    x = torch.randn(2, 128, 32, 32, device=device)
    y = res_block(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == x.shape, f"Unexpected shape: {y.shape}"
    print("   ✓ Passed")

    # Test Downsample
    print("\n4. Downsample:")
    down_block = Downsample(64, 128).to(device)
    x = torch.randn(2, 64, 32, 32, device=device)
    y = down_block(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (2, 128, 16, 16), f"Unexpected shape: {y.shape}"
    print("   ✓ Passed")

    # Test Upsample
    print("\n5. Upsample:")
    up_block = Upsample(128, 64).to(device)
    x = torch.randn(2, 128, 16, 16, device=device)
    y = up_block(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (2, 64, 32, 32), f"Unexpected shape: {y.shape}"
    print("   ✓ Passed")

    print("\n" + "-" * 50)
    print("✓ All block tests passed!")

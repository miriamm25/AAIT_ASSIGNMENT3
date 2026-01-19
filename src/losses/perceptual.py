"""
Component: Perceptual Loss (VGG-based)
Reference: Task1_VAE_Guide.md Step 9

Purpose:
    Compute perceptual loss using VGG16 features.
    This measures similarity in feature space rather than pixel space,
    which tends to produce sharper and more perceptually pleasing reconstructions.

Key implementation notes:
    - Use pre-trained VGG16 features
    - Extract features from early layers (before too much abstraction)
    - Freeze VGG weights (no training)
    - MSE in feature space
    - Reference: Task1_VAE_Guide.md Step 9

Teacher's advice incorporated:
    - "Adding a perceptual loss can make the training converge much faster"
    - "It produces sharper reconstructions"
    - "Use early VGG layers for better texture preservation"

Paper reference (optional reading):
    "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
    Johnson et al., 2016
    https://arxiv.org/abs/1603.08155
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.

    Reference: Task1_VAE_Guide.md Step 9
        "Adding a perceptual loss can make the training converge much faster
        and produce sharper reconstructions"

    The loss computes MSE between VGG features of original and reconstructed images.
    Features are extracted from early layers to preserve texture information.

    Args:
        layers: List of VGG layer indices to extract features from.
            Default uses layers before max pooling for better texture preservation.
        normalize_input: Whether to normalize input to ImageNet statistics.
            Set to True if input is in [0, 1] range (default).
        resize_input: Whether to resize input to 224x224 (VGG input size).
            Set to False if input is already 64x64 (will still work, just smaller).
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        normalize_input: bool = True,
        resize_input: bool = False,
    ):
        super().__init__()

        # Default layers: conv1_2, conv2_2, conv3_3 (before max pooling)
        # These preserve texture information well
        if layers is None:
            layers = [3, 8, 15]  # ReLU after conv1_2, conv2_2, conv3_3

        self.layers = layers
        self.normalize_input = normalize_input
        self.resize_input = resize_input

        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Extract feature layers up to the deepest one we need
        max_layer = max(layers) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])

        # Freeze VGG weights
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet normalization parameters
        # Input is expected to be in [0, 1], then normalized to ImageNet stats
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input to ImageNet statistics."""
        return (x - self.mean) / self.std

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from specified VGG layers.

        Args:
            x: Input tensor of shape (B, 3, H, W) in [0, 1] range

        Returns:
            List of feature tensors from each specified layer
        """
        # Normalize if needed
        if self.normalize_input:
            x = self.normalize(x)

        # Resize if needed
        if self.resize_input:
            x = nn.functional.interpolate(
                x,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        # Extract features at each specified layer
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layers:
                features.append(x)

        return features

    def forward(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute perceptual loss between original and reconstructed images.

        Args:
            x: Original images, shape (B, 3, H, W) in [0, 1] range
            recon: Reconstructed images, shape (B, 3, H, W) in [0, 1] range
            reduction: 'mean' (default), 'sum', or 'none'

        Returns:
            Perceptual loss (scalar if reduction='mean' or 'sum', else shape (B,))
        """
        # Extract features
        x_features = self.extract_features(x)
        recon_features = self.extract_features(recon)

        # Compute MSE in feature space for each layer
        loss = 0.0
        for x_feat, recon_feat in zip(x_features, recon_features):
            if reduction == "none":
                # Sum over C, H, W for each sample
                layer_loss = torch.sum((x_feat - recon_feat).pow(2), dim=(1, 2, 3))
            else:
                layer_loss = nn.functional.mse_loss(x_feat, recon_feat, reduction=reduction)
            loss = loss + layer_loss

        return loss


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test perceptual loss.
    """
    print("Testing Perceptual Loss...")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create perceptual loss module
    perceptual_loss = PerceptualLoss().to(device)
    perceptual_loss.eval()

    # Test 1: Identical images (should be ~0)
    print("\n1. Perceptual loss with identical images (should be ~0):")
    x = torch.rand(4, 3, 64, 64, device=device)
    loss = perceptual_loss(x, x)
    print(f"   Loss = {loss.item():.6f}")
    assert loss.item() < 1e-5, f"Expected ~0, got {loss.item()}"
    print("   ✓ Passed")

    # Test 2: Different images (should be positive)
    print("\n2. Perceptual loss with different images:")
    x = torch.rand(4, 3, 64, 64, device=device)
    recon = torch.rand(4, 3, 64, 64, device=device)
    loss = perceptual_loss(x, recon)
    print(f"   Loss = {loss.item():.4f}")
    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    print("   ✓ Passed")

    # Test 3: Gradients flow to recon but not to VGG
    print("\n3. Gradient flow:")
    x = torch.rand(4, 3, 64, 64, device=device)
    recon = torch.rand(4, 3, 64, 64, device=device, requires_grad=True)
    loss = perceptual_loss(x, recon)
    loss.backward()

    # Check that recon has gradients
    assert recon.grad is not None, "recon should have gradients"
    print(f"   recon.grad: {recon.grad.shape}")

    # Check that VGG parameters don't have gradients (frozen)
    for param in perceptual_loss.features.parameters():
        assert param.grad is None, "VGG params should not have gradients"
    print("   VGG params: No gradients (frozen)")
    print("   ✓ Passed")

    # Test 4: Reduction modes
    print("\n4. Reduction modes:")
    x = torch.rand(4, 3, 64, 64, device=device)
    recon = torch.rand(4, 3, 64, 64, device=device)

    loss_mean = perceptual_loss(x, recon, reduction="mean")
    loss_sum = perceptual_loss(x, recon, reduction="sum")
    loss_none = perceptual_loss(x, recon, reduction="none")

    print(f"   mean: {loss_mean.shape}, sum: {loss_sum.shape}, none: {loss_none.shape}")
    assert loss_mean.shape == (), f"Expected scalar, got {loss_mean.shape}"
    assert loss_sum.shape == (), f"Expected scalar, got {loss_sum.shape}"
    assert loss_none.shape == (4,), f"Expected (4,), got {loss_none.shape}"
    print("   ✓ Passed")

    print("\n" + "-" * 50)
    print("✓ All perceptual loss tests passed!")

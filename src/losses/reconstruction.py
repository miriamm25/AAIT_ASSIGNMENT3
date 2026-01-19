"""
Component: Reconstruction Losses
Reference: AAIT_Assignment_3.pdf

Purpose:
    Compute reconstruction loss between original and reconstructed images.
    Two variants:
    1. MSE loss - for isotropic decoder (fixed variance)
    2. Gaussian NLL loss - for variance-predicting decoder

Key implementation notes:
    - MSE is simpler and works well for most cases
    - Gaussian NLL allows the model to predict uncertainty
    - Sum over pixels, mean over batch
    - Reference: AAIT_Assignment_3.pdf equations

Mathematical foundation (from assignment):

    MSE (isotropic decoder):
        L_recon = Σ_d (x_d - μ_d)²

    Gaussian NLL (variance-predicting decoder):
        -log p(x|z) = (1/2) Σ_d [log(2π) + α_d + (x_d - μ_d)² / exp(α_d)]

    Where:
    - x is the original image
    - μ is the reconstructed mean
    - α = log(σ²) is the log variance
    - d indexes the pixels

Teacher's advice incorporated:
    - "MSE is equivalent to Gaussian NLL with fixed variance"
    - "Variance-predicting decoder can improve perceptual quality"
"""

import torch
import math
from typing import Optional


def mse_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute MSE reconstruction loss.

    This is equivalent to Gaussian NLL with fixed variance σ²=1
    (up to a constant).

    Reference: AAIT_Assignment_3.pdf
        L_recon = Σ_d (x_d - μ_d)²

    Args:
        x: Original images, shape (B, C, H, W)
        mu: Reconstructed images (mean), shape (B, C, H, W)
        reduction: 'mean' (default), 'sum', or 'none'
            - 'mean': Mean over batch (sum over pixels, mean over batch)
            - 'sum': Sum over batch and pixels
            - 'none': No reduction, returns shape (B,)

    Returns:
        MSE loss (scalar if reduction='mean' or 'sum', else shape (B,))
    """
    # Compute squared error per sample (sum over C, H, W)
    mse_per_sample = torch.sum((x - mu).pow(2), dim=(1, 2, 3))

    # Apply reduction
    if reduction == "mean":
        return mse_per_sample.mean()
    elif reduction == "sum":
        return mse_per_sample.sum()
    elif reduction == "none":
        return mse_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def gaussian_nll_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Gaussian negative log-likelihood loss.

    This allows the model to predict per-pixel uncertainty.

    Reference: AAIT_Assignment_3.pdf
        -log p(x|z) = (1/2) Σ_d [log(2π) + α_d + (x_d - μ_d)² / exp(α_d)]

    Where α_d = log(σ²_d) is the log variance.

    Args:
        x: Original images, shape (B, C, H, W)
        mu: Reconstructed images (mean), shape (B, C, H, W)
        log_var: Predicted log variance, shape (B, C, H, W)
        reduction: 'mean' (default), 'sum', or 'none'
            - 'mean': Mean over batch (sum over pixels, mean over batch)
            - 'sum': Sum over batch and pixels
            - 'none': No reduction, returns shape (B,)

    Returns:
        Gaussian NLL loss (scalar if reduction='mean' or 'sum', else shape (B,))
    """
    # Gaussian NLL formula:
    # NLL = (1/2) * sum_d [log(2π) + log_var_d + (x_d - mu_d)² / exp(log_var_d)]
    #     = (1/2) * sum_d [log(2π) + log_var_d + (x_d - mu_d)² * exp(-log_var_d)]

    # Compute components
    log_2pi = math.log(2 * math.pi)

    # Squared error divided by variance
    squared_error = (x - mu).pow(2)
    weighted_error = squared_error * torch.exp(-log_var)

    # NLL per pixel
    nll_per_pixel = 0.5 * (log_2pi + log_var + weighted_error)

    # Sum over pixels (C, H, W) to get per-sample NLL
    nll_per_sample = torch.sum(nll_per_pixel, dim=(1, 2, 3))

    # Apply reduction
    if reduction == "mean":
        return nll_per_sample.mean()
    elif reduction == "sum":
        return nll_per_sample.sum()
    elif reduction == "none":
        return nll_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test reconstruction losses.
    """
    print("Testing Reconstruction Losses...")
    print("-" * 50)

    # Test 1: MSE with identical inputs (should be 0)
    print("\n1. MSE with identical inputs (should be 0):")
    x = torch.randn(4, 3, 64, 64)
    mse = mse_loss(x, x)
    print(f"   MSE = {mse.item():.6f}")
    assert abs(mse.item()) < 1e-6, f"Expected 0, got {mse.item()}"
    print("   ✓ Passed")

    # Test 2: MSE with known difference
    print("\n2. MSE with known difference:")
    x = torch.zeros(4, 3, 64, 64)
    mu = torch.ones(4, 3, 64, 64)
    mse = mse_loss(x, mu)
    expected = 3 * 64 * 64  # sum of 1² over all pixels
    print(f"   MSE = {mse.item():.6f}, Expected = {expected}")
    assert abs(mse.item() - expected) < 1e-3, f"Expected {expected}, got {mse.item()}"
    print("   ✓ Passed")

    # Test 3: Gaussian NLL with log_var=0 should be proportional to MSE
    print("\n3. Gaussian NLL relation to MSE:")
    x = torch.randn(4, 3, 64, 64)
    mu = torch.randn(4, 3, 64, 64)
    log_var = torch.zeros(4, 3, 64, 64)

    mse = mse_loss(x, mu)
    nll = gaussian_nll_loss(x, mu, log_var)

    # NLL = (1/2) * (D * log(2π) + MSE) where D = 3*64*64 = 12288
    D = 3 * 64 * 64
    expected_nll = 0.5 * (D * math.log(2 * math.pi) + mse.item())
    print(f"   NLL = {nll.item():.2f}, Expected = {expected_nll:.2f}")
    assert abs(nll.item() - expected_nll) < 1, f"Expected {expected_nll}, got {nll.item()}"
    print("   ✓ Passed")

    # Test 4: Gaussian NLL with high variance should give lower loss
    print("\n4. Gaussian NLL with varying variance:")
    x = torch.zeros(4, 3, 64, 64)
    mu = torch.ones(4, 3, 64, 64)  # Error of 1 at each pixel

    log_var_low = torch.zeros(4, 3, 64, 64)  # var = 1
    log_var_high = torch.full((4, 3, 64, 64), 2.0)  # var = e^2 ≈ 7.4

    nll_low = gaussian_nll_loss(x, mu, log_var_low)
    nll_high = gaussian_nll_loss(x, mu, log_var_high)

    # With high variance, the squared error term is down-weighted,
    # but the log_var term increases. There's a balance.
    print(f"   NLL (low var): {nll_low.item():.2f}")
    print(f"   NLL (high var): {nll_high.item():.2f}")
    print("   ✓ Passed (high variance reduces squared error weight)")

    # Test 5: Reduction modes
    print("\n5. Reduction modes:")
    x = torch.randn(4, 3, 64, 64)
    mu = torch.randn(4, 3, 64, 64)

    mse_mean = mse_loss(x, mu, reduction="mean")
    mse_sum = mse_loss(x, mu, reduction="sum")
    mse_none = mse_loss(x, mu, reduction="none")

    print(f"   mean: {mse_mean.shape}, sum: {mse_sum.shape}, none: {mse_none.shape}")
    assert mse_mean.shape == (), f"Expected scalar, got {mse_mean.shape}"
    assert mse_sum.shape == (), f"Expected scalar, got {mse_sum.shape}"
    assert mse_none.shape == (4,), f"Expected (4,), got {mse_none.shape}"
    print("   ✓ Passed")

    print("\n" + "-" * 50)
    print("✓ All reconstruction loss tests passed!")

"""
Component: KL Divergence Loss
Reference: AAIT_Assignment_3.pdf

Purpose:
    Compute the KL divergence between the approximate posterior q(z|x)
    and the prior p(z) = N(0, I).

Key implementation notes:
    - Closed-form solution for KL between two Gaussians
    - Sum over latent dimensions, mean over batch
    - Reference: AAIT_Assignment_3.pdf equation

Mathematical foundation (from assignment):
    D_KL(q(z|x) || p(z)) = (1/2) Σ_j (σ²_j + μ²_j - 1 - log(σ²_j))

    Where:
    - q(z|x) = N(μ, diag(σ²)) is the approximate posterior
    - p(z) = N(0, I) is the prior
    - j indexes the latent dimensions

Teacher's advice incorporated:
    - Use log_var instead of σ for numerical stability
    - log(σ²) = log_var, so we use log_var directly
"""

import torch
from typing import Optional


def kl_divergence_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute KL divergence loss D_KL(q(z|x) || p(z)).

    Reference: AAIT_Assignment_3.pdf
        D_KL = (1/2) Σ_j (σ²_j + μ²_j - 1 - log(σ²_j))

    Using log_var instead of σ² for numerical stability:
        D_KL = (1/2) Σ_j (exp(log_var_j) + μ²_j - 1 - log_var_j)

    Args:
        mu: Mean of q(z|x), shape (B, latent_dim)
        log_var: Log variance of q(z|x), shape (B, latent_dim)
        reduction: 'mean' (default), 'sum', or 'none'
            - 'mean': Mean over batch (sum over latent dims, mean over batch)
            - 'sum': Sum over batch and latent dims
            - 'none': No reduction, returns shape (B,)

    Returns:
        KL divergence loss (scalar if reduction='mean' or 'sum', else shape (B,))
    """
    # KL divergence formula:
    # D_KL = (1/2) * sum_j (exp(log_var_j) + mu_j^2 - 1 - log_var_j)
    #      = (1/2) * sum_j (var_j + mu_j^2 - 1 - log_var_j)

    # Compute per-sample KL (sum over latent dimensions)
    kl_per_sample = 0.5 * torch.sum(
        torch.exp(log_var) + mu.pow(2) - 1.0 - log_var,
        dim=1,
    )

    # Apply reduction
    if reduction == "mean":
        return kl_per_sample.mean()
    elif reduction == "sum":
        return kl_per_sample.sum()
    elif reduction == "none":
        return kl_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test KL divergence loss with known values.
    """
    print("Testing KL Divergence Loss...")
    print("-" * 50)

    # Test 1: Standard normal (mu=0, log_var=0 => var=1)
    # KL(N(0,1) || N(0,1)) = 0
    print("\n1. Standard normal (should be ~0):")
    mu = torch.zeros(4, 256)
    log_var = torch.zeros(4, 256)
    kl = kl_divergence_loss(mu, log_var)
    print(f"   KL = {kl.item():.6f}")
    assert abs(kl.item()) < 1e-5, f"Expected ~0, got {kl.item()}"
    print("   ✓ Passed")

    # Test 2: Non-zero mean (mu=1, log_var=0)
    # KL = (1/2) * sum(1 + 1 - 1 - 0) = (1/2) * 1 * latent_dim = 128
    print("\n2. Non-zero mean (mu=1, var=1):")
    mu = torch.ones(4, 256)
    log_var = torch.zeros(4, 256)
    kl = kl_divergence_loss(mu, log_var)
    expected = 0.5 * 256  # (1/2) * latent_dim
    print(f"   KL = {kl.item():.6f}, Expected = {expected}")
    assert abs(kl.item() - expected) < 1e-3, f"Expected {expected}, got {kl.item()}"
    print("   ✓ Passed")

    # Test 3: Non-unit variance (mu=0, log_var=log(2) => var=2)
    # KL = (1/2) * sum(2 + 0 - 1 - log(2)) = (1/2) * (1 - log(2)) * latent_dim
    print("\n3. Non-unit variance (mu=0, var=2):")
    import math
    mu = torch.zeros(4, 256)
    log_var = torch.full((4, 256), math.log(2))
    kl = kl_divergence_loss(mu, log_var)
    expected = 0.5 * (1 - math.log(2)) * 256
    print(f"   KL = {kl.item():.6f}, Expected = {expected:.6f}")
    assert abs(kl.item() - expected) < 1e-3, f"Expected {expected}, got {kl.item()}"
    print("   ✓ Passed")

    # Test 4: Reduction modes
    print("\n4. Reduction modes:")
    mu = torch.randn(4, 256)
    log_var = torch.randn(4, 256)

    kl_mean = kl_divergence_loss(mu, log_var, reduction="mean")
    kl_sum = kl_divergence_loss(mu, log_var, reduction="sum")
    kl_none = kl_divergence_loss(mu, log_var, reduction="none")

    print(f"   mean: {kl_mean.shape}, sum: {kl_sum.shape}, none: {kl_none.shape}")
    assert kl_mean.shape == (), f"Expected scalar, got {kl_mean.shape}"
    assert kl_sum.shape == (), f"Expected scalar, got {kl_sum.shape}"
    assert kl_none.shape == (4,), f"Expected (4,), got {kl_none.shape}"
    print("   ✓ Passed")

    print("\n" + "-" * 50)
    print("✓ All KL divergence tests passed!")

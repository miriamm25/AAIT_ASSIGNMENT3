"""
Component: Evaluation Metrics
Reference: AAIT_Assignment_3.pdf

Purpose:
    Compute evaluation metrics for VAE:
    - Bits Per Dimension (BPD)
    - ELBO

Key implementation notes:
    - BPD measures compression quality in bits per pixel
    - Lower BPD is better (more efficient encoding)
    - Reference: AAIT_Assignment_3.pdf equations

Mathematical foundation (from assignment):
    BPD(x) = -log p_θ(x) / (D · log(2))

    Where:
    - log p_θ(x) ≈ ELBO = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
    - D = total number of dimensions (C × H × W)
    - log(2) converts from nats to bits

    Since ELBO is a lower bound on log p(x), we use:
    BPD ≈ -ELBO / (D · log(2))
"""

import torch
import math
from typing import Optional


def compute_bpd(
    elbo: torch.Tensor,
    num_dimensions: int,
) -> torch.Tensor:
    """
    Compute Bits Per Dimension (BPD) from ELBO.

    Reference: AAIT_Assignment_3.pdf
        BPD(x) = -log p_θ(x) / (D · log(2))

    Since ELBO ≈ log p(x), we have:
        BPD ≈ -ELBO / (D · log(2))

    Note: ELBO is typically negative (or the negative ELBO is positive).
    If you're passing the loss (negative ELBO), this should be positive.

    Args:
        elbo: ELBO value(s). Can be:
            - Positive (if you're passing -ELBO, i.e., the loss)
            - Negative (if you're passing the actual ELBO)
        num_dimensions: Total number of dimensions (C × H × W)
            For 64×64 RGB images: 3 × 64 × 64 = 12288

    Returns:
        BPD value(s)
    """
    # BPD = -ELBO / (D * log(2))
    # If elbo is actually the loss (-ELBO), then:
    # BPD = loss / (D * log(2))

    bpd = elbo / (num_dimensions * math.log(2))

    return bpd


def compute_elbo(
    recon_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute the (negative) ELBO loss.

    Reference: AAIT_Assignment_3.pdf
        ELBO = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
        Loss = -ELBO = -E_q[log p(x|z)] + D_KL

    The reconstruction loss is typically:
        -E_q[log p(x|z)] (negative log-likelihood)

    So the total loss is:
        Loss = recon_loss + kl_weight * kl_loss

    Args:
        recon_loss: Reconstruction loss (negative log-likelihood)
        kl_loss: KL divergence loss
        kl_weight: Weight for KL loss (for annealing)

    Returns:
        Total loss (negative ELBO with KL weight)
    """
    return recon_loss + kl_weight * kl_loss


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test metric calculations.
    """
    print("Testing Metrics...")
    print("-" * 50)

    # Test BPD calculation
    print("\n1. BPD calculation:")

    # For a 64×64 RGB image
    num_dims = 3 * 64 * 64  # 12288

    # If ELBO = -1000 nats, BPD should be positive
    # (we pass negative ELBO, i.e., the loss)
    loss = torch.tensor(1000.0)  # This is -ELBO
    bpd = compute_bpd(loss, num_dims)
    expected_bpd = 1000.0 / (12288 * math.log(2))
    print(f"   Loss = 1000 nats")
    print(f"   BPD = {bpd.item():.4f}, Expected = {expected_bpd:.4f}")
    assert abs(bpd.item() - expected_bpd) < 1e-4
    print("   ✓ Passed")

    # Test with batch
    print("\n2. Batch BPD calculation:")
    losses = torch.tensor([1000.0, 2000.0, 1500.0])
    bpds = compute_bpd(losses, num_dims)
    print(f"   Losses: {losses.tolist()}")
    print(f"   BPDs: {[f'{b:.4f}' for b in bpds.tolist()]}")
    assert bpds.shape == (3,)
    print("   ✓ Passed")

    # Test ELBO computation
    print("\n3. ELBO computation:")
    recon_loss = torch.tensor(500.0)
    kl_loss = torch.tensor(100.0)

    # Full weight
    total = compute_elbo(recon_loss, kl_loss, kl_weight=1.0)
    print(f"   recon={recon_loss.item()}, kl={kl_loss.item()}, weight=1.0")
    print(f"   Total = {total.item()}, Expected = 600.0")
    assert abs(total.item() - 600.0) < 1e-6
    print("   ✓ Passed")

    # Half weight (annealing)
    total_annealed = compute_elbo(recon_loss, kl_loss, kl_weight=0.5)
    print(f"   recon={recon_loss.item()}, kl={kl_loss.item()}, weight=0.5")
    print(f"   Total = {total_annealed.item()}, Expected = 550.0")
    assert abs(total_annealed.item() - 550.0) < 1e-6
    print("   ✓ Passed")

    print("\n" + "-" * 50)
    print("✓ All metric tests passed!")

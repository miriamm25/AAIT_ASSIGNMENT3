"""
Component: KL Annealing
Reference: Task1_VAE_Guide.md Step 8

Purpose:
    Implement KL weight annealing to prevent posterior collapse.
    Starts with low KL weight and gradually increases to full weight.

Key implementation notes:
    - Linear warmup over specified number of epochs
    - Prevents the model from ignoring the latent space early in training
    - Reference: Task1_VAE_Guide.md Step 8

Teacher's advice incorporated:
    - "KL annealing can help prevent posterior collapse"
    - "Linear warmup: kl_weight = min(1.0, epoch / warmup_epochs)"

Mathematical foundation:
    During training, the loss is:
        L = L_recon + β * L_KL

    Where β starts at 0 and linearly increases to 1 over warmup_epochs.
    This allows the model to first learn good reconstructions,
    then gradually learn a meaningful latent space.
"""

import math


class KLAnnealer:
    """
    KL weight annealer with linear warmup.

    Reference: Task1_VAE_Guide.md Step 8
        "Linear warmup: kl_weight = min(1.0, epoch / warmup_epochs)"

    Args:
        warmup_epochs: Number of epochs for linear warmup
        start_weight: Starting weight (default 0.0)
        end_weight: Final weight (default 1.0)
        schedule: Annealing schedule type ('linear', 'cosine')
    """

    def __init__(
        self,
        warmup_epochs: int,
        start_weight: float = 0.0,
        end_weight: float = 1.0,
        schedule: str = "linear",
    ):
        self.warmup_epochs = warmup_epochs
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.schedule = schedule

        if schedule not in ["linear", "cosine"]:
            raise ValueError(f"Unknown schedule: {schedule}")

    def get_weight(self, epoch: int) -> float:
        """
        Get KL weight for the given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            KL weight (between start_weight and end_weight)
        """
        if epoch >= self.warmup_epochs:
            return self.end_weight

        if self.schedule == "linear":
            # Linear warmup
            # Reference: "kl_weight = min(1.0, epoch / warmup_epochs)"
            progress = epoch / self.warmup_epochs
            weight = self.start_weight + (self.end_weight - self.start_weight) * progress

        elif self.schedule == "cosine":
            # Cosine warmup (smoother transition)
            progress = epoch / self.warmup_epochs
            # Cosine schedule: starts slow, accelerates, then slows down
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            weight = self.start_weight + (self.end_weight - self.start_weight) * cosine_factor

        return weight

    def __repr__(self) -> str:
        return (
            f"KLAnnealer(warmup_epochs={self.warmup_epochs}, "
            f"start_weight={self.start_weight}, end_weight={self.end_weight}, "
            f"schedule='{self.schedule}')"
        )


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test KL annealing schedules.
    """
    print("Testing KL Annealing...")
    print("-" * 50)

    # Test 1: Linear annealing
    print("\n1. Linear annealing (10 epoch warmup):")
    annealer = KLAnnealer(warmup_epochs=10, schedule="linear")
    for epoch in [0, 2, 5, 8, 10, 15]:
        weight = annealer.get_weight(epoch)
        print(f"   Epoch {epoch:2d}: weight = {weight:.3f}")

    # Verify specific values
    assert abs(annealer.get_weight(0) - 0.0) < 1e-6
    assert abs(annealer.get_weight(5) - 0.5) < 1e-6
    assert abs(annealer.get_weight(10) - 1.0) < 1e-6
    assert abs(annealer.get_weight(15) - 1.0) < 1e-6
    print("   ✓ Passed")

    # Test 2: Cosine annealing
    print("\n2. Cosine annealing (10 epoch warmup):")
    annealer = KLAnnealer(warmup_epochs=10, schedule="cosine")
    for epoch in [0, 2, 5, 8, 10, 15]:
        weight = annealer.get_weight(epoch)
        print(f"   Epoch {epoch:2d}: weight = {weight:.3f}")

    # Verify bounds
    assert abs(annealer.get_weight(0) - 0.0) < 1e-6
    assert abs(annealer.get_weight(10) - 1.0) < 1e-6
    print("   ✓ Passed")

    # Test 3: Custom start/end weights
    print("\n3. Custom weights (0.1 to 0.5):")
    annealer = KLAnnealer(
        warmup_epochs=10,
        start_weight=0.1,
        end_weight=0.5,
        schedule="linear",
    )
    for epoch in [0, 5, 10]:
        weight = annealer.get_weight(epoch)
        print(f"   Epoch {epoch:2d}: weight = {weight:.3f}")

    assert abs(annealer.get_weight(0) - 0.1) < 1e-6
    assert abs(annealer.get_weight(5) - 0.3) < 1e-6
    assert abs(annealer.get_weight(10) - 0.5) < 1e-6
    print("   ✓ Passed")

    print("\n" + "-" * 50)
    print("✓ All KL annealing tests passed!")

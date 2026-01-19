"""
Component: Full VAE Model
Reference: AAIT_Assignment_3.pdf, Task1_VAE_Guide.md

Purpose:
    Combine encoder and decoder with the reparameterization trick.
    This is the main model class that handles the full VAE pipeline.

Key implementation notes:
    - Encoder produces mu and log_var for q(z|x)
    - Reparameterization trick: z = mu + sigma * epsilon
    - Decoder reconstructs x from z
    - Reference: AAIT_Assignment_3.pdf equations

Teacher's advice incorporated:
    - Reparameterization allows backpropagation through sampling
    - log_var is used instead of sigma for numerical stability

Mathematical foundation (from assignment):
    Reparameterization: z = μ(x) + σ(x) · ε, where ε ~ N(0, I)
    This allows gradients to flow through the sampling operation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Union, Optional

from .encoder import Encoder
from .decoder import Decoder


class VAE(nn.Module):
    """
    Variational Autoencoder combining encoder and decoder.

    Reference: AAIT_Assignment_3.pdf
        "The VAE consists of an encoder q_φ(z|x) and decoder p_θ(x|z)"

    Args:
        in_channels: Number of input/output channels (default 3 for RGB)
        base_channels: Base number of channels (default 64)
        latent_dim: Dimension of latent space (default 256)
        blocks_per_level: Number of residual blocks per level (default 2)
        image_size: Input/output image size (default 64)
        predict_variance: Whether decoder predicts variance (default False)
    """

    def __init__(
        self,
        in_channels: int = 3,
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

        # Encoder: x -> (mu, log_var)
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            blocks_per_level=blocks_per_level,
            image_size=image_size,
        )

        # Decoder: z -> x (or z -> (mu_x, log_var_x))
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            blocks_per_level=blocks_per_level,
            image_size=image_size,
            predict_variance=predict_variance,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample z from q(z|x).

        Reference: AAIT_Assignment_3.pdf
            "z = μ(x) + σ(x) · ε, where ε ~ N(0, I)"

        This allows backpropagation through the sampling operation by
        expressing the random sample as a deterministic transformation
        of the parameters and a random noise variable.

        Args:
            mu: Mean of q(z|x), shape (B, latent_dim)
            log_var: Log variance of q(z|x), shape (B, latent_dim)

        Returns:
            z: Sampled latent vector, shape (B, latent_dim)
        """
        # Standard deviation from log variance
        # σ = exp(0.5 * log(σ²)) = exp(0.5 * log_var)
        std = torch.exp(0.5 * log_var)

        # Sample epsilon from N(0, I)
        eps = torch.randn_like(std)

        # Reparameterization: z = μ + σ * ε
        z = mu + std * eps

        return z

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (mu, log_var), each of shape (B, latent_dim)
        """
        return self.encoder(x)

    def decode(
        self, z: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode latent vector to image.

        Args:
            z: Latent vector of shape (B, latent_dim)

        Returns:
            If predict_variance=False: mu_x of shape (B, C, H, W)
            If predict_variance=True: (mu_x, log_var_x) of shape (B, C, H, W) each
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full VAE.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Dictionary containing:
                - 'mu_z': Encoder mean, shape (B, latent_dim)
                - 'log_var_z': Encoder log variance, shape (B, latent_dim)
                - 'z': Sampled latent, shape (B, latent_dim)
                - 'mu_x': Decoder mean (reconstruction), shape (B, C, H, W)
                - 'log_var_x': Decoder log variance (if predict_variance=True)
        """
        # Encode
        mu_z, log_var_z = self.encode(x)

        # Reparameterize
        z = self.reparameterize(mu_z, log_var_z)

        # Decode
        decoder_output = self.decode(z)

        # Build output dictionary
        output = {
            'mu_z': mu_z,
            'log_var_z': log_var_z,
            'z': z,
        }

        if self.predict_variance:
            mu_x, log_var_x = decoder_output
            output['mu_x'] = mu_x
            output['log_var_x'] = log_var_x
        else:
            output['mu_x'] = decoder_output

        return output

    def sample(
        self,
        num_samples: int,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate samples from the prior p(z) = N(0, I).

        Reference: Task1_VAE_Guide.md - Temperature sampling
            "Sample z ~ N(0, τ²I) where τ is the temperature"

        Args:
            num_samples: Number of samples to generate
            temperature: Sampling temperature (default 1.0)
                - Lower temperature (<1): More conservative samples
                - Higher temperature (>1): More diverse samples
            device: Device to generate samples on

        Returns:
            Generated images of shape (num_samples, C, H, W)
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from prior with temperature scaling
        z = torch.randn(num_samples, self.latent_dim, device=device) * temperature

        # Decode
        with torch.no_grad():
            if self.predict_variance:
                mu_x, _ = self.decode(z)
            else:
                mu_x = self.decode(z)

        return mu_x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Reconstructed images of shape (B, C, H, W)
        """
        with torch.no_grad():
            output = self.forward(x)
        return output['mu_x']

    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.

        Reference: AAIT_Assignment_3.pdf
            "z_i = z_1 · α + (1 - α) · z_2"

        Args:
            x1: First image, shape (1, C, H, W)
            x2: Second image, shape (1, C, H, W)
            num_steps: Number of interpolation steps

        Returns:
            Interpolated images, shape (num_steps, C, H, W)
        """
        with torch.no_grad():
            # Encode both images (use mean, not sample)
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            # Generate interpolation alphas
            alphas = torch.linspace(0, 1, num_steps, device=mu1.device)

            # Interpolate in latent space
            interpolations = []
            for alpha in alphas:
                z = alpha * mu1 + (1 - alpha) * mu2
                if self.predict_variance:
                    recon, _ = self.decode(z)
                else:
                    recon = self.decode(z)
                interpolations.append(recon)

            # Stack into single tensor
            interpolations = torch.cat(interpolations, dim=0)

        return interpolations


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test full VAE with forward pass, sampling, and interpolation.
    """
    print("Testing VAE...")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test isotropic VAE
    print("\n1. Isotropic VAE:")
    vae_iso = VAE(
        in_channels=3,
        base_channels=64,
        latent_dim=256,
        blocks_per_level=2,
        image_size=64,
        predict_variance=False,
    ).to(device)

    num_params = sum(p.numel() for p in vae_iso.parameters())
    print(f"   Total parameters: {num_params:,}")

    # Forward pass
    x = torch.randn(4, 3, 64, 64, device=device)
    output = vae_iso(x)

    print(f"   Input shape: {x.shape}")
    print(f"   mu_z shape: {output['mu_z'].shape}")
    print(f"   log_var_z shape: {output['log_var_z'].shape}")
    print(f"   z shape: {output['z'].shape}")
    print(f"   mu_x shape: {output['mu_x'].shape}")

    assert output['mu_z'].shape == (4, 256)
    assert output['log_var_z'].shape == (4, 256)
    assert output['z'].shape == (4, 256)
    assert output['mu_x'].shape == (4, 3, 64, 64)
    print("   ✓ Forward pass passed")

    # Sampling
    samples = vae_iso.sample(8, temperature=1.0, device=device)
    print(f"   Samples shape: {samples.shape}")
    assert samples.shape == (8, 3, 64, 64)
    print("   ✓ Sampling passed")

    # Reconstruction
    recon = vae_iso.reconstruct(x)
    print(f"   Reconstruction shape: {recon.shape}")
    assert recon.shape == (4, 3, 64, 64)
    print("   ✓ Reconstruction passed")

    # Interpolation
    x1 = x[:1]
    x2 = x[1:2]
    interp = vae_iso.interpolate(x1, x2, num_steps=10)
    print(f"   Interpolation shape: {interp.shape}")
    assert interp.shape == (10, 3, 64, 64)
    print("   ✓ Interpolation passed")

    # Test variance-predicting VAE
    print("\n2. Variance-Predicting VAE:")
    vae_var = VAE(
        in_channels=3,
        base_channels=64,
        latent_dim=256,
        blocks_per_level=2,
        image_size=64,
        predict_variance=True,
    ).to(device)

    output = vae_var(x)
    print(f"   mu_x shape: {output['mu_x'].shape}")
    print(f"   log_var_x shape: {output['log_var_x'].shape}")

    assert output['mu_x'].shape == (4, 3, 64, 64)
    assert output['log_var_x'].shape == (4, 3, 64, 64)
    print("   ✓ Variance-predicting VAE passed")

    print("\n" + "-" * 50)
    print("✓ All VAE tests passed!")

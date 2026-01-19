"""
Data loading module for VAE CelebA.
Reference: AAIT_Assignment_3.pdf - CelebA dataset at 64×64 resolution
"""

from .celeba import get_celeba_dataloaders, CelebADataset

__all__ = ["get_celeba_dataloaders", "CelebADataset"]

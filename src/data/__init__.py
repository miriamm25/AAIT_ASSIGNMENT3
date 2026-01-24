"""
Data loading module for VAE CelebA.
Reference: AAIT_Assignment_3.pdf - CelebA dataset at 64×64 resolution
"""

from .celeba import get_celeba_dataloaders, CelebADataset
from .celeba_with_attributes import (
    get_celeba_attribute_dataloaders,
    CelebAWithAttributes,
    CELEBA_ATTRIBUTES,
    get_images_by_identity,
    get_unique_identities,
)

__all__ = [
    # Basic CelebA (Task 1)
    "get_celeba_dataloaders",
    "CelebADataset",
    # CelebA with attributes (Task 2)
    "get_celeba_attribute_dataloaders",
    "CelebAWithAttributes",
    "CELEBA_ATTRIBUTES",
    "get_images_by_identity",
    "get_unique_identities",
]

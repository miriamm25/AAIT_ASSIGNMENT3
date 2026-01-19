"""
Component: CelebA Data Loading
Reference: AAIT_Assignment_3.pdf, Task1_VAE_Guide.md Step 1

Purpose:
    Load and preprocess the CelebA dataset for VAE training.
    Images are resized to 64×64 as specified in the assignment.

Key implementation notes:
    - Use native train/test splits from CelebA
    - Resize images to 64×64
    - Normalize to [0, 1] range (ToTensor handles this)
    - Reference: "64×64 pixel images" (AAIT_Assignment_3.pdf)

Teacher's advice incorporated:
    - "batch_size 16-32 can help prevent mode collapse" (Task1_VAE_Guide.md Step 1)
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from typing import Tuple, Optional
import os


class CelebADataset(Dataset):
    """
    Wrapper around torchvision CelebA dataset.

    Reference: AAIT_Assignment_3.pdf specifies using CelebA at 64×64 resolution.

    Args:
        root: Root directory for the dataset
        split: One of 'train', 'valid', 'test', or 'all'
        image_size: Target image size (default 64)
        download: Whether to download if not present
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 64,
        download: bool = True,
    ):
        self.image_size = image_size

        # Define transforms
        # Reference: Images should be 64×64 pixels (AAIT_Assignment_3.pdf)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])

        # Load CelebA dataset
        # Using native splits as recommended
        self.dataset = CelebA(
            root=root,
            split=split,
            transform=self.transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns only the image (no labels needed for VAE).

        Args:
            idx: Index of the sample

        Returns:
            Image tensor of shape (3, image_size, image_size) in [0, 1] range
        """
        # CelebA returns (image, target), we only need the image
        image, _ = self.dataset[idx]
        return image


def get_celeba_dataloaders(
    root: str = "./data",
    image_size: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for CelebA.

    Reference: Task1_VAE_Guide.md Step 1
        - "batch_size 16-32 can help prevent mode collapse"
        - Use native train/valid/test splits

    Args:
        root: Root directory for the dataset
        image_size: Target image size (default 64 as per assignment)
        batch_size: Batch size (default 32, within recommended 16-32 range)
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        download: Whether to download if not present

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets for each split
    train_dataset = CelebADataset(
        root=root,
        split="train",
        image_size=image_size,
        download=download,
    )

    val_dataset = CelebADataset(
        root=root,
        split="valid",
        image_size=image_size,
        download=download,
    )

    test_dataset = CelebADataset(
        root=root,
        split="test",
        image_size=image_size,
        download=download,
    )

    # Create dataloaders
    # Reference: batch_size 16-32 recommended (Task1_VAE_Guide.md)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch for consistent batch sizes
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test data loading with dummy forward pass.
    Run this to ensure data loading works correctly.
    """
    print("Testing CelebA data loading...")
    print("-" * 50)

    # Test with a small subset
    try:
        train_loader, val_loader, test_loader = get_celeba_dataloaders(
            root="./data",
            batch_size=4,
            num_workers=0,  # Use 0 workers for testing
            download=True,
        )

        # Get a batch
        batch = next(iter(train_loader))

        print(f"✓ Train dataset size: {len(train_loader.dataset)}")
        print(f"✓ Val dataset size: {len(val_loader.dataset)}")
        print(f"✓ Test dataset size: {len(test_loader.dataset)}")
        print(f"✓ Batch shape: {batch.shape}")
        print(f"✓ Batch dtype: {batch.dtype}")
        print(f"✓ Value range: [{batch.min():.3f}, {batch.max():.3f}]")

        # Verify shape is correct (B, 3, 64, 64)
        assert batch.shape == (4, 3, 64, 64), f"Unexpected shape: {batch.shape}"
        assert batch.min() >= 0.0, f"Values below 0: {batch.min()}"
        assert batch.max() <= 1.0, f"Values above 1: {batch.max()}"

        print("-" * 50)
        print("✓ All data loading tests passed!")

    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        print("  Make sure CelebA dataset is available or can be downloaded.")

"""
Component: CelebA Data Loading with Attributes
Reference: AAIT_Assignment_3.pdf Task 2 - Label Guidance

Purpose:
    Load CelebA dataset with both images and attribute labels.
    This is needed for training the attribute classifier for Task 2.2.

Key implementation notes:
    - CelebA has 40 binary attributes
    - Attributes are used for training the classifier
    - Also provides access to identity labels for Task 2.3

CelebA Attributes (index: name):
    0: 5_o_Clock_Shadow     10: Blurry              20: Male                30: Rosy_Cheeks
    1: Arched_Eyebrows      11: Brown_Hair          21: Mouth_Slightly_Open 31: Smiling
    2: Attractive           12: Bushy_Eyebrows      22: Mustache            32: Straight_Hair
    3: Bags_Under_Eyes      13: Chubby              23: Narrow_Eyes         33: Wavy_Hair
    4: Bald                 14: Double_Chin         24: No_Beard            34: Wearing_Earrings
    5: Bangs                15: Eyeglasses          25: Oval_Face           35: Wearing_Hat
    6: Big_Lips             16: Goatee              26: Pale_Skin           36: Wearing_Lipstick
    7: Big_Nose             17: Gray_Hair           27: Pointy_Nose         37: Wearing_Necklace
    8: Black_Hair           18: Heavy_Makeup        28: Receding_Hairline   38: Wearing_Necktie
    9: Blond_Hair           19: High_Cheekbones     29: Rosy_Cheeks         39: Young
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from typing import Tuple, Optional, Dict, List
import os


# CelebA attribute names for reference
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]


class CelebAWithAttributes(Dataset):
    """
    CelebA dataset that returns both images and attribute labels.

    Reference: AAIT_Assignment_3.pdf Task 2 - Label Guidance
        "Train classifier f(x) = label"

    Args:
        root: Root directory for the dataset
        split: One of 'train', 'valid', 'test', or 'all'
        image_size: Target image size (default 64)
        download: Whether to download if not present
        return_identity: Whether to also return identity labels
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 64,
        download: bool = True,
        return_identity: bool = False,
    ):
        self.image_size = image_size
        self.return_identity = return_identity

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])

        # Load CelebA dataset with attributes
        self.dataset = CelebA(
            root=root,
            split=split,
            target_type=["attr", "identity"] if return_identity else "attr",
            transform=self.transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns image and attribute labels.

        Args:
            idx: Index of the sample

        Returns:
            If return_identity=False:
                (image, attributes) - image tensor and 40 binary attributes
            If return_identity=True:
                (image, attributes, identity) - also includes identity label
        """
        if self.return_identity:
            image, (attrs, identity) = self.dataset[idx]
            # CelebA attributes are -1/1, convert to 0/1
            attrs = ((attrs + 1) / 2).float()
            return image, attrs, identity
        else:
            image, attrs = self.dataset[idx]
            # CelebA attributes are -1/1, convert to 0/1
            attrs = ((attrs + 1) / 2).float()
            return image, attrs

    @staticmethod
    def get_attribute_name(index: int) -> str:
        """Get attribute name by index."""
        return CELEBA_ATTRIBUTES[index]

    @staticmethod
    def get_attribute_index(name: str) -> int:
        """Get attribute index by name."""
        return CELEBA_ATTRIBUTES.index(name)


def get_celeba_attribute_dataloaders(
    root: str = "./data",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
    return_identity: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for CelebA with attributes.

    Reference: AAIT_Assignment_3.pdf Task 2.2 - Label Guidance
        "Train classifier f(x) = label"

    Args:
        root: Root directory for the dataset
        image_size: Target image size (default 64)
        batch_size: Batch size for classifier training
        num_workers: Number of dataloader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        download: Whether to download if not present
        return_identity: Whether to return identity labels

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets for each split
    train_dataset = CelebAWithAttributes(
        root=root,
        split="train",
        image_size=image_size,
        download=download,
        return_identity=return_identity,
    )

    val_dataset = CelebAWithAttributes(
        root=root,
        split="valid",
        image_size=image_size,
        download=download,
        return_identity=return_identity,
    )

    test_dataset = CelebAWithAttributes(
        root=root,
        split="test",
        image_size=image_size,
        download=download,
        return_identity=return_identity,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
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


def get_images_by_identity(
    dataset: CelebAWithAttributes,
    identity_id: int,
    max_images: int = 10,
) -> List[torch.Tensor]:
    """
    Get images belonging to a specific identity.

    Reference: AAIT_Assignment_3.pdf Task 2.3 - Identity Transfer
        "Encode multiple images of each anchor → average latent"

    Args:
        dataset: CelebA dataset with identity labels
        identity_id: Identity ID to search for
        max_images: Maximum number of images to return

    Returns:
        List of image tensors for that identity
    """
    images = []
    for idx in range(len(dataset)):
        image, attrs, identity = dataset[idx]
        if identity.item() == identity_id:
            images.append(image)
            if len(images) >= max_images:
                break
    return images


def get_unique_identities(
    dataset: CelebAWithAttributes,
    n_identities: int = 10,
    min_images: int = 5,
) -> List[int]:
    """
    Get identity IDs that have at least min_images samples.

    Args:
        dataset: CelebA dataset with identity labels
        n_identities: Number of identities to return
        min_images: Minimum images required per identity

    Returns:
        List of identity IDs
    """
    from collections import Counter

    identity_counts = Counter()
    for idx in range(len(dataset)):
        _, _, identity = dataset[idx]
        identity_counts[identity.item()] += 1

    # Get identities with enough images
    valid_identities = [
        id_ for id_, count in identity_counts.items()
        if count >= min_images
    ]

    return valid_identities[:n_identities]


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test CelebA data loading with attributes.
    """
    print("Testing CelebA with attributes...")
    print("-" * 50)

    try:
        # Test basic loading
        train_loader, val_loader, test_loader = get_celeba_attribute_dataloaders(
            root="./data",
            batch_size=4,
            num_workers=0,
            download=True,
        )

        # Get a batch
        images, attrs = next(iter(train_loader))

        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Image shape: {images.shape}")
        print(f"Attributes shape: {attrs.shape}")
        print(f"Attribute value range: [{attrs.min():.1f}, {attrs.max():.1f}]")

        # Print some attribute names
        print("\nSample attributes for first image:")
        for i in range(5):
            attr_name = CelebAWithAttributes.get_attribute_name(i)
            attr_value = attrs[0, i].item()
            print(f"  {attr_name}: {attr_value:.0f}")

        print("-" * 50)
        print("All data loading tests passed!")

    except Exception as e:
        print(f"Data loading test failed: {e}")
        print("Make sure CelebA dataset is available or can be downloaded.")

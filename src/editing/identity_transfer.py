"""
Component: Identity Transfer (Task 2.3)
Reference: AAIT_Assignment_3.pdf Task 2 - Identity Transfer

Purpose:
    Morph subject faces toward anchor identities while preserving pose/expression.
    Uses the frozen VAE from Task 1.

Key implementation notes:
    Teacher's BIG HINTS:
    1. "Encode multiple images of each anchor → average latent = essence"
    2. "Try Integrated Gradients on latent space"
    3. "Try PCA-like method on latent space"

Methods implemented:
    1. Simple interpolation: z_morph = (1-α)*z_subject + α*z_anchor
    2. PCA-based: Transfer identity components while preserving pose/expression

The PCA approach:
    - First K PCA components often encode identity (global features)
    - Later components encode pose, expression, lighting (local features)
    - Transfer only identity components from anchor to subject
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA


def compute_anchor_latent(
    vae: nn.Module,
    anchor_images: List[torch.Tensor],
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Compute the average latent representation for an anchor identity.

    Reference: AAIT_Assignment_3.pdf Task 2.3 - Identity Transfer
        "Encode multiple images of each anchor → average latent = essence"

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        anchor_images: List of image tensors for the same identity
        device: Computation device

    Returns:
        Average latent vector (1, latent_dim)
    """
    vae.eval()
    latents = []

    with torch.no_grad():
        for image in anchor_images:
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(device)
            mu, _ = vae.encode(image)
            latents.append(mu)

    # Stack and compute mean
    latents = torch.cat(latents, dim=0)  # (N, latent_dim)
    anchor_latent = latents.mean(dim=0, keepdim=True)  # (1, latent_dim)

    return anchor_latent


def transfer_identity_simple(
    vae: nn.Module,
    subject_image: torch.Tensor,
    anchor_latent: torch.Tensor,
    alpha: float = 0.5,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Simple interpolation-based identity transfer.

    z_morph = (1 - α) * z_subject + α * z_anchor

    Args:
        vae: Trained VAE model (frozen)
        subject_image: Subject image tensor (1, C, H, W)
        anchor_latent: Anchor identity latent (1, latent_dim)
        alpha: Interpolation factor (0 = subject, 1 = anchor)
        device: Computation device

    Returns:
        Morphed image tensor (1, C, H, W)
    """
    vae.eval()
    subject_image = subject_image.to(device)
    anchor_latent = anchor_latent.to(device)

    with torch.no_grad():
        # Encode subject
        subject_mu, _ = vae.encode(subject_image)

        # Interpolate
        z_morph = (1 - alpha) * subject_mu + alpha * anchor_latent

        # Decode
        if hasattr(vae, 'predict_variance') and vae.predict_variance:
            morphed, _ = vae.decode(z_morph)
        else:
            morphed = vae.decode(z_morph)

        morphed = torch.clamp(morphed, 0, 1)

    return morphed


def fit_latent_pca(
    vae: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_latents: int = 5000,
    n_components: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> Tuple[PCA, torch.Tensor]:
    """
    Fit PCA on latent representations.

    Reference: AAIT_Assignment_3.pdf Task 2.3 - Identity Transfer
        "Try PCA-like method on latent space"

    Args:
        vae: Trained VAE model (frozen)
        dataloader: DataLoader for images
        n_latents: Number of latents to fit PCA on
        n_components: Number of PCA components (None = all)
        device: Computation device

    Returns:
        Tuple of (fitted PCA object, mean latent)
    """
    vae.eval()
    latents = []

    with torch.no_grad():
        images_processed = 0
        for batch in tqdm(dataloader, desc="Collecting latents for PCA"):
            if images_processed >= n_latents:
                break

            # Handle both (image,) and (image, attrs) formats
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            batch = batch.to(device)
            mu, _ = vae.encode(batch)
            latents.append(mu.cpu())

            images_processed += batch.shape[0]

    # Concatenate all latents
    latents = torch.cat(latents, dim=0)[:n_latents]  # (N, latent_dim)
    latent_dim = latents.shape[1]

    # Compute mean
    latent_mean = latents.mean(dim=0, keepdim=True)

    # Fit PCA
    if n_components is None:
        n_components = latent_dim

    pca = PCA(n_components=n_components)
    pca.fit(latents.numpy())

    print(f"PCA fitted on {latents.shape[0]} latents")
    print(f"Explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")

    return pca, latent_mean


def transfer_identity_pca(
    vae: nn.Module,
    subject_image: torch.Tensor,
    anchor_latent: torch.Tensor,
    pca: PCA,
    n_identity_components: int = 64,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    PCA-based identity transfer.

    Transfer identity-encoding components (first K) from anchor,
    while keeping pose/expression components from subject.

    Reference: AAIT_Assignment_3.pdf Task 2.3 - Identity Transfer
        "Try PCA-like method on latent space"

    Args:
        vae: Trained VAE model (frozen)
        subject_image: Subject image tensor (1, C, H, W)
        anchor_latent: Anchor identity latent (1, latent_dim)
        pca: Fitted PCA object
        n_identity_components: Number of components to treat as "identity"
        device: Computation device

    Returns:
        Morphed image tensor (1, C, H, W)
    """
    vae.eval()
    subject_image = subject_image.to(device)
    anchor_latent = anchor_latent.to(device)

    with torch.no_grad():
        # Encode subject
        subject_mu, _ = vae.encode(subject_image)

        # Transform to PCA space
        subject_pca = pca.transform(subject_mu.cpu().numpy())
        anchor_pca = pca.transform(anchor_latent.cpu().numpy())

        # Create morphed PCA representation
        morphed_pca = subject_pca.copy()

        # Transfer identity components (first K)
        morphed_pca[:, :n_identity_components] = anchor_pca[:, :n_identity_components]

        # Transform back to latent space
        morphed_latent = pca.inverse_transform(morphed_pca)
        morphed_latent = torch.tensor(morphed_latent, dtype=torch.float32, device=device)

        # Decode
        if hasattr(vae, 'predict_variance') and vae.predict_variance:
            morphed, _ = vae.decode(morphed_latent)
        else:
            morphed = vae.decode(morphed_latent)

        morphed = torch.clamp(morphed, 0, 1)

    return morphed


def create_identity_grid_simple(
    vae: nn.Module,
    subject_images: torch.Tensor,
    anchor_latents: List[torch.Tensor],
    alphas: List[float] = [0.3, 0.5, 0.7],
    device: torch.device = torch.device("cuda"),
) -> Dict[str, torch.Tensor]:
    """
    Create identity transfer grid using simple interpolation.

    Args:
        vae: Trained VAE model (frozen)
        subject_images: Subject images (N, C, H, W)
        anchor_latents: List of anchor latent vectors
        alphas: Interpolation factors to use
        device: Computation device

    Returns:
        Dictionary with:
            - "subjects": Original subject images
            - "morphed_anchor_{i}_alpha_{a}": Morphed images for each anchor and alpha
    """
    vae.eval()
    n_subjects = subject_images.shape[0]
    n_anchors = len(anchor_latents)

    results = {
        "subjects": subject_images.clone(),
    }

    for anchor_idx, anchor_latent in enumerate(anchor_latents):
        for alpha in alphas:
            key = f"morphed_anchor_{anchor_idx}_alpha_{alpha:.1f}"
            morphed_list = []

            for i in range(n_subjects):
                morphed = transfer_identity_simple(
                    vae=vae,
                    subject_image=subject_images[i:i+1],
                    anchor_latent=anchor_latent,
                    alpha=alpha,
                    device=device,
                )
                morphed_list.append(morphed.cpu())

            results[key] = torch.cat(morphed_list, dim=0)

    return results


def create_identity_grid_pca(
    vae: nn.Module,
    subject_images: torch.Tensor,
    anchor_latents: List[torch.Tensor],
    pca: PCA,
    n_identity_components: int = 64,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, torch.Tensor]:
    """
    Create identity transfer grid using PCA method.

    Reference: AAIT_Assignment_3.pdf Task 2.3
        "3 anchor people × 8 different subjects"

    Args:
        vae: Trained VAE model (frozen)
        subject_images: Subject images (N, C, H, W)
        anchor_latents: List of anchor latent vectors
        pca: Fitted PCA object
        n_identity_components: Number of identity components
        device: Computation device

    Returns:
        Dictionary with:
            - "subjects": Original subject images
            - "morphed_anchor_{i}": Morphed images for each anchor
    """
    vae.eval()
    n_subjects = subject_images.shape[0]
    n_anchors = len(anchor_latents)

    results = {
        "subjects": subject_images.clone(),
    }

    for anchor_idx, anchor_latent in enumerate(anchor_latents):
        morphed_list = []

        for i in tqdm(range(n_subjects), desc=f"Processing anchor {anchor_idx + 1}"):
            morphed = transfer_identity_pca(
                vae=vae,
                subject_image=subject_images[i:i+1],
                anchor_latent=anchor_latent,
                pca=pca,
                n_identity_components=n_identity_components,
                device=device,
            )
            morphed_list.append(morphed.cpu())

        results[f"morphed_anchor_{anchor_idx}"] = torch.cat(morphed_list, dim=0)

    return results


def create_identity_grid(
    vae: nn.Module,
    subject_images: torch.Tensor,
    anchor_latents: List[torch.Tensor],
    method: str = "pca",
    pca: Optional[PCA] = None,
    n_identity_components: int = 64,
    alphas: List[float] = [0.3, 0.5, 0.7],
    device: torch.device = torch.device("cuda"),
) -> Dict[str, torch.Tensor]:
    """
    Create identity transfer grid.

    Args:
        vae: Trained VAE model (frozen)
        subject_images: Subject images (N, C, H, W)
        anchor_latents: List of anchor latent vectors
        method: "simple" or "pca"
        pca: Fitted PCA object (required for method="pca")
        n_identity_components: Number of identity components (for PCA)
        alphas: Interpolation factors (for simple method)
        device: Computation device

    Returns:
        Dictionary with results
    """
    if method == "pca":
        if pca is None:
            raise ValueError("PCA object required for method='pca'")
        return create_identity_grid_pca(
            vae, subject_images, anchor_latents, pca,
            n_identity_components, device
        )
    else:
        return create_identity_grid_simple(
            vae, subject_images, anchor_latents, alphas, device
        )


def visualize_identity_transfer_pca(
    results: Dict[str, torch.Tensor],
    n_anchors: int,
    save_path: Path,
    anchor_images: Optional[List[torch.Tensor]] = None,
    title: str = "Identity Transfer (PCA)",
):
    """
    Visualize identity transfer results for PCA method.

    Args:
        results: Dictionary from create_identity_grid_pca
        n_anchors: Number of anchors
        save_path: Path to save figure
        anchor_images: Optional list of anchor representative images
        title: Figure title
    """
    subjects = results["subjects"]
    n_subjects = subjects.shape[0]

    # Create grid: subjects (rows) × (original + anchors) (cols)
    n_cols = 1 + n_anchors
    fig, axes = plt.subplots(n_subjects, n_cols, figsize=(n_cols * 1.5, n_subjects * 1.5))

    if n_subjects == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_subjects):
        # Subject original
        img = subjects[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Subject", fontsize=9)

        # Morphed toward each anchor
        for j in range(n_anchors):
            morphed = results[f"morphed_anchor_{j}"][i].cpu().permute(1, 2, 0).numpy()
            morphed = np.clip(morphed, 0, 1)
            axes[i, j + 1].imshow(morphed)
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f"→ Anchor {j + 1}", fontsize=9)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_identity_transfer_simple(
    results: Dict[str, torch.Tensor],
    n_anchors: int,
    alphas: List[float],
    save_path: Path,
    title: str = "Identity Transfer (Interpolation)",
):
    """
    Visualize identity transfer results for simple interpolation method.

    Args:
        results: Dictionary from create_identity_grid_simple
        n_anchors: Number of anchors
        alphas: Alpha values used
        save_path: Path to save figure
        title: Figure title
    """
    subjects = results["subjects"]
    n_subjects = subjects.shape[0]

    # For each anchor, create a separate grid
    for anchor_idx in range(n_anchors):
        n_cols = 1 + len(alphas)  # Subject + morphed at each alpha
        fig, axes = plt.subplots(n_subjects, n_cols, figsize=(n_cols * 1.5, n_subjects * 1.5))

        if n_subjects == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_subjects):
            # Subject original
            img = subjects[i].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title("Original", fontsize=9)

            # Morphed at each alpha
            for j, alpha in enumerate(alphas):
                key = f"morphed_anchor_{anchor_idx}_alpha_{alpha:.1f}"
                morphed = results[key][i].permute(1, 2, 0).numpy()
                morphed = np.clip(morphed, 0, 1)
                axes[i, j + 1].imshow(morphed)
                axes[i, j + 1].axis('off')
                if i == 0:
                    axes[i, j + 1].set_title(f"α={alpha:.1f}", fontsize=9)

        plt.suptitle(f"{title} - Anchor {anchor_idx + 1}", fontsize=12)
        plt.tight_layout()

        # Save with anchor index in filename
        save_path_anchor = save_path.parent / f"{save_path.stem}_anchor{anchor_idx}{save_path.suffix}"
        plt.savefig(save_path_anchor, dpi=150, bbox_inches='tight')
        plt.close()


def visualize_identity_transfer_all(
    results: Dict[str, torch.Tensor],
    n_anchors: int,
    output_dir: Path,
    method: str = "pca",
    alphas: Optional[List[float]] = None,
    anchor_images: Optional[List[torch.Tensor]] = None,
):
    """
    Save all identity transfer visualizations.

    Args:
        results: Dictionary from create_identity_grid
        n_anchors: Number of anchors
        output_dir: Directory to save figures
        method: "simple" or "pca"
        alphas: Alpha values (for simple method)
        anchor_images: Optional anchor representative images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if method == "pca":
        visualize_identity_transfer_pca(
            results, n_anchors,
            output_dir / "identity_transfer_pca.png",
            anchor_images,
            "Identity Transfer (PCA)"
        )

        # Also save individual anchor visualizations
        subjects = results["subjects"]
        n_subjects = subjects.shape[0]

        for anchor_idx in range(n_anchors):
            fig, axes = plt.subplots(2, n_subjects, figsize=(n_subjects * 1.5, 3))

            for i in range(n_subjects):
                # Subject
                img = subjects[i].cpu().permute(1, 2, 0).numpy()
                axes[0, i].imshow(np.clip(img, 0, 1))
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel("Subject", fontsize=10)

                # Morphed
                morphed = results[f"morphed_anchor_{anchor_idx}"][i].cpu().permute(1, 2, 0).numpy()
                axes[1, i].imshow(np.clip(morphed, 0, 1))
                axes[1, i].axis('off')
                if i == 0:
                    axes[1, i].set_ylabel(f"→ Anchor {anchor_idx+1}", fontsize=10)

            plt.suptitle(f"Identity Transfer to Anchor {anchor_idx + 1}", fontsize=12)
            plt.tight_layout()
            plt.savefig(output_dir / f"anchor_{anchor_idx}.png", dpi=150, bbox_inches='tight')
            plt.close()

    else:
        visualize_identity_transfer_simple(
            results, n_anchors, alphas,
            output_dir / "identity_transfer_simple.png",
            "Identity Transfer (Interpolation)"
        )

    print(f"Saved identity transfer visualizations to {output_dir}")


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test identity transfer with dummy models.
    """
    print("Testing Identity Transfer...")
    print("-" * 50)

    # Create dummy VAE
    class DummyVAE(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.predict_variance = False

        def encode(self, x):
            batch_size = x.shape[0]
            mu = torch.randn(batch_size, self.latent_dim)
            log_var = torch.zeros(batch_size, self.latent_dim)
            return mu, log_var

        def decode(self, z):
            batch_size = z.shape[0]
            return torch.rand(batch_size, 3, 64, 64)

    device = torch.device("cpu")
    vae = DummyVAE().to(device)

    # Test compute_anchor_latent
    print("Testing compute_anchor_latent...")
    anchor_images = [torch.rand(1, 3, 64, 64) for _ in range(5)]
    anchor_latent = compute_anchor_latent(vae, anchor_images, device)
    print(f"  Anchor latent shape: {anchor_latent.shape}")
    assert anchor_latent.shape == (1, 256)
    print("  compute_anchor_latent passed!")

    # Test transfer_identity_simple
    print("\nTesting transfer_identity_simple...")
    subject_image = torch.rand(1, 3, 64, 64)
    morphed = transfer_identity_simple(vae, subject_image, anchor_latent, alpha=0.5, device=device)
    print(f"  Morphed shape: {morphed.shape}")
    assert morphed.shape == (1, 3, 64, 64)
    print("  transfer_identity_simple passed!")

    # Test create_identity_grid_simple
    print("\nTesting create_identity_grid_simple...")
    subject_images = torch.rand(4, 3, 64, 64)  # 4 subjects
    anchor_latents = [torch.randn(1, 256) for _ in range(3)]  # 3 anchors
    alphas = [0.3, 0.5, 0.7]

    results = create_identity_grid_simple(
        vae, subject_images, anchor_latents, alphas, device
    )

    print(f"  Subjects shape: {results['subjects'].shape}")
    for anchor_idx in range(3):
        for alpha in alphas:
            key = f"morphed_anchor_{anchor_idx}_alpha_{alpha:.1f}"
            print(f"  {key} shape: {results[key].shape}")

    print("  create_identity_grid_simple passed!")

    print("\n" + "-" * 50)
    print("All identity transfer tests passed!")

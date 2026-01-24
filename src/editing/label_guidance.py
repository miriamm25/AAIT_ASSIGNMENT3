"""
Component: Label Guidance (Task 2.2)
Reference: AAIT_Assignment_3.pdf Task 2 - Label Guidance

Purpose:
    Modify latent vectors so that a classifier assigns desired labels.
    Uses gradient descent on the latent space with a frozen VAE and classifier.

Key implementation notes:
    Teacher's BIG HINT:
    - "Train classifier f(x) = label"
    - "Set z as optimizable parameter"
    - "Gradient descent: minimize CrossEntropy(f(dec(z)), target_label)"

Algorithm:
    1. Encode original image to get z_0
    2. Set z as optimizable parameter (z = z_0.clone(), requires_grad=True)
    3. For N steps:
        a. Decode z to get reconstructed image
        b. Pass through classifier to get predicted labels
        c. Compute loss = BCE(predicted[target_attr], target_value)
        d. Optional: Add regularization ||z - z_0||^2 to stay close
        e. Backpropagate and update z
    4. Final reconstruction = decode(z)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm

from .classifier import AttributeClassifier


def optimize_latent_for_label(
    vae: nn.Module,
    classifier: AttributeClassifier,
    image: torch.Tensor,
    target_attr_idx: int,
    target_value: float = 1.0,
    n_steps: int = 200,
    lr: float = 0.01,
    regularization_weight: float = 0.1,
    device: torch.device = torch.device("cuda"),
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
    """
    Optimize latent vector to achieve target attribute.

    Reference: AAIT_Assignment_3.pdf Task 2.2 - Label Guidance
        "Set z as optimizable parameter"
        "Gradient descent: minimize CrossEntropy(f(dec(z)), target_label)"

    Args:
        vae: Trained VAE model (frozen, in eval mode)
        classifier: Trained attribute classifier (frozen, in eval mode)
        image: Input image tensor (1, C, H, W)
        target_attr_idx: Index of attribute to modify
        target_value: Target value for the attribute (0.0 or 1.0)
        n_steps: Number of optimization steps
        lr: Learning rate for latent optimization
        regularization_weight: Weight for ||z - z_0||^2 regularization
        device: Computation device
        verbose: Whether to print progress

    Returns:
        Tuple of:
            - original_image: Original reconstruction (1, C, H, W)
            - modified_image: Modified reconstruction (1, C, H, W)
            - history: Dict with loss history
    """
    vae.eval()
    classifier.eval()

    # Move to device
    image = image.to(device)

    # Encode original image
    with torch.no_grad():
        mu, _ = vae.encode(image)
        z_original = mu.clone()

    # Set z as optimizable parameter
    z = z_original.clone().requires_grad_(True)

    # Optimizer for latent
    optimizer = optim.Adam([z], lr=lr)

    # Loss function
    bce_loss = nn.BCELoss()

    # Target tensor
    target = torch.tensor([[target_value]], device=device)

    # History
    history = {
        "total_loss": [],
        "attr_loss": [],
        "reg_loss": [],
        "attr_prob": [],
    }

    # Optimization loop
    iterator = range(n_steps)
    if verbose:
        iterator = tqdm(iterator, desc="Optimizing latent")

    for step in iterator:
        optimizer.zero_grad()

        # Decode current z
        if hasattr(vae, 'predict_variance') and vae.predict_variance:
            recon, _ = vae.decode(z)
        else:
            recon = vae.decode(z)

        # Clamp to valid range
        recon = torch.clamp(recon, 0, 1)

        # Get classifier prediction for target attribute
        attr_probs = classifier(recon)
        attr_prob = attr_probs[:, target_attr_idx:target_attr_idx+1]

        # Attribute loss (BCE)
        attr_loss = bce_loss(attr_prob, target)

        # Regularization loss (stay close to original)
        reg_loss = torch.mean((z - z_original) ** 2)

        # Total loss
        total_loss = attr_loss + regularization_weight * reg_loss

        # Backward and update
        total_loss.backward()
        optimizer.step()

        # Record history
        history["total_loss"].append(total_loss.item())
        history["attr_loss"].append(attr_loss.item())
        history["reg_loss"].append(reg_loss.item())
        history["attr_prob"].append(attr_prob.item())

    # Get final reconstructions
    with torch.no_grad():
        # Original reconstruction
        if hasattr(vae, 'predict_variance') and vae.predict_variance:
            original_recon, _ = vae.decode(z_original)
        else:
            original_recon = vae.decode(z_original)

        # Modified reconstruction
        if hasattr(vae, 'predict_variance') and vae.predict_variance:
            modified_recon, _ = vae.decode(z)
        else:
            modified_recon = vae.decode(z)

        # Clamp to valid range
        original_recon = torch.clamp(original_recon, 0, 1)
        modified_recon = torch.clamp(modified_recon, 0, 1)

    return original_recon, modified_recon, history


def create_label_guidance_grid(
    vae: nn.Module,
    classifier: AttributeClassifier,
    images: torch.Tensor,
    target_attributes: List[Dict],
    n_steps: int = 200,
    lr: float = 0.01,
    regularization_weight: float = 0.1,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, torch.Tensor]:
    """
    Create label guidance visualizations for multiple images and attributes.

    Reference: AAIT_Assignment_3.pdf Task 2.2
        "4 different label changes × 8 samples"

    Args:
        vae: Trained VAE model (frozen)
        classifier: Trained attribute classifier (frozen)
        images: Input images (N, C, H, W)
        target_attributes: List of dicts with keys:
            - "name": Attribute name
            - "index": Attribute index
            - "target_value": Target value (0 or 1)
        n_steps: Number of optimization steps
        lr: Learning rate
        regularization_weight: Regularization weight
        device: Computation device

    Returns:
        Dictionary with:
            - "originals": Original reconstructions (N, C, H, W)
            - "modified_{attr_name}": Modified images for each attribute (N, C, H, W)
            - "histories": Optimization histories
    """
    vae.eval()
    classifier.eval()

    n_samples = images.shape[0]
    n_attrs = len(target_attributes)

    results = {
        "originals": [],
        "histories": {},
    }

    for attr_info in target_attributes:
        attr_name = attr_info["name"]
        results[f"modified_{attr_name}"] = []
        results["histories"][attr_name] = []

    # Process each image
    for i in tqdm(range(n_samples), desc="Processing images"):
        image = images[i:i+1]  # (1, C, H, W)

        # Store original (only need to compute once)
        with torch.no_grad():
            mu, _ = vae.encode(image.to(device))
            if hasattr(vae, 'predict_variance') and vae.predict_variance:
                original, _ = vae.decode(mu)
            else:
                original = vae.decode(mu)
            results["originals"].append(torch.clamp(original, 0, 1).cpu())

        # Apply each attribute modification
        for attr_info in target_attributes:
            attr_name = attr_info["name"]
            attr_idx = attr_info["index"]
            target_val = attr_info["target_value"]

            _, modified, history = optimize_latent_for_label(
                vae=vae,
                classifier=classifier,
                image=image,
                target_attr_idx=attr_idx,
                target_value=target_val,
                n_steps=n_steps,
                lr=lr,
                regularization_weight=regularization_weight,
                device=device,
                verbose=False,
            )

            results[f"modified_{attr_name}"].append(modified.cpu())
            results["histories"][attr_name].append(history)

    # Stack results
    results["originals"] = torch.cat(results["originals"], dim=0)
    for attr_info in target_attributes:
        attr_name = attr_info["name"]
        results[f"modified_{attr_name}"] = torch.cat(
            results[f"modified_{attr_name}"], dim=0
        )

    return results


def visualize_label_guidance(
    results: Dict[str, torch.Tensor],
    attribute_names: List[str],
    save_path: Path,
    title: str = "Label Guidance Results",
):
    """
    Visualize label guidance results as before/after grid.

    Args:
        results: Dictionary from create_label_guidance_grid
        attribute_names: List of attribute names
        save_path: Path to save figure
        title: Figure title
    """
    originals = results["originals"]
    n_samples = originals.shape[0]
    n_attrs = len(attribute_names)

    # Create grid: samples (rows) × (original + attributes) (cols)
    n_cols = 1 + n_attrs  # Original + modified versions
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(n_cols * 1.5, n_samples * 1.5))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Original
        img = originals[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=9)

        # Modified versions
        for j, attr_name in enumerate(attribute_names):
            modified = results[f"modified_{attr_name}"][i].permute(1, 2, 0).numpy()
            modified = np.clip(modified, 0, 1)
            axes[i, j + 1].imshow(modified)
            axes[i, j + 1].axis('off')
            if i == 0:
                axes[i, j + 1].set_title(f"+{attr_name}", fontsize=9)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_label_guidance_detailed(
    results: Dict[str, torch.Tensor],
    attribute_names: List[str],
    output_dir: Path,
):
    """
    Save detailed visualizations for each attribute.

    Args:
        results: Dictionary from create_label_guidance_grid
        attribute_names: List of attribute names
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    originals = results["originals"]
    n_samples = originals.shape[0]

    # Save combined grid
    visualize_label_guidance(
        results, attribute_names,
        output_dir / "all_attributes.png",
        "Label Guidance - All Attributes"
    )

    # Save per-attribute grids
    for attr_name in attribute_names:
        modified = results[f"modified_{attr_name}"]

        # Create before/after grid
        fig, axes = plt.subplots(n_samples, 2, figsize=(4, n_samples * 2))

        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # Original
            orig_img = originals[i].permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img, 0, 1)
            axes[i, 0].imshow(orig_img)
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title("Before", fontsize=10)

            # Modified
            mod_img = modified[i].permute(1, 2, 0).numpy()
            mod_img = np.clip(mod_img, 0, 1)
            axes[i, 1].imshow(mod_img)
            axes[i, 1].axis('off')
            if i == 0:
                axes[i, 1].set_title(f"After (+{attr_name})", fontsize=10)

        plt.suptitle(f"Label Guidance: {attr_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"attribute_{attr_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved label guidance visualizations to {output_dir}")


def plot_optimization_history(
    history: Dict[str, List[float]],
    save_path: Path,
    title: str = "Latent Optimization",
):
    """
    Plot optimization history.

    Args:
        history: Dictionary with loss histories
        save_path: Path to save figure
        title: Figure title
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Total loss
    axes[0, 0].plot(history["total_loss"])
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Total Loss")

    # Attribute loss
    axes[0, 1].plot(history["attr_loss"])
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Attribute Loss")
    axes[0, 1].set_title("Attribute Loss (BCE)")

    # Regularization loss
    axes[1, 0].plot(history["reg_loss"])
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Reg Loss")
    axes[1, 0].set_title("Regularization Loss")

    # Attribute probability
    axes[1, 1].plot(history["attr_prob"])
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Target')
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Probability")
    axes[1, 1].set_title("Target Attribute Probability")
    axes[1, 1].legend()

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test label guidance with dummy models.
    """
    print("Testing Label Guidance...")
    print("-" * 50)

    # Create dummy VAE
    class DummyVAE(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.latent_dim = latent_dim
            self.predict_variance = False
            self.fc_enc = nn.Linear(3 * 64 * 64, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, 3 * 64 * 64)

        def encode(self, x):
            x_flat = x.view(x.shape[0], -1)
            mu = self.fc_enc(x_flat)
            log_var = torch.zeros_like(mu)
            return mu, log_var

        def decode(self, z):
            x_flat = torch.sigmoid(self.fc_dec(z))
            return x_flat.view(-1, 3, 64, 64)

    # Create dummy classifier
    class DummyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 64 * 64, 40)
            self.n_attributes = 40

        def forward(self, x):
            x_flat = x.view(x.shape[0], -1)
            return torch.sigmoid(self.fc(x_flat))

    device = torch.device("cpu")
    vae = DummyVAE().to(device)
    classifier = DummyClassifier().to(device)

    # Test optimize_latent_for_label
    print("Testing optimize_latent_for_label...")
    image = torch.rand(1, 3, 64, 64)
    original, modified, history = optimize_latent_for_label(
        vae=vae,
        classifier=classifier,
        image=image,
        target_attr_idx=15,  # Eyeglasses
        target_value=1.0,
        n_steps=10,
        lr=0.1,
        device=device,
        verbose=False,
    )

    print(f"  Original shape: {original.shape}")
    print(f"  Modified shape: {modified.shape}")
    print(f"  History keys: {list(history.keys())}")
    print(f"  Final attr probability: {history['attr_prob'][-1]:.4f}")

    assert original.shape == (1, 3, 64, 64)
    assert modified.shape == (1, 3, 64, 64)
    print("  optimize_latent_for_label passed!")

    # Test create_label_guidance_grid
    print("\nTesting create_label_guidance_grid...")
    images = torch.rand(2, 3, 64, 64)  # 2 samples
    target_attributes = [
        {"name": "Eyeglasses", "index": 15, "target_value": 1.0},
        {"name": "Smiling", "index": 31, "target_value": 1.0},
    ]

    results = create_label_guidance_grid(
        vae=vae,
        classifier=classifier,
        images=images,
        target_attributes=target_attributes,
        n_steps=5,
        lr=0.1,
        device=device,
    )

    print(f"  Originals shape: {results['originals'].shape}")
    print(f"  Modified_Eyeglasses shape: {results['modified_Eyeglasses'].shape}")
    print(f"  Modified_Smiling shape: {results['modified_Smiling'].shape}")

    assert results["originals"].shape == (2, 3, 64, 64)
    assert results["modified_Eyeglasses"].shape == (2, 3, 64, 64)
    assert results["modified_Smiling"].shape == (2, 3, 64, 64)
    print("  create_label_guidance_grid passed!")

    print("\n" + "-" * 50)
    print("All label guidance tests passed!")

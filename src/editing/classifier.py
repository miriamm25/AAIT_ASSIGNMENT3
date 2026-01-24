"""
Component: Attribute Classifier for Task 2.2 - Label Guidance
Reference: AAIT_Assignment_3.pdf Task 2 - Label Guidance

Purpose:
    Train a classifier f(x) = label on CelebA attributes.
    This classifier is used to guide latent optimization toward desired attributes.

Key implementation notes:
    - "Train classifier f(x) = label" (from teacher's hint)
    - Uses ResNet18 backbone (pretrained on ImageNet)
    - Multi-label classification with BCE loss
    - CelebA has 40 binary attributes

Teacher's BIG HINT:
    "Set z as optimizable parameter"
    "Gradient descent: minimize CrossEntropy(f(dec(z)), target_label)"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from tqdm import tqdm
import json


class AttributeClassifier(nn.Module):
    """
    Multi-label attribute classifier for CelebA.

    Reference: AAIT_Assignment_3.pdf Task 2.2 - Label Guidance
        "Train classifier f(x) = label"

    Uses ResNet18 backbone pretrained on ImageNet with a new FC head
    for 40 CelebA attributes. Multi-label classification with sigmoid output.

    Args:
        n_attributes: Number of attributes to predict (default 40 for CelebA)
        pretrained: Whether to use pretrained ResNet18 weights
        freeze_backbone: Whether to freeze backbone weights
    """

    def __init__(
        self,
        n_attributes: int = 40,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.n_attributes = n_attributes

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Replace final FC layer for multi-label classification
        in_features = self.backbone.fc.in_features  # 512 for ResNet18
        self.backbone.fc = nn.Linear(in_features, n_attributes)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze final FC layer
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W) in [0, 1] range

        Returns:
            Predicted attribute probabilities (B, n_attributes) in [0, 1]
        """
        # Normalize to ImageNet stats (ResNet expects this)
        # Note: Input is [0, 1], we need to normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x - mean) / std

        # Get logits
        logits = self.backbone(x_normalized)

        # Apply sigmoid for multi-label
        return torch.sigmoid(logits)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits (before sigmoid).

        Useful for loss computation where BCEWithLogitsLoss is more stable.

        Args:
            x: Input images (B, 3, H, W) in [0, 1] range

        Returns:
            Raw logits (B, n_attributes)
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x - mean) / std

        return self.backbone(x_normalized)


def train_classifier(
    model: AttributeClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda"),
    save_path: Optional[Path] = None,
    log_interval: int = 100,
) -> Dict[str, List[float]]:
    """
    Train the attribute classifier.

    Reference: AAIT_Assignment_3.pdf Task 2.2
        "Train classifier f(x) = label"

    Args:
        model: AttributeClassifier model
        train_loader: Training dataloader (yields image, attributes)
        val_loader: Validation dataloader
        epochs: Number of training epochs
        lr: Learning rate
        device: Computation device
        save_path: Path to save best model
        log_interval: Log every N batches

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    model.train()

    # BCE loss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images, attrs = batch[0], batch[1]
            images = images.to(device)
            attrs = attrs.to(device)

            optimizer.zero_grad()

            # Forward pass (get logits for numerical stability)
            logits = model.get_logits(images)

            # Compute loss
            loss = criterion(logits, attrs)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / n_batches
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_loss, val_acc = evaluate_classifier(model, val_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Update scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_classifier(model, save_path, history, epoch)
            print(f"  Saved best model to {save_path}")

    return history


def evaluate_classifier(
    model: AttributeClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate classifier on a dataset.

    Args:
        model: AttributeClassifier model
        dataloader: DataLoader (yields image, attributes)
        criterion: Loss function
        device: Computation device

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images, attrs = batch[0], batch[1]
            images = images.to(device)
            attrs = attrs.to(device)

            logits = model.get_logits(images)
            loss = criterion(logits, attrs)

            total_loss += loss.item() * images.shape[0]

            # Compute accuracy (threshold at 0.5)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == attrs).sum().item()
            total += attrs.numel()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate_per_attribute(
    model: AttributeClassifier,
    dataloader: DataLoader,
    device: torch.device,
    attribute_names: Optional[List[str]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate classifier accuracy per attribute.

    Args:
        model: AttributeClassifier model
        dataloader: DataLoader (yields image, attributes)
        device: Computation device
        attribute_names: Optional list of attribute names

    Returns:
        Dictionary mapping attribute index to metrics
    """
    model.eval()
    n_attributes = model.n_attributes

    # Accumulators
    tp = torch.zeros(n_attributes)  # True positives
    tn = torch.zeros(n_attributes)  # True negatives
    fp = torch.zeros(n_attributes)  # False positives
    fn = torch.zeros(n_attributes)  # False negatives

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating per-attribute"):
            images, attrs = batch[0], batch[1]
            images = images.to(device)
            attrs = attrs.to(device)

            preds = (model(images) > 0.5).float()

            tp += ((preds == 1) & (attrs == 1)).sum(dim=0).cpu()
            tn += ((preds == 0) & (attrs == 0)).sum(dim=0).cpu()
            fp += ((preds == 1) & (attrs == 0)).sum(dim=0).cpu()
            fn += ((preds == 0) & (attrs == 1)).sum(dim=0).cpu()

    results = {}
    for i in range(n_attributes):
        accuracy = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0

        name = attribute_names[i] if attribute_names else f"attr_{i}"
        results[i] = {
            "name": name,
            "accuracy": accuracy.item(),
            "precision": precision.item() if isinstance(precision, torch.Tensor) else precision,
            "recall": recall.item() if isinstance(recall, torch.Tensor) else recall,
        }

    return results


def save_classifier(
    model: AttributeClassifier,
    save_path: Path,
    history: Optional[Dict] = None,
    epoch: Optional[int] = None,
):
    """
    Save classifier checkpoint.

    Args:
        model: AttributeClassifier model
        save_path: Path to save checkpoint
        history: Optional training history
        epoch: Optional epoch number
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "n_attributes": model.n_attributes,
    }

    if history is not None:
        checkpoint["history"] = history

    if epoch is not None:
        checkpoint["epoch"] = epoch

    torch.save(checkpoint, save_path)


def load_classifier(
    checkpoint_path: Path,
    device: torch.device = torch.device("cuda"),
) -> AttributeClassifier:
    """
    Load classifier from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded AttributeClassifier model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = AttributeClassifier(
        n_attributes=checkpoint["n_attributes"],
        pretrained=False,  # We're loading weights
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def predict_attributes(
    model: AttributeClassifier,
    images: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict attributes for given images.

    Args:
        model: AttributeClassifier model
        images: Input images (B, 3, H, W) in [0, 1] range
        threshold: Classification threshold

    Returns:
        Tuple of (probabilities, binary_predictions)
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)

    with torch.no_grad():
        probs = model(images)
        preds = (probs > threshold).float()

    return probs, preds


# =============================================================================
# Verification
# =============================================================================
if __name__ == "__main__":
    """
    Verification: Test classifier architecture and forward pass.
    """
    print("Testing AttributeClassifier...")
    print("-" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create classifier
    model = AttributeClassifier(
        n_attributes=40,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.rand(4, 3, 64, 64, device=device)
    probs = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {probs.shape}")
    print(f"  Output range: [{probs.min():.3f}, {probs.max():.3f}]")

    assert probs.shape == (4, 40)
    assert probs.min() >= 0 and probs.max() <= 1
    print("  Forward pass passed!")

    # Test logits
    print("\nTesting get_logits...")
    logits = model.get_logits(x)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

    assert logits.shape == (4, 40)
    print("  get_logits passed!")

    # Test save/load
    print("\nTesting save/load...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = Path(f.name)

    save_classifier(model, save_path, history={"test": [1, 2, 3]})
    loaded_model = load_classifier(save_path, device)

    # Compare outputs
    with torch.no_grad():
        orig_out = model(x)
        loaded_out = loaded_model(x)

    assert torch.allclose(orig_out, loaded_out)
    print("  Save/load passed!")

    # Cleanup
    save_path.unlink()

    print("\n" + "-" * 50)
    print("All AttributeClassifier tests passed!")

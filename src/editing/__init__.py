"""
Component: Editing Module for Task 2 - Latent Space Manipulation
Reference: AAIT_Assignment_3.pdf Task 2

Purpose:
    This module provides tools for editing images by manipulating the VAE's
    latent space. Uses the FROZEN trained VAE from Task 1 (no retraining).

Sub-tasks:
    1. Feature Amplification - Find meaningful latent dimensions
    2. Label Guidance - Optimize latent to achieve target attributes
    3. Identity Transfer - Morph faces toward anchor identities

Key implementation notes:
    - VAE is always in eval() mode and frozen
    - Only the encoder mean (mu) is used, not sampled z
    - Classifier is trained separately for label guidance
"""

from .feature_amplification import (
    find_meaningful_dimensions,
    amplify_dimension,
    create_amplification_grid,
)
from .label_guidance import (
    optimize_latent_for_label,
    create_label_guidance_grid,
)
from .identity_transfer import (
    compute_anchor_latent,
    transfer_identity_simple,
    transfer_identity_pca,
    create_identity_grid,
)
from .classifier import (
    AttributeClassifier,
    train_classifier,
    load_classifier,
)

__all__ = [
    # Feature Amplification
    "find_meaningful_dimensions",
    "amplify_dimension",
    "create_amplification_grid",
    # Label Guidance
    "optimize_latent_for_label",
    "create_label_guidance_grid",
    # Identity Transfer
    "compute_anchor_latent",
    "transfer_identity_simple",
    "transfer_identity_pca",
    "create_identity_grid",
    # Classifier
    "AttributeClassifier",
    "train_classifier",
    "load_classifier",
]

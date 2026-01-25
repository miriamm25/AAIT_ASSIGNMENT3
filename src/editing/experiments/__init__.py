"""
Component: Feature Amplification Experiments Module
Reference: Task 2.1 Improvement Plan

Purpose:
    This module contains experimental improvements for Task 2.1 Feature Amplification:
    1. Attribute-correlated dimension discovery
    2. Wider alpha range testing
    3. Manual dimension exploration

These experiments complement the baseline variance-based approach.
"""

from .attribute_correlation import (
    find_attribute_correlated_dimensions,
    compute_all_correlations,
)

from .alpha_range import (
    compare_alpha_ranges,
    generate_comparison_grid,
)

from .dimension_explorer import (
    explore_top_n_dimensions,
    generate_dimension_report,
)

from .identity_utils import (
    find_rich_identities,
    get_images_by_identity_id,
    compute_true_anchor_latent,
)

__all__ = [
    # Attribute correlation
    "find_attribute_correlated_dimensions",
    "compute_all_correlations",
    # Alpha range
    "compare_alpha_ranges",
    "generate_comparison_grid",
    # Dimension explorer
    "explore_top_n_dimensions",
    "generate_dimension_report",
    # Identity utils
    "find_rich_identities",
    "get_images_by_identity_id",
    "compute_true_anchor_latent",
]

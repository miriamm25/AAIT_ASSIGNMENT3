"""
Component: Feature Amplification Experiments Runner
Reference: Task 2.1 Improvement Plan

Purpose:
    Run all three feature amplification improvement experiments:
    1. Attribute-correlated dimension discovery
    2. Wider alpha range testing [-5, 5]
    3. Manual exploration of top-20 dimensions

Usage:
    python scripts/experiments/run_feature_amplification_experiments.py
    python scripts/experiments/run_feature_amplification_experiments.py --exp 1  # Only exp 1
    python scripts/experiments/run_feature_amplification_experiments.py --exp 2  # Only exp 2
    python scripts/experiments/run_feature_amplification_experiments.py --exp 3  # Only exp 3

Prerequisites:
    - Trained VAE from Task 1 (checkpoint required)
    - CelebA dataset with attributes
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add src directory to path
script_dir = Path(__file__).parent.absolute()
project_dir = script_dir.parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

from models.vae import VAE
from data.celeba_with_attributes import (
    get_celeba_attribute_dataloaders,
    CELEBA_ATTRIBUTES,
)
from editing.feature_amplification import (
    find_meaningful_dimensions,
    create_amplification_grid,
    visualize_amplification_all_samples,
)
from editing.experiments.attribute_correlation import (
    find_attribute_correlated_dimensions,
    compute_all_correlations,
)
from editing.experiments.alpha_range import (
    compare_alpha_ranges,
    visualize_alpha_range_comparison,
)
from editing.experiments.dimension_explorer import (
    explore_top_n_dimensions,
    generate_dimension_report,
    create_selection_template,
)


# =============================================================================
# Configuration
# =============================================================================

# Default config - can be overridden via command line or config file
DEFAULT_CONFIG = {
    "vae": {
        "checkpoint_path": "outputs/checkpoints/best_model.pt",
        "in_channels": 3,
        "base_channels": 64,
        "latent_dim": 256,
        "blocks_per_level": 2,
        "image_size": 64,
        "predict_variance": False,
    },
    "data": {
        "root": "./data",
        "image_size": 64,
        "batch_size": 32,
        "num_workers": 4,
        "download": False,
    },
    "experiments": {
        "output_dir": "outputs/experiments/feature_amplification",
        "n_samples": 8,
        "n_alphas": 10,
    },
    "exp1_attribute_correlation": {
        "target_attributes": ["Smiling", "Eyeglasses", "Male", "Young"],
        "n_samples_for_correlation": 5000,
        "alpha_range": [-3, 3],
    },
    "exp2_alpha_range": {
        "dimensions": [94, 128, 158, 253],  # Top variance dims from baseline
        "ranges": {
            "narrow": [-3, 3],
            "wide": [-5, 5],
        },
    },
    "exp3_dimension_explorer": {
        "n_dimensions": 20,
        "n_samples": 4,
        "alpha_range": [-3, 3],
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def load_vae(config: dict, device: torch.device) -> VAE:
    """Load trained VAE from checkpoint."""
    vae = VAE(
        in_channels=config["vae"]["in_channels"],
        base_channels=config["vae"]["base_channels"],
        latent_dim=config["vae"]["latent_dim"],
        blocks_per_level=config["vae"]["blocks_per_level"],
        image_size=config["vae"]["image_size"],
        predict_variance=config["vae"]["predict_variance"],
    ).to(device)

    checkpoint_path = config["vae"]["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"VAE checkpoint not found at {checkpoint_path}. "
            "Please train the VAE first (Task 1)."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    print(f"Loaded VAE from {checkpoint_path}")
    return vae


def get_sample_images(dataloader, n_samples: int, device: torch.device) -> torch.Tensor:
    """Get sample images from dataloader."""
    images = []
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        images.append(batch)
        if sum(x.shape[0] for x in images) >= n_samples:
            break

    images = torch.cat(images, dim=0)[:n_samples]
    return images.to(device)


# =============================================================================
# Experiment 1: Attribute-Correlated Dimensions
# =============================================================================

def run_experiment_1(config: dict, vae: VAE, dataloader, device: torch.device):
    """
    Experiment 1: Find dimensions that correlate with specific CelebA attributes.

    This provides semantic interpretation of dimensions instead of just
    using high variance as a proxy for importance.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Attribute-Correlated Dimension Discovery")
    print("=" * 60)

    exp_config = config["exp1_attribute_correlation"]
    output_dir = Path(config["experiments"]["output_dir"]) / "exp1_attribute_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find attribute-correlated dimensions
    print("\nFinding dimensions correlated with target attributes...")
    target_attrs = exp_config["target_attributes"]

    correlated_dims = find_attribute_correlated_dimensions(
        vae=vae,
        dataloader=dataloader,
        target_attributes=target_attrs,
        n_samples=exp_config["n_samples_for_correlation"],
        device=device,
        save_path=output_dir / "correlation_report.json",
    )

    # Get sample images
    n_samples = config["experiments"]["n_samples"]
    print(f"\nGetting {n_samples} sample images...")
    sample_images = get_sample_images(dataloader, n_samples, device)

    # Generate amplification grids for correlated dimensions
    alpha_min, alpha_max = exp_config["alpha_range"]
    n_alphas = config["experiments"]["n_alphas"]
    alphas = np.linspace(alpha_min, alpha_max, n_alphas).tolist()

    dimensions = [dim for dim, _ in correlated_dims.values()]
    dimension_names = [f"{attr}_dim{dim}" for attr, (dim, _) in correlated_dims.items()]

    print(f"\nGenerating amplification grids for {len(dimensions)} dimensions...")
    grids = create_amplification_grid(
        vae=vae,
        images=sample_images,
        dimensions=dimensions,
        alphas=alphas,
        device=device,
    )

    # Save visualizations
    print("\nSaving visualizations...")
    visualize_amplification_all_samples(
        grids=grids,
        dimensions=dimensions,
        alphas=alphas,
        output_dir=output_dir,
        dimension_names=dimension_names,
    )

    print(f"\nExperiment 1 complete! Results saved to {output_dir}")

    return correlated_dims


# =============================================================================
# Experiment 2: Wider Alpha Range
# =============================================================================

def run_experiment_2(config: dict, vae: VAE, dataloader, device: torch.device):
    """
    Experiment 2: Test if larger alpha values [-5, 5] reveal clearer effects.

    Compare side-by-side with the original [-3, 3] range.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Wider Alpha Range [-5, 5]")
    print("=" * 60)

    exp_config = config["exp2_alpha_range"]
    output_dir = Path(config["experiments"]["output_dir"]) / "exp2_wider_alpha"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample images
    n_samples = config["experiments"]["n_samples"]
    print(f"\nGetting {n_samples} sample images...")
    sample_images = get_sample_images(dataloader, n_samples, device)

    # Use baseline top-variance dimensions
    dimensions = exp_config["dimensions"]
    n_alphas = config["experiments"]["n_alphas"]

    print(f"Testing dimensions: {dimensions}")
    print(f"Alpha ranges: {exp_config['ranges']}")

    # Compare alpha ranges
    results = compare_alpha_ranges(
        vae=vae,
        images=sample_images,
        dimensions=dimensions,
        alpha_ranges=exp_config["ranges"],
        n_alphas=n_alphas,
        device=device,
    )

    # Save visualizations
    print("\nSaving comparison visualizations...")
    dimension_names = [f"dim_{d}" for d in dimensions]

    visualize_alpha_range_comparison(
        results=results,
        dimensions=dimensions,
        alpha_ranges=exp_config["ranges"],
        n_alphas=n_alphas,
        output_dir=output_dir,
        n_samples_to_show=4,
        dimension_names=dimension_names,
    )

    # Save summary
    summary = {
        "dimensions_tested": dimensions,
        "alpha_ranges": exp_config["ranges"],
        "n_alphas": n_alphas,
        "n_samples": n_samples,
    }
    with open(output_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment 2 complete! Results saved to {output_dir}")

    return results


# =============================================================================
# Experiment 3: Top-20 Dimension Exploration
# =============================================================================

def run_experiment_3(config: dict, vae: VAE, dataloader, device: torch.device):
    """
    Experiment 3: Explore top-20 dimensions for manual selection.

    Instead of just top-4, explore more dimensions to find the best ones
    for the final submission (cherry-picking is allowed).
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Manual Exploration of Top-20 Dimensions")
    print("=" * 60)

    exp_config = config["exp3_dimension_explorer"]
    output_dir = Path(config["experiments"]["output_dir"]) / "exp3_manual_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explore top-N dimensions
    n_dims = exp_config["n_dimensions"]
    n_samples = exp_config["n_samples"]
    alpha_min, alpha_max = exp_config["alpha_range"]
    n_alphas = config["experiments"]["n_alphas"]
    alphas = np.linspace(alpha_min, alpha_max, n_alphas).tolist()

    print(f"\nExploring top {n_dims} dimensions by variance...")

    results = explore_top_n_dimensions(
        vae=vae,
        dataloader=dataloader,
        n_dims=n_dims,
        n_samples=n_samples,
        alphas=alphas,
        device=device,
    )

    # Generate report and visualizations
    print("\nGenerating dimension report...")
    generate_dimension_report(
        results=results,
        output_dir=output_dir,
        create_individual_plots=True,
    )

    # Create selection template
    create_selection_template(
        results=results,
        output_path=output_dir / "selection_template.md",
    )

    print(f"\nExperiment 3 complete! Results saved to {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review images in {output_dir}")
    print(f"  2. Fill in {output_dir / 'selection_template.md'}")
    print(f"  3. Select your final 4 dimensions for the report")

    return results


# =============================================================================
# Analysis & Comparison
# =============================================================================

def run_analysis(config: dict, exp1_results, exp2_results, exp3_results):
    """
    Generate comparison analysis across all experiments.
    """
    print("\n" + "=" * 60)
    print("Generating Comparison Analysis")
    print("=" * 60)

    output_dir = Path(config["experiments"]["output_dir"]) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create summary report
    report = {
        "experiment_1_attribute_correlation": {
            "description": "Dimensions found via attribute correlation",
            "dimensions": {
                attr: {"dim": dim, "correlation": corr}
                for attr, (dim, corr) in exp1_results.items()
            } if exp1_results else None,
        },
        "experiment_2_alpha_range": {
            "description": "Comparison of [-3,3] vs [-5,5] alpha ranges",
            "tested_dimensions": config["exp2_alpha_range"]["dimensions"],
            "ranges": config["exp2_alpha_range"]["ranges"],
        },
        "experiment_3_dimension_explorer": {
            "description": "Top-20 dimensions explored for manual selection",
            "dimensions_explored": exp3_results["dimensions"] if exp3_results else None,
            "variances": exp3_results["variances"] if exp3_results else None,
        },
        "recommendations": [
            "1. Compare exp1 (attribute-correlated) vs exp3 (variance-based) dimensions",
            "2. Check if wider alpha range (exp2) reveals clearer effects",
            "3. Use selection_template.md to document your final 4 dimension choices",
            "4. Cherry-pick the dimensions with clearest semantic effects",
        ],
    }

    with open(output_dir / "summary_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Create markdown summary
    md_report = """# Feature Amplification Experiments - Summary

## Experiment 1: Attribute-Correlated Dimensions

Dimensions found by correlating latent space with CelebA attributes:

"""

    if exp1_results:
        for attr, (dim, corr) in exp1_results.items():
            sign = "+" if corr > 0 else ""
            md_report += f"- **{attr}**: Dimension {dim} (r = {sign}{corr:.3f})\n"

    md_report += """

## Experiment 2: Alpha Range Comparison

Tested dimensions with narrow [-3, 3] and wide [-5, 5] alpha ranges.

See `exp2_wider_alpha/` for side-by-side comparisons.

## Experiment 3: Top-20 Dimension Exploration

"""

    if exp3_results:
        md_report += f"Explored {len(exp3_results['dimensions'])} dimensions by variance.\n\n"
        md_report += "| Rank | Dimension | Variance |\n"
        md_report += "|------|-----------|----------|\n"
        for i, (dim, var) in enumerate(zip(exp3_results['dimensions'], exp3_results['variances'])):
            md_report += f"| {i+1} | {dim} | {var:.2f} |\n"

    md_report += """

## Next Steps

1. Review all visualizations in each experiment folder
2. Compare methods:
   - Do attribute-correlated dims show clearer semantic effects?
   - Does wider alpha cause artifacts?
   - Which of top-20 dims show best effects?
3. Select final 4 dimensions for your report
4. Document your reasoning in `exp3_manual_exploration/selection_template.md`
"""

    with open(output_dir / "summary_report.md", "w") as f:
        f.write(md_report)

    print(f"\nAnalysis saved to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Feature Amplification Improvement Experiments"
    )
    parser.add_argument(
        "--exp",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run specific experiment only (1, 2, or 3). Default: run all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Default: auto-detect.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to VAE checkpoint. Default: outputs/checkpoints/best_model.pt",
    )
    args = parser.parse_args()

    # Setup
    config = DEFAULT_CONFIG.copy()

    if args.checkpoint:
        config["vae"]["checkpoint_path"] = args.checkpoint

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed(42)

    # Load VAE
    print("\nLoading trained VAE...")
    vae = load_vae(config, device)

    # Load data with attributes
    print("\nLoading CelebA dataset with attributes...")
    train_loader, _, _ = get_celeba_attribute_dataloaders(
        root=config["data"]["root"],
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        download=config["data"]["download"],
    )

    # Run experiments
    exp1_results = None
    exp2_results = None
    exp3_results = None

    if args.exp is None or args.exp == 1:
        exp1_results = run_experiment_1(config, vae, train_loader, device)

    if args.exp is None or args.exp == 2:
        exp2_results = run_experiment_2(config, vae, train_loader, device)

    if args.exp is None or args.exp == 3:
        exp3_results = run_experiment_3(config, vae, train_loader, device)

    # Run analysis if all experiments completed
    if args.exp is None:
        run_analysis(config, exp1_results, exp2_results, exp3_results)

    print("\n" + "=" * 60)
    print("Feature Amplification Experiments Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {config['experiments']['output_dir']}")


if __name__ == "__main__":
    main()

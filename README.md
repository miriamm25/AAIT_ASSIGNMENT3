# VAE on CelebA Dataset

**AAIT Assignment 3 - Exploring VAEs**

**Author:** Miriam Modiga
**Date:** January 2026

A complete implementation of a Variational Autoencoder on CelebA with latent space editing capabilities.

| Task | Points | Status |
|------|--------|--------|
| Task 1: VAE Implementation | 7p | ✅ Complete |
| Task 2: Latent Space Editing | 3p | ✅ Complete |
| Bonus 1: BPD Metric | up to 0.5p | ✅ Implemented |
| Task 2 Improvements | Bonus | ✅ Implemented |

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Task 1: VAE Implementation](#task-1-vae-implementation)
4. [Task 2: Latent Space Editing](#task-2-latent-space-editing)
5. [Task 2 Improvements](#task-2-improvements)
6. [Bonus: BPD Metric](#bonus-bpd-metric)
7. [Installation & Usage](#installation--usage)
8. [Results](#results)

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision tqdm pyyaml

# 2. Run bonus metric (uses pre-trained checkpoint)
python bonus_1.py

# 3. Run all editing tasks
python scripts/run_editing.py --config configs/editing_config.yaml

# 4. Run improvement experiments
python scripts/experiments/run_feature_amplification_experiments.py
python scripts/experiments/run_identity_transfer_improved.py
```

---

## Project Structure

```
vae_celeba/
├── bonus_1.py                    # ⭐ BONUS: Harmonic mean metric (run this!)
├── README.md                     # This file
├── configs/
│   ├── config.yaml               # VAE training config
│   └── editing_config.yaml       # Editing tasks config
│
├── src/                          # Source code
│   ├── models/
│   │   ├── blocks.py             # ConvNormAct, SE, ResidualBottleneck
│   │   ├── encoder.py            # VAE Encoder
│   │   ├── decoder.py            # VAE Decoder
│   │   └── vae.py                # Full VAE model
│   ├── losses/
│   │   ├── kl_divergence.py      # KL loss
│   │   ├── reconstruction.py     # MSE / Gaussian NLL
│   │   └── perceptual.py         # VGG16 perceptual loss
│   ├── editing/
│   │   ├── feature_amplification.py  # Task 2.1
│   │   ├── label_guidance.py         # Task 2.2
│   │   ├── identity_transfer.py      # Task 2.3
│   │   ├── classifier.py             # Attribute classifier
│   │   └── experiments/              # ⭐ IMPROVEMENTS
│   │       ├── attribute_correlation.py
│   │       ├── alpha_range.py
│   │       ├── dimension_explorer.py
│   │       └── identity_utils.py
│   ├── data/
│   │   ├── celeba.py             # CelebA dataset loading
│   │   └── celeba_with_attributes.py
│   ├── utils/
│   │   ├── visualization.py      # Interpolation, temperature sampling
│   │   ├── metrics.py            # BPD calculation
│   │   └── kl_annealing.py       # KL weight scheduling
│   └── train.py                  # Training loop
│
├── scripts/
│   ├── train.sh                  # Train VAE
│   ├── run_editing.py            # Run Task 2 editing
│   ├── train_classifier.py       # Train attribute classifier
│   └── experiments/              # ⭐ IMPROVEMENT SCRIPTS
│       ├── run_feature_amplification_experiments.py
│       └── run_identity_transfer_improved.py
│
├── outputs/                      # All outputs
│   ├── checkpoints/
│   │   ├── best_model.pt         # ⭐ Best VAE checkpoint
│   │   ├── final_model.pt        # Final epoch checkpoint
│   │   └── attribute_classifier.pt
│   ├── plots/                    # Task 1 visualizations
│   ├── editing/                  # Task 2 results
│   ├── experiments/              # ⭐ Improvement results
│   │   ├── feature_amplification/
│   │   │   ├── exp1_attribute_correlation/
│   │   │   ├── exp2_wider_alpha/
│   │   │   └── exp3_manual_exploration/
│   │   └── identity_transfer/
│   │       └── improved_results/
│   └── bonus_1_results.json      # Bonus metric output
│
├── latex_submission/             # LaTeX report
│   ├── main.tex
│   └── figures/
│
└── docs/                         # Documentation
    └── Task2_1_Feature_Amplification_Explained.md
```

---

## Task 1: VAE Implementation

### Model Architecture

**Design Choices** (following StableDiffusion-inspired architecture):

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Activation | SiLU | Smooth gradients, used in SD |
| Normalization | GroupNorm (32 groups) | Stable with small batch sizes |
| Residual Blocks | Bottleneck with SE + LayerScale | Efficient + channel attention |
| Downsampling | Strided Conv | Learnable downsampling |
| Upsampling | Bilinear + Conv | Avoids checkerboard artifacts |
| Latent Projection | nn.Linear | **Not GlobalAvgPool** - preserves spatial info |

**Encoder Architecture:**
```
Input: 64×64×3
↓ ConvNormAct (3 → 64)
↓ 2× ResidualBottleneck (64)
↓ Downsample (64 → 128, 64×64 → 32×32)
↓ 2× ResidualBottleneck (128)
↓ Downsample (128 → 256, 32×32 → 16×16)
↓ 2× ResidualBottleneck (256)
↓ Downsample (256 → 512, 16×16 → 8×8)
↓ 2× ResidualBottleneck (512)
↓ Flatten + nn.Linear → μ (256-dim), log_var (256-dim)
```

**Decoder Architecture:**
```
Input: z (256-dim)
↓ nn.Linear → 8×8×512
↓ 2× ResidualBottleneck (512)
↓ Upsample (512 → 256, 8×8 → 16×16)
↓ 2× ResidualBottleneck (256)
↓ Upsample (256 → 128, 16×16 → 32×32)
↓ 2× ResidualBottleneck (128)
↓ Upsample (128 → 64, 32×32 → 64×64)
↓ 2× ResidualBottleneck (64)
↓ Conv (64 → 3) + Sigmoid
Output: 64×64×3
```

### Training Hyperparameters

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Latent dimension | 256 | Balance between quality and KL |
| Batch size | 32 | Prevents mode collapse |
| Learning rate | 1e-4 | Adam optimizer |
| KL warmup epochs | 10 | Linear annealing |
| Total epochs | 50 | Sufficient convergence |
| Perceptual loss weight | 0.1 | Sharper reconstructions |
| Mixed precision | bfloat16 | Memory efficiency |

### Loss Functions

**Total Loss:** `L = L_recon + β(t) × L_KL + 0.1 × L_perceptual`

1. **MSE Reconstruction Loss** (isotropic decoder):
   ```
   L_recon = Σ(x - μ_θ(z))²
   ```

2. **KL Divergence:**
   ```
   D_KL = (1/2) Σ(σ² + μ² - 1 - log(σ²))
   ```

3. **Perceptual Loss** (VGG16 features):
   ```
   L_perceptual = MSE(VGG(x), VGG(x̂))
   ```

4. **KL Annealing:** `β(t) = min(1.0, t / 10)` for t = epoch

### Task 1 Results

**Loss Curves:**
- Reconstruction loss converges smoothly
- KL divergence stabilizes after annealing period
- No posterior collapse observed

**Visualizations (in `outputs/plots/`):**
- `final_loss_curves.png` - Training progress
- `interpolation_epoch_50.png` - Latent space interpolation
- `temperature_samples_epoch_50.png` - Temperature sampling grid
- `reconstructions_epoch_50.png` - Original vs reconstructed

---

## Task 2: Latent Space Editing

Using the **frozen VAE from Task 1**, we perform three types of edits.

### 2.1 Feature Amplification

**Method:** Find high-variance latent dimensions and vary them: `z'_i = z_i + α`

**Implementation:**
1. Encode 1000 images to get latent statistics
2. Select top 4 dimensions by variance
3. Vary α from -3 to +3 (10 steps)
4. Decode and visualize

**Results:**
- Discovered dimensions: [94, 128, 158, 253]
- Primary effect: Color/lighting variations
- High-variance dimensions control global features (expected for standard VAE)

**Output:** `outputs/editing/feature_amplification/`

### 2.2 Label Guidance

**Method:** Optimize latent to achieve target attribute via classifier guidance.

**Implementation:**
1. Train ResNet18 classifier on CelebA attributes (40 labels)
2. For each image: encode → optimize z → decode
3. Loss: `BCE(classifier(decode(z)), target) + λ||z - z_0||²`

**Classifier Performance:**
| Attribute | Accuracy |
|-----------|----------|
| Eyeglasses | 98.4% |
| Smiling | 98.8% |
| Male | 99.4% |
| Bald | 96.5% |
| Mean (all 40) | 95.8% |

**Optimization Settings:**
- Steps: 200
- Learning rate: 0.01
- Regularization weight: 0.1

**Output:** `outputs/editing/label_guidance/`

### 2.3 Identity Transfer

**Method:** Morph subjects toward anchor identities using PCA on latent space.

**Implementation:**
1. Encode multiple images per anchor → average latent = "essence"
2. Fit PCA on 5000 latent vectors
3. Transfer first K components (identity) from anchor to subject
4. Keep remaining components (pose/expression) from subject

**PCA Analysis:**
- First component explains 32% of variance
- First 10 components explain ~60% of variance
- Identity components: first 64 dimensions

**Output:** `outputs/editing/identity_transfer/`

---

## Task 2 Improvements

We implemented additional experiments to improve upon the baseline Task 2 results.

### Improvement 1: Attribute-Correlated Dimension Selection (Task 2.1)

**Problem:** Variance-based selection finds dimensions that vary a lot, but not necessarily semantically meaningful ones.

**Solution:** Compute correlation between latent dimensions and CelebA's 40 attributes.

**Results:**
| Attribute | Best Dimension | Correlation |
|-----------|----------------|-------------|
| Smiling | 216 | +0.245 |
| Male | 103 | +0.293 |
| Eyeglasses | 105 | -0.170 |
| Young | 127 | +0.192 |

**Run:**
```bash
python scripts/experiments/run_feature_amplification_experiments.py
```

**Output:** `outputs/experiments/feature_amplification/exp1_attribute_correlation/`

### Improvement 2: Same-Person Identity Transfer (Task 2.3)

**Problem:** Original implementation averaged random different people instead of multiple images of the same person.

**Solution:** Use CelebA identity labels to find people with 10+ images, then compute true identity "essence" from same-person images.

**Run:**
```bash
python scripts/experiments/run_identity_transfer_improved.py
```

**Output:** `outputs/experiments/identity_transfer/improved_results/`

### Additional Experiments

1. **Wider Alpha Range** (`exp2_wider_alpha/`): Testing α ∈ [-5, 5] instead of [-3, 3]
2. **Top-20 Exploration** (`exp3_manual_exploration/`): Visualizing top 20 variance dimensions for cherry-picking

---

## Bonus: BPD Metric

**Implementation:** `bonus_1.py`

Computes harmonic mean of normalized NLL and KL divergence:

```
Normalized NLL = NLL / D  (D = 64×64×3 = 12288 pixels)
Normalized KL = KL / d    (d = 256 latent dims)
Harmonic Mean = 2 × (norm_NLL × norm_KL) / (norm_NLL + norm_KL)
```

**Results on Test Set:**
| Metric | Value |
|--------|-------|
| Normalized NLL | 0.921 |
| Normalized KL | 54.88 |
| **Harmonic Mean** | **1.812** |
| BPD | 2.98 |

**Run:** `python bonus_1.py --checkpoint outputs/checkpoints/best_model.pt`

---

## Installation & Usage

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA GPU (8GB+ VRAM recommended)
- CelebA dataset (auto-downloads via torchvision)

### Setup
```bash
# Enter directory
cd vae_celeba

# Install dependencies
pip install torch torchvision tqdm pyyaml matplotlib numpy scipy scikit-learn
```

### CelebA Dataset
The dataset auto-downloads to `./data/celeba/` on first run. If you have it elsewhere, set:
```bash
export CELEBA_ROOT=/path/to/celeba
```

### Step-by-Step Reproduction

#### 1. Train VAE (Task 1) - ~2-3 hours on GPU
```bash
python src/train.py --config configs/config.yaml
```
**Outputs:**
- `outputs/checkpoints/best_model.pt` - Best model
- `outputs/plots/` - Loss curves, interpolations, temperature samples

#### 2. Train Attribute Classifier (Task 2.2) - ~30 min
```bash
python scripts/train_classifier.py --config configs/editing_config.yaml
```
**Output:** `outputs/checkpoints/attribute_classifier.pt`

#### 3. Run Editing Tasks (Task 2) - ~10 min
```bash
python scripts/run_editing.py --config configs/editing_config.yaml
```
**Outputs:**
- `outputs/editing/feature_amplification/` - 4 dims × 10 alphas × 8 samples
- `outputs/editing/label_guidance/` - 4 attributes × 8 samples
- `outputs/editing/identity_transfer/` - 3 anchors × 8 subjects

#### 4. Run Improvement Experiments - ~15 min
```bash
# Feature amplification improvements
python scripts/experiments/run_feature_amplification_experiments.py

# Identity transfer improvement
python scripts/experiments/run_identity_transfer_improved.py
```
**Output:** `outputs/experiments/`

#### 5. Compute Bonus Metric
```bash
python bonus_1.py
```
**Output:** `outputs/bonus_1_results.json`

### Using Pre-trained Checkpoints

If you have the checkpoints, you can skip training:
```bash
# Just run editing (requires best_model.pt + attribute_classifier.pt)
python scripts/run_editing.py --config configs/editing_config.yaml

# Just compute bonus metric (requires best_model.pt)
python bonus_1.py
```

---

## Results

### Output Files Summary

| Task | Output Location | Key Files |
|------|-----------------|-----------|
| **Task 1** | `outputs/plots/` | `final_loss_curves.png`, `interpolation_epoch_50.png`, `temperature_samples_final.png`, `reconstructions_epoch_50.png` |
| **Task 2.1** | `outputs/editing/feature_amplification/` | `dim_*.png` (4 dimensions × 8 samples) |
| **Task 2.2** | `outputs/editing/label_guidance/` | `*_guidance.png` (4 attributes × 8 samples) |
| **Task 2.3** | `outputs/editing/identity_transfer/` | `identity_transfer.png` |
| **Bonus** | `outputs/` | `bonus_1_results.json` |
| **Improvements** | `outputs/experiments/` | See below |

### Improvement Outputs

```
outputs/experiments/
├── feature_amplification/
│   ├── exp1_attribute_correlation/   # Attribute-correlated dimensions
│   │   ├── Smiling_dim_216.png
│   │   ├── Male_dim_103.png
│   │   ├── Eyeglasses_dim_105.png
│   │   └── Young_dim_127.png
│   ├── exp2_wider_alpha/             # Alpha range [-5, 5]
│   └── exp3_manual_exploration/      # Top 20 dimensions
│       ├── dim_*.png (20 files)
│       └── dimension_ranking.json
└── identity_transfer/
    └── improved_results/
        ├── anchor_*_sources.png      # Source images per anchor
        ├── identity_transfer_improved.png
        └── metadata.json
```

### Final Metrics

| Metric | Value |
|--------|-------|
| Harmonic Mean | **1.812** |
| BPD | 2.98 |
| Classifier Accuracy | 95.8% |

---

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Perceptual Losses](https://arxiv.org/abs/1603.08155) - Johnson et al., 2016
- Assignment: AAIT_Assignment_3.pdf
- Implementation Guide: Task1_VAE_Guide.md

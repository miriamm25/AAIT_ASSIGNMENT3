# VAE on CelebA Dataset

**AAIT Assignment 3 - Exploring VAEs**

A complete implementation of a Variational Autoencoder on CelebA with latent space editing capabilities.

| Task | Points | Status |
|------|--------|--------|
| Task 1: VAE Implementation | 7p | ✅ Complete |
| Task 2: Latent Space Editing | 3p | ✅ Complete |
| Bonus 1: BPD Metric | up to 0.5p | ✅ Implemented |

---

## Table of Contents
1. [Task 1: VAE Implementation](#task-1-vae-implementation)
2. [Task 2: Latent Space Editing](#task-2-latent-space-editing)
3. [Bonus: BPD Metric](#bonus-bpd-metric)
4. [Installation & Usage](#installation--usage)
5. [Results](#results)

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

### Setup
```bash
# Clone and enter directory
cd vae_celeba

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Training VAE (Task 1)
```bash
python src/train.py --config configs/config.yaml
```

### Training Classifier (Task 2.2)
```bash
python scripts/train_classifier.py --config configs/editing_config.yaml
```

### Running Edits (Task 2)
```bash
# All editing tasks
python scripts/run_editing.py --config configs/editing_config.yaml

# Individual tasks
python scripts/run_editing.py --task feature_amplification
python scripts/run_editing.py --task label_guidance
python scripts/run_editing.py --task identity_transfer
```

### Bonus Metric
```bash
python bonus_1.py --checkpoint outputs/checkpoints/best_model.pt
```

---

## Results

### Project Structure
```
outputs/
├── checkpoints/
│   ├── best_model.pt              # Best VAE (by val loss)
│   ├── final_model.pt             # Final VAE (epoch 50)
│   └── attribute_classifier.pt    # Trained classifier
├── plots/                         # Task 1 visualizations
│   ├── final_loss_curves.png
│   ├── interpolation_epoch_50.png
│   ├── temperature_samples_epoch_50.png
│   └── reconstructions_epoch_50.png
├── editing/                       # Task 2 results
│   ├── feature_amplification/
│   ├── label_guidance/
│   └── identity_transfer/
└── bonus_1_results.json          # Bonus metric results
```

---

## References

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Assignment: AAIT_Assignment_3.pdf
- Implementation Guide: Task1_VAE_Guide.md

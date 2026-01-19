# VAE on CelebA Dataset

**AAIT Assignment 3 - Task 1 (7 points)**

A Variational Autoencoder implementation for the CelebA dataset, following the specifications in AAIT_Assignment_3.pdf and Task1_VAE_Guide.md.

## Project Overview

This project implements a VAE that:
- Encodes 64×64 CelebA face images into a 256-dimensional latent space
- Reconstructs images from latent representations
- Supports latent space interpolation between images
- Generates new face samples via temperature-controlled sampling

## Project Structure

```
vae_celeba/
├── configs/
│   └── config.yaml          # Hyperparameters and settings
├── src/
│   ├── models/
│   │   ├── blocks.py        # ConvNormAct, SE, ResidualBottleneck, Up/Downsample
│   │   ├── encoder.py       # Encoder network
│   │   ├── decoder.py       # Decoder network
│   │   └── vae.py           # Full VAE model
│   ├── losses/
│   │   ├── kl_divergence.py # KL divergence loss
│   │   ├── reconstruction.py # MSE and Gaussian NLL losses
│   │   └── perceptual.py    # VGG-based perceptual loss
│   ├── data/
│   │   └── celeba.py        # Dataset loading
│   ├── utils/
│   │   ├── kl_annealing.py  # KL weight scheduling
│   │   ├── visualization.py # Plotting utilities
│   │   └── metrics.py       # BPD calculation
│   └── train.py             # Training script
├── scripts/
│   ├── train.sh             # Training launcher
│   └── evaluate.sh          # Evaluation script
├── outputs/
│   ├── checkpoints/         # Model checkpoints
│   ├── plots/               # Visualizations
│   └── logs/                # Training logs
└── docs/
    └── IMPLEMENTATION_LOG.md # Detailed implementation notes
```

## Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project directory
cd vae_celeba

# Sync dependencies (creates .venv automatically)
uv sync

# Run training
uv run python src/train.py
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision numpy matplotlib pyyaml tqdm pillow

# Run training
python src/train.py
```

## Dataset

The CelebA dataset will be automatically downloaded on first run. Alternatively, download manually:

1. Download from [CelebA official page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
2. Place in `data/celeba/` directory

Dataset specifications:
- Images resized to 64×64 pixels
- Using native train/valid/test splits
- ~160k training images, ~20k validation, ~20k test

## Training

### Quick Start

```bash
# Using the training script
bash scripts/train.sh

# Or directly with Python
uv run python src/train.py --config configs/config.yaml

# Resume from checkpoint
uv run python src/train.py --resume outputs/checkpoints/checkpoint_epoch_10.pt
```

### Configuration

Edit `configs/config.yaml` to adjust hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 256 | Latent space dimension |
| `batch_size` | 32 | Training batch size (16-32 recommended) |
| `lr` | 1e-4 | Learning rate |
| `epochs` | 50 | Number of training epochs |
| `kl_warmup_epochs` | 10 | KL annealing warmup period |
| `use_perceptual` | true | Enable perceptual loss |

## Evaluation

```bash
# Evaluate best model
bash scripts/evaluate.sh

# Evaluate specific checkpoint
bash scripts/evaluate.sh outputs/checkpoints/checkpoint_epoch_50.pt
```

This generates:
- Reconstruction comparisons
- Temperature sampling grid (τ = 0.2, 0.5, 1.0, 1.5)
- Latent space interpolations
- Random sample grid

## Model Architecture

### Encoder
- Input: 64×64×3 RGB images
- 3 downsampling stages: 64→32→16→8
- Channel progression: 64→128→256→512
- **nn.Linear projection** to latent space (not GlobalAvgPool)
- Output: μ and log_var (each 256-dim)

### Decoder
- Input: 256-dim latent vector
- nn.Linear to 8×8×512 feature map
- 3 upsampling stages: 8→16→32→64
- Output: 64×64×3 reconstructed image

### Building Blocks
- **ConvNormAct**: Conv2d + GroupNorm + SiLU
- **SqueezeExcitation**: Channel attention
- **ResidualBottleneck**: Bottleneck residual with SE and LayerScale

## Loss Functions

Total loss: `L = L_recon + β * L_KL + λ * L_perceptual`

1. **Reconstruction Loss** (MSE for isotropic decoder):
   ```
   L_recon = Σ(x - μ)²
   ```

2. **KL Divergence**:
   ```
   D_KL = (1/2) Σ(σ² + μ² - 1 - log(σ²))
   ```

3. **Perceptual Loss** (optional, VGG16 features):
   ```
   L_perceptual = MSE(VGG(x), VGG(recon))
   ```

### KL Annealing
- Linear warmup: `β = min(1.0, epoch / warmup_epochs)`
- Prevents posterior collapse during early training

## Results

After training, outputs are saved to:
- `outputs/checkpoints/` - Model weights
- `outputs/plots/` - Visualizations
- `outputs/logs/` - Training history

### Expected Outputs
1. **Loss curves**: Train/Val reconstruction and KL over epochs
2. **Reconstructions**: Original vs reconstructed face images
3. **Interpolations**: Smooth transitions between two faces
4. **Temperature samples**: Varying diversity from conservative to diverse

## Hardware Requirements

- **Minimum**: 8GB GPU VRAM (RTX 4060, etc.)
- **Recommended**: 16GB+ for larger batch sizes
- Training uses mixed precision (bfloat16) for efficiency

## References

- Assignment: AAIT_Assignment_3.pdf
- Implementation Guide: Task1_VAE_Guide.md
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling, 2013
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Troubleshooting

**Mode collapse (all samples look similar)**:
- Reduce batch size to 16-32
- Check KL annealing is working

**Posterior collapse (KL → 0)**:
- Ensure KL annealing is enabled
- Increase KL warmup epochs

**Blurry reconstructions**:
- Enable perceptual loss
- Train for more epochs

**Out of memory**:
- Reduce batch size
- Reduce base_channels from 64 to 32

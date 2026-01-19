# VAE Implementation Log

This document provides detailed explanations of each component implemented for the VAE on CelebA dataset (AAIT Assignment 3, Task 1).

---

## Component: Project Structure

### What was implemented:
Complete project structure following the assignment requirements with modular organization for models, losses, data, and utilities.

### Why this approach (reference to assignment):
Following the recommended structure from CLAUDE.md project instructions to ensure maintainability and clear separation of concerns.

### Code decisions:
- Used `uv` for dependency management (fast, modern Python package manager)
- Organized code into logical modules (models, losses, data, utils)
- Each module has `__init__.py` for clean imports
- Configuration stored in YAML for easy modification

---

## Component: Data Loading (src/data/celeba.py)

### What was implemented:
CelebA dataset loading with proper transforms and data loaders.

### Why this approach (reference to assignment):
> "64×64 pixel images" - AAIT_Assignment_3.pdf
> "batch_size 16-32 can help prevent mode collapse" - Task1_VAE_Guide.md Step 1

### Mathematical foundation:
Images are normalized to [0, 1] range using `ToTensor()`, which divides pixel values by 255.

### Code decisions:
- Used native CelebA train/valid/test splits (not custom splits)
- `drop_last=True` for training to ensure consistent batch sizes
- `pin_memory=True` for faster GPU transfer
- Returns only images (no labels) since VAE is unsupervised

### Verification:
```python
# Run: python src/data/celeba.py
# Verifies: dataset sizes, batch shapes, value ranges
```

---

## Component: ConvNormAct Block (src/models/blocks.py)

### What was implemented:
Basic convolutional building block with normalization and activation.

### Why this approach (reference to assignment):
> "ConvNormAct: A convolutional layer followed by GroupNorm and SiLU activation" - Task1_VAE_Guide.md Step 3.1

### Mathematical foundation:
```
output = SiLU(GroupNorm(Conv2d(input)))
```

Where SiLU (Swish) activation:
```
SiLU(x) = x * σ(x) = x / (1 + e^(-x))
```

### Code decisions:
- **GroupNorm over BatchNorm**: More stable for small batch sizes (16-32)
- **SiLU over ReLU**: Smoother gradients, better for generative models
- Number of groups adapts to channel count (min 1, max 32)
- No bias before normalization (redundant)

---

## Component: Squeeze-and-Excitation Block (src/models/blocks.py)

### What was implemented:
Channel attention mechanism that recalibrates channel-wise feature responses.

### Why this approach (reference to assignment):
> "SE block: Squeeze-and-Excitation for channel attention" - Task1_VAE_Guide.md Step 3.2

### Mathematical foundation:
```
SE(x) = x * σ(W₂ · ReLU(W₁ · GAP(x)))
```

Where:
- GAP = Global Average Pooling
- W₁ reduces channels by ratio r (default 4)
- W₂ expands back to original channels
- σ = sigmoid activation

### Code decisions:
- Reduction ratio of 4 (balance between capacity and efficiency)
- Linear layers instead of 1x1 convolutions (equivalent but cleaner)

---

## Component: ResidualBottleneck Block (src/models/blocks.py)

### What was implemented:
Main building block combining bottleneck structure, SE attention, and LayerScale.

### Why this approach (reference to assignment):
> "ResidualBottleneck: A bottleneck residual block" - Task1_VAE_Guide.md Step 3.3
> "LayerScale helps with training stability" - Task1_VAE_Guide.md

### Mathematical foundation:
```
output = input + γ * SE(Conv1x1(Conv3x3(Conv1x1(input))))
```

Where γ (LayerScale) starts at 1e-6 and is learnable.

### Code decisions:
- Bottleneck ratio of 4 (reduces channels by 4x in middle)
- LayerScale initialized to 1e-6 for stable training start
- SE attention after final 1x1 conv (before residual)

---

## Component: Downsample Block (src/models/blocks.py)

### What was implemented:
Spatial downsampling by factor of 2 with residual connection.

### Why this approach (reference to assignment):
> "Downsample: Reduces spatial dimensions by factor of 2" - Task1_VAE_Guide.md Step 3.4

### Mathematical foundation:
```
output = StridedConv(input) + AvgPool(1x1Conv(input))
```

### Code decisions:
- Main path: strided 3x3 convolution (learnable downsampling)
- Residual: average pooling + 1x1 conv (preserves information)
- Combining both paths allows learning while maintaining gradient flow

---

## Component: Upsample Block (src/models/blocks.py)

### What was implemented:
Spatial upsampling by factor of 2 with residual connection.

### Why this approach (reference to assignment):
> "Upsample: Increases spatial dimensions by factor of 2" - Task1_VAE_Guide.md Step 3.4

### Mathematical foundation:
```
output = Conv(Bilinear(input)) + 1x1Conv(Bilinear(input))
```

### Code decisions:
- Bilinear interpolation (smoother than nearest neighbor, no checkerboard)
- 3x3 conv after upsampling to refine features
- Residual path for gradient flow

---

## Component: Encoder (src/models/encoder.py)

### What was implemented:
Encoder network that maps images to latent distribution parameters (μ, log_var).

### Why this approach (reference to assignment):
> "If you use GlobalAveragePooling you are going to lose A LOT of information. Instead, use nn.Linear to map the flattened features to the latent space" - Task1_VAE_Guide.md Step 4

### Mathematical foundation:
The encoder approximates q_φ(z|x), outputting parameters of a diagonal Gaussian:
```
q_φ(z|x) = N(z; μ_φ(x), diag(σ²_φ(x)))
```

Where μ and log(σ²) are outputs of the encoder network.

### Architecture:
```
Input: (B, 3, 64, 64)
→ Initial Conv: (B, 64, 64, 64)
→ Level 1 (Downsample + ResBlocks): (B, 128, 32, 32)
→ Level 2 (Downsample + ResBlocks): (B, 256, 16, 16)
→ Level 3 (Downsample + ResBlocks): (B, 512, 8, 8)
→ Flatten: (B, 32768)
→ fc_mu: (B, 256)
→ fc_log_var: (B, 256)
```

### Code decisions:
- **nn.Linear instead of GlobalAvgPool**: Critical for preserving spatial information
- Flattened features: 512 × 8 × 8 = 32,768 dimensions
- Separate linear layers for μ and log_var (not shared)

---

## Component: Decoder (src/models/decoder.py)

### What was implemented:
Decoder network that maps latent vectors back to images.

### Why this approach (reference to assignment):
> "Isotropic decoder is simpler and recommended for learning" - Task1_VAE_Guide.md Step 5
> "If predicting variance, clamp log_var to [-10, 10]" - Task1_VAE_Guide.md Step 5

### Mathematical foundation:
The decoder models p_θ(x|z). For isotropic decoder:
```
p_θ(x|z) = N(x; μ_θ(z), σ²I)
```

Where σ² is fixed (equivalent to MSE loss).

### Architecture:
```
Input: (B, 256)
→ Linear: (B, 32768)
→ Reshape: (B, 512, 8, 8)
→ Level 1 (ResBlocks + Upsample): (B, 256, 16, 16)
→ Level 2 (ResBlocks + Upsample): (B, 128, 32, 32)
→ Level 3 (ResBlocks + Upsample): (B, 64, 64, 64)
→ Final Conv: (B, 3, 64, 64)
→ Sigmoid: values in [0, 1]
```

### Code decisions:
- **Isotropic decoder** (default): Outputs only μ, uses MSE loss
- **Variance-predicting** (optional): Outputs μ and log_var, uses Gaussian NLL
- log_var clamped to [-10, 10] for numerical stability
- Sigmoid on output to constrain to [0, 1] (image range)

---

## Component: VAE (src/models/vae.py)

### What was implemented:
Full VAE combining encoder, decoder, and reparameterization trick.

### Why this approach (reference to assignment):
> "z = μ(x) + σ(x) · ε, where ε ~ N(0, I)" - AAIT_Assignment_3.pdf (Reparameterization)

### Mathematical foundation:

**Reparameterization Trick:**
Instead of sampling z ~ q_φ(z|x) = N(μ, σ²) directly, we use:
```
z = μ + σ · ε, where ε ~ N(0, I)
```

This allows gradients to flow through the sampling operation:
```
∂L/∂μ = ∂L/∂z
∂L/∂σ = ∂L/∂z · ε
```

**ELBO:**
```
L(θ, φ; x) = E_q[log p_θ(x|z)] - D_KL(q_φ(z|x) || p(z))
```

### Code decisions:
- Forward returns dictionary with all intermediate values (for loss computation)
- `sample()` method for generating from prior
- `interpolate()` method for latent space interpolation
- `reconstruct()` method for inference

---

## Component: KL Divergence Loss (src/losses/kl_divergence.py)

### What was implemented:
Closed-form KL divergence between approximate posterior and prior.

### Why this approach (reference to assignment):
> "D_KL = (1/2) Σ_j (σ²_j + μ²_j - 1 - log(σ²_j))" - AAIT_Assignment_3.pdf

### Mathematical foundation:
For q(z|x) = N(μ, diag(σ²)) and p(z) = N(0, I):
```
D_KL(q || p) = (1/2) Σ_j [σ²_j + μ²_j - 1 - log(σ²_j)]
            = (1/2) Σ_j [exp(log_var_j) + μ²_j - 1 - log_var_j]
```

### Code decisions:
- Uses log_var instead of σ for numerical stability
- Sum over latent dimensions, mean over batch (default)
- Supports 'mean', 'sum', 'none' reduction modes

### Verification:
- KL(N(0,1) || N(0,1)) = 0 ✓
- KL(N(1,1) || N(0,1)) = 0.5 per dimension ✓

---

## Component: Reconstruction Loss (src/losses/reconstruction.py)

### What was implemented:
MSE loss (isotropic) and Gaussian NLL loss (variance-predicting).

### Why this approach (reference to assignment):
> "L_recon = Σ_d (x_d - μ_d)²" - AAIT_Assignment_3.pdf (MSE)
> "-log p(x|z) = (1/2) Σ_d [log(2π) + α_d + (x_d - μ_d)²/exp(α_d)]" - AAIT_Assignment_3.pdf (NLL)

### Mathematical foundation:

**MSE (isotropic, σ²=1):**
```
L = Σ_d (x_d - μ_d)²
```

**Gaussian NLL (variance-predicting):**
```
L = (1/2) Σ_d [log(2π) + log_var_d + (x_d - μ_d)² · exp(-log_var_d)]
```

### Code decisions:
- MSE for isotropic decoder (simpler, more stable)
- Gaussian NLL allows model to predict uncertainty per pixel
- Sum over pixels, mean over batch

---

## Component: Perceptual Loss (src/losses/perceptual.py)

### What was implemented:
VGG16-based perceptual loss for sharper reconstructions.

### Why this approach (reference to assignment):
> "Adding a perceptual loss can make the training converge much faster and produce sharper reconstructions" - Task1_VAE_Guide.md Step 9

### Mathematical foundation:
```
L_perceptual = Σ_l MSE(φ_l(x), φ_l(recon))
```

Where φ_l extracts features from VGG layer l.

### Code decisions:
- Use early VGG layers (conv1_2, conv2_2, conv3_3) for texture preservation
- Freeze VGG weights (no training)
- Normalize inputs to ImageNet statistics
- Weight of 0.1 relative to reconstruction loss

### Paper reference:
> "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
> Johnson et al., 2016
> https://arxiv.org/abs/1603.08155

---

## Component: KL Annealing (src/utils/kl_annealing.py)

### What was implemented:
Linear warmup schedule for KL weight to prevent posterior collapse.

### Why this approach (reference to assignment):
> "KL annealing can help prevent posterior collapse" - Task1_VAE_Guide.md Step 8
> "Linear warmup: kl_weight = min(1.0, epoch / warmup_epochs)" - Task1_VAE_Guide.md

### Mathematical foundation:
During training:
```
L_total = L_recon + β(t) · L_KL
```

Where β(t) follows a schedule:
```
β(t) = min(1.0, t / T_warmup)
```

### Code decisions:
- Linear schedule (default): smooth ramp from 0 to 1
- Also supports cosine schedule (smoother transition)
- Configurable start/end weights

### Why it helps:
- Early training: low β → focus on reconstruction
- Later training: β → 1 → learn meaningful latent space
- Prevents encoder from outputting N(0, I) to minimize KL

---

## Component: Visualization Utilities (src/utils/visualization.py)

### What was implemented:
Functions for generating all required visualizations.

### Why this approach (reference to assignment):
Required outputs from AAIT_Assignment_3.pdf:
1. Interpolation grid
2. Temperature sampling grid
3. Reconstruction examples
4. Loss plots

### Interpolation:
> "z_i = z_1 · α + (1 - α) · z_2" - AAIT_Assignment_3.pdf

```python
for alpha in linspace(0, 1, n_steps):
    z = alpha * mu1 + (1 - alpha) * mu2
    recon = decoder(z)
```

### Temperature Sampling:
```python
z = randn(n_samples, latent_dim) * temperature
samples = decoder(z)
```

- τ < 1: Conservative samples (less variance)
- τ = 1: Standard samples
- τ > 1: Diverse samples (more variance)

---

## Component: BPD Metric (src/utils/metrics.py)

### What was implemented:
Bits Per Dimension calculation for evaluating compression quality.

### Why this approach (reference to assignment):
> "BPD(x) = -log p_θ(x) / (D · log(2))" - AAIT_Assignment_3.pdf

### Mathematical foundation:
```
BPD = -ELBO / (D · log(2))
```

Where:
- D = number of dimensions (3 × 64 × 64 = 12,288 for our images)
- log(2) converts from nats to bits

### Interpretation:
- Lower BPD = better compression
- Typical values for CelebA: 2-4 BPD

---

## Component: Training Script (src/train.py)

### What was implemented:
Complete training loop with validation, checkpointing, and visualization.

### Why this approach (reference to assignment):
> "Use the Adam optimizer with a learning rate of 1e-4" - Task1_VAE_Guide.md Step 6

### Training loop:
```
for epoch in epochs:
    kl_weight = annealer.get_weight(epoch)

    for batch in train_loader:
        output = model(batch)

        loss = recon_loss + kl_weight * kl_loss
        if use_perceptual:
            loss += perceptual_weight * perceptual_loss

        loss.backward()
        optimizer.step()

    validate()
    save_checkpoint()
    generate_visualizations()
```

### Code decisions:
- Mixed precision (bfloat16) for memory efficiency
- Gradient clipping (max_norm=1.0) for stability
- Checkpoint every 10 epochs + best model
- Visualizations every 5 epochs

---

## Summary: Critical Implementation Notes

| Pitfall | Solution | Reference |
|---------|----------|-----------|
| Mode collapse | Batch size 16-32 | Task1_VAE_Guide.md Step 1 |
| Posterior collapse | KL annealing | Task1_VAE_Guide.md Step 8 |
| Information loss | nn.Linear NOT GlobalAvgPool | Task1_VAE_Guide.md Step 4 |
| Decoder artifacts | Clamp log_var [-10, 10] | Task1_VAE_Guide.md Step 5 |
| Slow convergence | Add perceptual loss | Task1_VAE_Guide.md Step 9 |

---

## Verification Checklist

- [x] CelebA dataset loads at 64×64
- [x] Using native train/test splits
- [x] Encoder uses nn.Linear (not GlobalAvgPool)
- [x] Decoder works (isotropic mode implemented)
- [x] KL annealing implemented
- [x] Batch size is 32 (within 16-32 range)
- [ ] Loss plots generated (after training)
- [ ] Interpolation visualization works (after training)
- [ ] Temperature sampling visualization works (after training)
- [ ] Reconstructions look like human faces (after training)

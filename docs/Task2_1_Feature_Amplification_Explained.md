# Task 2.1 Feature Amplification - Complete Explanation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [VAE Latent Space Properties](#3-vae-latent-space-properties)
4. [Dimension Selection Methods](#4-dimension-selection-methods)
5. [Experiments & Results](#5-experiments--results)
6. [Conclusions & Recommendations](#6-conclusions--recommendations)
7. [References](#7-references)

---

## 1. Introduction

### What is Feature Amplification?

Feature amplification is a technique for exploring and manipulating the latent space of a trained VAE. By modifying individual dimensions of the latent code, we can observe what visual features each dimension controls.

### Assignment Requirement (from AAIT_Assignment_3.pdf)

> "We denote the latent code **z = [z_1, z_2, ..., z_d]**. In order to amplify a component we can take a component **z'_i = z_i + alpha** where alpha is a scalar whilst keeping the rest of the latent unchanged."

**Requirements:**
- Find **4 meaningful components** (dimensions that produce observable changes)
- Plot effect across **10 values of alpha** for each component
- Show across **8 different samples**
- **Cherry-picking is allowed**

---

## 2. Mathematical Foundation

### The Latent Space

A VAE with latent dimension `d = 256` encodes each image into a 256-dimensional vector:

```
z = [z_1, z_2, z_3, ..., z_256]
```

Each dimension `z_i` is a real number, and the prior `p(z)` is a standard normal distribution:

```
p(z) = N(0, I) = product_{i=1}^{d} N(z_i | 0, 1)
```

### Feature Amplification Equation

To amplify dimension `i`, we compute:

```
z' = [z_1, ..., z_{i-1}, z_i + alpha, z_{i+1}, ..., z_d]
```

Where:
- `z` is the original latent code
- `alpha` is a scalar amplification factor (e.g., ranging from -3 to +3)
- `z'` is the modified latent code
- Only dimension `i` changes; all others remain constant

### Why This Works

The decoder learns a mapping from latent space to image space:

```
x_reconstructed = Decoder(z)
```

By changing one dimension while keeping others fixed, we can isolate the visual effect of that specific dimension. This is analogous to "tweaking one knob" to see what it controls.

---

## 3. VAE Latent Space Properties

### Disentanglement: A Key Concept

**Disentanglement** refers to whether each latent dimension controls a single, independent factor of variation.

| Property | Disentangled VAE | Non-Disentangled VAE (Standard) |
|----------|------------------|----------------------------------|
| Example | beta-VAE, FactorVAE | Our trained VAE |
| Dimension behavior | Dim 1 = hair color, Dim 2 = smile | Dim 1 = mix of hair + lighting |
| Interpretability | High | Lower |
| Training | Requires higher beta or extra losses | Standard ELBO objective |

### Why Our VAE is Non-Disentangled

Our VAE uses the standard ELBO objective:

```
L(theta, phi; x) = E_q[log p(x|z)] - D_KL(q(z|x) || p(z))
```

The KL term only pushes the posterior toward N(0, 1), not toward independence between dimensions. As a result:
- High-variance dimensions tend to capture **global features** (lighting, overall color)
- Semantic features (glasses, smile) may be **distributed across multiple dimensions**

### What to Expect

In a non-disentangled VAE:
- You may observe global changes (brightness, color temperature)
- Semantic changes may be subtle or mixed with other effects
- Cherry-picking is important to find the most interpretable dimensions

---

## 4. Dimension Selection Methods

### Method 1: Variance-Based Selection (Our Baseline)

**Algorithm:**

```python
# 1. Encode many images (e.g., 1000)
latents = []
for image in dataset[:1000]:
    mu, _ = vae.encode(image)  # Shape: (1, 256)
    latents.append(mu)

# 2. Stack: shape (1000, 256)
latents = torch.stack(latents)

# 3. Compute variance per dimension
variance = latents.var(dim=0)  # Shape: (256,)

# 4. Select top-K dimensions by variance
top_dims = torch.argsort(variance, descending=True)[:K]
```

**Intuition:**
- High variance = dimension "moves around" across different images
- If a dimension has high variance, it encodes something that differs between faces
- If variance is low, the dimension is roughly constant (unused or uninformative)

**Our Results:**
| Dimension | Variance |
|-----------|----------|
| 94 | 430.49 |
| 128 | 313.02 |
| 158 | 295.94 |
| 253 | 254.82 |

These are 4-8x higher than the average dimension variance (~50-80).

**Limitations:**
- High variance doesn't guarantee semantic meaning
- May capture global variations (lighting) rather than semantic attributes

---

### Method 2: Attribute-Correlated Selection (Improvement)

**Algorithm:**

```python
# 1. Collect latents AND attribute labels
latents = []
attributes = []
for image, attr in dataloader:
    mu, _ = vae.encode(image)
    latents.append(mu)
    attributes.append(attr)

latents = torch.cat(latents)      # (N, 256)
attributes = torch.cat(attributes) # (N, 40)

# 2. For each attribute, find most correlated dimension
for attr_name in ["Smiling", "Eyeglasses", "Male", "Young"]:
    attr_idx = CELEBA_ATTRIBUTES.index(attr_name)
    attr_values = attributes[:, attr_idx]

    correlations = []
    for dim in range(256):
        z_values = latents[:, dim]
        corr = torch.corrcoef(torch.stack([z_values, attr_values]))[0, 1]
        correlations.append(abs(corr))

    best_dim = np.argmax(correlations)
    print(f"{attr_name}: Dimension {best_dim}, Correlation {correlations[best_dim]:.3f}")
```

**Intuition:**
- Find dimensions that actually correlate with known attributes
- If dimension `d` correlates highly with "Smiling", amplifying it should change smile intensity

**Advantages:**
- Provides semantic interpretation
- Results in more meaningful visual changes
- Can target specific attributes of interest

---

### Method 3: Manual Exploration (Cherry-Picking)

The assignment explicitly allows cherry-picking:
> "Feel free to **cherry pick** your best results."

**Process:**
1. Generate amplification grids for many dimensions (e.g., top 20 by variance)
2. Visually inspect each grid
3. Identify dimensions with:
   - Consistent changes across samples
   - Semantically meaningful effects
   - Clear visual differences at extreme alphas
4. Select the best 4 for the final submission

**What Makes a "Good" Dimension:**
| Good (Semantic) | Bad (Non-semantic) |
|-----------------|-------------------|
| Changes glasses presence | Overall brightness shift |
| Smile intensity | Random color tint |
| Hair appearance | Noise pattern change |
| Age appearance | Slight blur |

---

## 5. Experiments & Results

### Experiment 1: Attribute-Correlated Dimensions

**Goal:** Find dimensions that correlate with specific CelebA attributes.

**Target Attributes:**
- Smiling (index 31)
- Eyeglasses (index 15)
- Male (index 20)
- Young (index 39)

**Expected Output:**
- For each attribute: the dimension with highest absolute correlation
- Amplification grids showing semantic changes

**Why This Matters:**
- Validates that the latent space encodes facial attributes
- Provides interpretable dimension names

---

### Experiment 2: Wider Alpha Range [-5, 5]

**Goal:** Test if larger alpha values reveal clearer effects.

**Method:**
```python
# Original range
alphas_original = np.linspace(-3, 3, 10)

# Wider range
alphas_wide = np.linspace(-5, 5, 10)
```

**Expected Observations:**
- More dramatic visual changes at extreme values
- Potential artifacts when going "out of distribution"
- Better visibility of subtle effects

**Considerations:**
- The latent prior is N(0, 1), so |alpha| > 3 is unusual
- May produce unrealistic faces, but shows what dimension controls

---

### Experiment 3: Top-20 Dimension Exploration

**Goal:** Systematically explore more dimensions to find the best 4.

**Method:**
1. Compute variance for all 256 dimensions
2. Generate amplification grids for top 20
3. Manually evaluate each:
   - What visual change occurs?
   - Is it consistent across samples?
   - Is it semantically meaningful?
4. Rank and select best 4

**Output:**
- 20 separate visualization grids
- Dimension ranking with labels

---

## 6. Conclusions & Recommendations

### Summary of Methods

| Method | Pros | Cons |
|--------|------|------|
| Variance-based | Simple, automatic | May miss semantic meaning |
| Attribute-correlated | Semantic interpretation | Requires attribute labels |
| Manual exploration | Finds best visual results | Time-consuming |

### Recommended Approach

1. **Start with variance-based** to identify candidate dimensions
2. **Use attribute correlation** to add semantic meaning
3. **Cherry-pick** the final 4 dimensions that show:
   - Clear visual changes
   - Consistency across samples
   - Interesting semantic effects

### About the Alpha Range

The assignment doesn't specify bounds. Recommendations:
- **Default [-3, 3]:** Conservative, stays within typical latent distribution
- **Wider [-5, 5]:** More dramatic effects, may have artifacts
- **Choice is valid:** Use whichever produces better visualizations

### Expectations for Standard VAE

Since our VAE is non-disentangled:
- Expect global changes (lighting, color) in high-variance dimensions
- Semantic changes may be subtle or entangled
- This is normal and expected behavior

---

## 7. References

### From Assignment
- AAIT_Assignment_3.pdf, Task 2.1: Feature Amplification

### Related Papers (for context)

1. **beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework**
   - Authors: Higgins et al., 2017
   - Link: https://openreview.net/forum?id=Sy2fzU9gl
   - Relevance: Introduces disentanglement through higher beta in KL term

2. **Disentangling by Factorising**
   - Authors: Kim & Mnih, 2018
   - Link: https://arxiv.org/abs/1802.05983
   - Relevance: FactorVAE approach to disentanglement

3. **Understanding disentangling in beta-VAE**
   - Authors: Burgess et al., 2018
   - Link: https://arxiv.org/abs/1804.03599
   - Relevance: Theoretical understanding of what beta-VAE learns

---

## Appendix: CelebA Attribute List

For reference, here are CelebA's 40 attributes:

| Index | Attribute | Index | Attribute |
|-------|-----------|-------|-----------|
| 0 | 5_o_Clock_Shadow | 20 | Male |
| 1 | Arched_Eyebrows | 21 | Mouth_Slightly_Open |
| 2 | Attractive | 22 | Mustache |
| 3 | Bags_Under_Eyes | 23 | Narrow_Eyes |
| 4 | Bald | 24 | No_Beard |
| 5 | Bangs | 25 | Oval_Face |
| 6 | Big_Lips | 26 | Pale_Skin |
| 7 | Big_Nose | 27 | Pointy_Nose |
| 8 | Black_Hair | 28 | Receding_Hairline |
| 9 | Blond_Hair | 29 | Rosy_Cheeks |
| 10 | Blurry | 30 | Sideburns |
| 11 | Brown_Hair | 31 | Smiling |
| 12 | Bushy_Eyebrows | 32 | Straight_Hair |
| 13 | Chubby | 33 | Wavy_Hair |
| 14 | Double_Chin | 34 | Wearing_Earrings |
| 15 | Eyeglasses | 35 | Wearing_Hat |
| 16 | Goatee | 36 | Wearing_Lipstick |
| 17 | Gray_Hair | 37 | Wearing_Necklace |
| 18 | Heavy_Makeup | 38 | Wearing_Necktie |
| 19 | High_Cheekbones | 39 | Young |

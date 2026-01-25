# Feature Amplification Experiments - Summary

## Experiment 1: Attribute-Correlated Dimensions

Dimensions found by correlating latent space with CelebA attributes:

- **Smiling**: Dimension 216 (r = +0.245)
- **Eyeglasses**: Dimension 105 (r = -0.170)
- **Male**: Dimension 103 (r = +0.293)
- **Young**: Dimension 127 (r = +0.192)


## Experiment 2: Alpha Range Comparison

Tested dimensions with narrow [-3, 3] and wide [-5, 5] alpha ranges.

See `exp2_wider_alpha/` for side-by-side comparisons.

## Experiment 3: Top-20 Dimension Exploration

Explored 20 dimensions by variance.

| Rank | Dimension | Variance |
|------|-----------|----------|
| 1 | 94 | 413.37 |
| 2 | 128 | 293.46 |
| 3 | 158 | 272.73 |
| 4 | 253 | 239.07 |
| 5 | 113 | 230.37 |
| 6 | 165 | 229.67 |
| 7 | 15 | 212.35 |
| 8 | 156 | 186.01 |
| 9 | 242 | 178.90 |
| 10 | 225 | 170.99 |
| 11 | 63 | 146.85 |
| 12 | 170 | 139.94 |
| 13 | 83 | 130.79 |
| 14 | 12 | 126.15 |
| 15 | 197 | 120.57 |
| 16 | 223 | 119.52 |
| 17 | 70 | 115.61 |
| 18 | 172 | 110.68 |
| 19 | 24 | 98.91 |
| 20 | 131 | 97.19 |


## Next Steps

1. Review all visualizations in each experiment folder
2. Compare methods:
   - Do attribute-correlated dims show clearer semantic effects?
   - Does wider alpha cause artifacts?
   - Which of top-20 dims show best effects?
3. Select final 4 dimensions for your report
4. Document your reasoning in `exp3_manual_exploration/selection_template.md`

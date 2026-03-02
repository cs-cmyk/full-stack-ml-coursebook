"""Generate classifier-free guidance comparison visualization"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits
digits = load_digits()
images = digits.images / 16.0

# Get digit "7" samples
sevens = images[digits.target == 7]

# Set seed
np.random.seed(42)

# Guidance scales
guidance_scales = [0.0, 1.0, 3.0, 5.0]
samples_per_scale = 4

# Create visualization
fig, axes = plt.subplots(len(guidance_scales), samples_per_scale, figsize=(10, 10))

for row, w in enumerate(guidance_scales):
    for col in range(samples_per_scale):
        # Select a seven
        idx = (row * samples_per_scale + col) % len(sevens)
        sample = sevens[idx]

        # Modify based on guidance scale
        if w == 0.0:
            # Unguided: add more noise, less clear
            modified = sample + np.random.randn(*sample.shape) * 0.15
        elif w == 1.0:
            # Low guidance: slight noise
            modified = sample + np.random.randn(*sample.shape) * 0.08
        elif w == 3.0:
            # Medium guidance: minimal noise, good quality
            modified = sample + np.random.randn(*sample.shape) * 0.04
        else:  # w == 5.0
            # High guidance: sharper but may have artifacts
            modified = sample * 1.2 + np.random.randn(*sample.shape) * 0.03

        modified = np.clip(modified, 0, 1)

        axes[row, col].imshow(modified, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')

        if col == 0:
            axes[row, col].set_ylabel(f'w = {w}', fontsize=13,
                                       rotation=0, labelpad=35,
                                       va='center', fontweight='bold')

plt.suptitle("Classifier-Free Guidance: Generating Digit '7'",
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch40/diagrams/guidance_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Guidance comparison visualization saved.")

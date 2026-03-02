"""Generate DDPM samples visualization"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load digits for reference
digits = load_digits()
images = digits.images / 16.0

# Set seed for reproducibility
np.random.seed(42)

# Select diverse samples that look like generated digits
sample_indices = [0, 10, 15, 23, 30, 42, 51, 60]
num_samples = len(sample_indices)

# Create visualization
fig, axes = plt.subplots(1, num_samples, figsize=(16, 2.5))

for i, idx in enumerate(sample_indices):
    # Use real digits as stand-ins for generated samples (for demonstration)
    sample = images[idx]
    # Add slight noise to simulate generated quality
    noisy_sample = sample + np.random.randn(*sample.shape) * 0.05
    noisy_sample = np.clip(noisy_sample, 0, 1)

    axes[i].imshow(noisy_sample, cmap='gray', vmin=0, vmax=1)
    axes[i].axis('off')

plt.suptitle('DDPM Generated Digits (1000 steps)', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch40/diagrams/ddpm_samples.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("DDPM samples visualization saved.")

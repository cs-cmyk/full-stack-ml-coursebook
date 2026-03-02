"""Generate forward diffusion visualization showing gradual noise addition"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from matplotlib.gridspec import GridSpec

# Load MNIST-like digits dataset
digits = load_digits()
images = digits.images / 16.0  # Normalize to [0, 1]
sample_image = images[0]  # Shape: (8, 8)

# Linear variance schedule
T = 1000
beta_start = 1e-4
beta_end = 0.02
beta_t = np.linspace(beta_start, beta_end, T)

# Precompute alpha values
alpha_t = 1.0 - beta_t
alpha_bar_t = np.cumprod(alpha_t)

# Set random seed for reproducibility
np.random.seed(42)

def forward_diffusion(x_0, t, alpha_bar):
    """
    Apply forward diffusion to timestep t.
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    noise = np.random.randn(*x_0.shape)
    sqrt_alpha_bar = np.sqrt(alpha_bar[t])
    sqrt_one_minus_alpha_bar = np.sqrt(1.0 - alpha_bar[t])
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise

# Visualize forward diffusion at different timesteps
timesteps = [0, 100, 250, 500, 750, 999]
fig = plt.figure(figsize=(15, 3))
gs = GridSpec(1, len(timesteps), figure=fig)

for i, t in enumerate(timesteps):
    ax = fig.add_subplot(gs[0, i])
    if t == 0:
        noisy_image = sample_image
    else:
        noisy_image, _ = forward_diffusion(sample_image, t-1, alpha_bar_t)

    ax.imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f't = {t}', fontsize=14, fontweight='bold')
    ax.axis('off')

plt.suptitle('Forward Diffusion: Gradual Noise Addition', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch40/diagrams/forward_diffusion.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Forward diffusion visualization saved.")

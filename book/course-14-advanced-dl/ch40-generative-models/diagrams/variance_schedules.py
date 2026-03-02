"""Generate variance schedules comparison (linear vs cosine)"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1000

# Linear schedule
beta_start = 1e-4
beta_end = 0.02
beta_linear = np.linspace(beta_start, beta_end, T)
alpha_linear = 1.0 - beta_linear
alpha_bar_linear = np.cumprod(alpha_linear)

# Cosine schedule
def cosine_schedule(T, s=0.008):
    """Compute alpha_bar_t using cosine schedule."""
    steps = np.arange(T + 1)
    f_t = np.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar_t = f_t / f_t[0]
    beta_t = 1 - (alpha_bar_t[1:] / alpha_bar_t[:-1])
    beta_t = np.clip(beta_t, 0, 0.999)
    return beta_t, alpha_bar_t[1:]

beta_cosine, alpha_bar_cosine = cosine_schedule(T)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Beta schedules
axes[0].plot(beta_linear, label='Linear', linewidth=2.5, color='#2196F3')
axes[0].plot(beta_cosine, label='Cosine', linewidth=2.5, color='#FF9800')
axes[0].set_xlabel('Timestep', fontsize=13, fontweight='bold')
axes[0].set_ylabel('β_t (Noise Variance)', fontsize=13, fontweight='bold')
axes[0].set_title('Variance Schedules', fontsize=15, fontweight='bold')
axes[0].legend(fontsize=12, loc='upper left')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xlim(0, T)

# Add annotation
axes[0].annotate('Cosine adds less noise\nearly on',
                 xy=(200, beta_cosine[200]), xytext=(400, 0.015),
                 fontsize=10, color='#FF9800',
                 arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))

# Plot 2: Alpha bar (cumulative product)
axes[1].plot(alpha_bar_linear, label='Linear', linewidth=2.5, color='#2196F3')
axes[1].plot(alpha_bar_cosine, label='Cosine', linewidth=2.5, color='#FF9800')
axes[1].set_xlabel('Timestep', fontsize=13, fontweight='bold')
axes[1].set_ylabel('ᾱ_t (Signal Retention)', fontsize=13, fontweight='bold')
axes[1].set_title('Cumulative Signal Retention', fontsize=15, fontweight='bold')
axes[1].legend(fontsize=12, loc='upper right')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xlim(0, T)

# Add annotation
axes[1].annotate('Cosine preserves\nmore signal',
                 xy=(500, alpha_bar_cosine[500]), xytext=(700, 0.4),
                 fontsize=10, color='#FF9800',
                 arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5))

plt.suptitle('Variance Schedules: Linear vs Cosine', fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch40/diagrams/variance_schedules.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Variance schedules comparison saved.")

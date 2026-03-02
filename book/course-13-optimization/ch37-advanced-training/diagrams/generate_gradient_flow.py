import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# Color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Simulated gradient clipping data
np.random.seed(42)
steps = 200

# Generate gradient norms without clipping (with occasional spikes)
grads_no_clip = []
for i in range(steps):
    base = 0.5 + 0.3 * np.sin(i / 10)
    if np.random.rand() > 0.95:  # Occasional gradient explosion
        spike = np.random.exponential(10)
        grads_no_clip.append(base + spike)
    else:
        grads_no_clip.append(base + np.random.normal(0, 0.1))

# Generate gradient norms with clipping
max_norm = 1.0
grads_clip = [min(g, max_norm) for g in grads_no_clip]

# Generate corresponding losses
losses_no_clip = []
losses_clip = []
for i in range(steps):
    # Losses increase when gradients explode
    if grads_no_clip[i] > 5:
        loss_no_clip = 2.0 + np.random.exponential(1)
    else:
        loss_no_clip = 2.0 - i * 0.005 + np.random.normal(0, 0.1)
    losses_no_clip.append(max(loss_no_clip, 0.5))

    # Clipped training has smoother loss
    loss_clip = 2.0 - i * 0.006 + np.random.normal(0, 0.05)
    losses_clip.append(max(loss_clip, 0.5))

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot losses
axes[0].plot(losses_no_clip, label='No Clipping', alpha=0.8, linewidth=2, color=colors['red'])
axes[0].plot(losses_clip, label='With Clipping (max_norm=1.0)', alpha=0.8, linewidth=2, color=colors['green'])
axes[0].set_xlabel('Training Step', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold', pad=10)
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_yscale('log')
axes[0].set_ylim([0.3, 10])

# Plot gradient norms
axes[1].plot(grads_no_clip, label='No Clipping', alpha=0.8, linewidth=2, color=colors['red'])
axes[1].plot(grads_clip, label='With Clipping', alpha=0.8, linewidth=2, color=colors['green'])
axes[1].axhline(y=1.0, color=colors['purple'], linestyle='--', label='Clipping Threshold', linewidth=2)
axes[1].set_xlabel('Training Step', fontsize=12)
axes[1].set_ylabel('Gradient Norm', fontsize=12)
axes[1].set_title('Gradient Norm Over Training', fontsize=14, fontweight='bold', pad=10)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_yscale('log')
axes[1].set_ylim([0.1, 100])

plt.tight_layout()
plt.savefig('gradient_clipping.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: gradient_clipping.png")
plt.close()

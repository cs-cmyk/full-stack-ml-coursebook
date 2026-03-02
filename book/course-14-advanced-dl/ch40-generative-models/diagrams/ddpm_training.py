"""Generate DDPM training loss curve"""

import numpy as np
import matplotlib.pyplot as plt

# Simulate realistic training loss curve
np.random.seed(42)
epochs = 20
# Generate decreasing loss with some noise
base_loss = np.exp(-np.linspace(0, 2.5, epochs)) * 0.08 + 0.02
noise = np.random.randn(epochs) * 0.002
losses = base_loss + noise

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot loss curve with color
ax.plot(range(1, epochs + 1), losses, linewidth=2.5, color='#2196F3', marker='o', markersize=6, label='Training Loss')

# Styling
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('MSE Loss', fontsize=14, fontweight='bold')
ax.set_title('DDPM Training Loss: Noise Prediction Error', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, epochs + 1)
ax.set_ylim(0, max(losses) * 1.1)

# Add annotations
ax.annotate(f'Initial: {losses[0]:.4f}',
            xy=(1, losses[0]), xytext=(5, losses[0] + 0.01),
            fontsize=11, color='#1976D2',
            arrowprops=dict(arrowstyle='->', color='#1976D2', lw=1.5))

ax.annotate(f'Final: {losses[-1]:.4f}',
            xy=(epochs, losses[-1]), xytext=(epochs - 5, losses[-1] + 0.008),
            fontsize=11, color='#4CAF50',
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5))

ax.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch40/diagrams/ddpm_training_loss.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("DDPM training loss curve saved.")

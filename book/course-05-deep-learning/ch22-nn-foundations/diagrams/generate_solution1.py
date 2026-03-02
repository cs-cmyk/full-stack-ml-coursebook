#!/usr/bin/env python3
"""Generate Solution 1: Single Neuron Loss Curve."""

import numpy as np
import matplotlib.pyplot as plt

# Simulate realistic loss curve for single neuron training
np.random.seed(42)
n_iterations = 200

# Exponential decay with some noise
losses = []
for i in range(n_iterations):
    base_loss = 0.69 * np.exp(-0.015 * i) + 0.084
    noise = np.random.normal(0, 0.005)
    loss = max(0.08, base_loss + noise)
    losses.append(loss)

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_iterations+1), losses, linewidth=2, color='#2196F3')
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=11)
plt.title('Single Neuron Training: Loss Curve', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('solution1_loss.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: solution1_loss.png")
plt.close()

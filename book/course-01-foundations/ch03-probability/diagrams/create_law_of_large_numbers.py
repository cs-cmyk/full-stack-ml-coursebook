#!/usr/bin/env python3
"""
Create law_of_large_numbers.png - showing convergence for fair and unfair coins
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
RED = '#F44336'

# Simulate 10,000 fair coin flips
n_flips = 10000
flips = np.random.choice(['H', 'T'], size=n_flips, p=[0.5, 0.5])

# Calculate running proportion of heads
cumulative_heads = np.cumsum(flips == 'H')
flip_numbers = np.arange(1, n_flips + 1)
running_proportion = cumulative_heads / flip_numbers

# Unfair coin (P(H) = 0.7)
flips_unfair = np.random.choice(['H', 'T'], size=n_flips, p=[0.7, 0.3])
cumulative_heads_unfair = np.cumsum(flips_unfair == 'H')
running_proportion_unfair = cumulative_heads_unfair / flip_numbers

# Create figure
plt.figure(figsize=(14, 5))

# Panel 1: Fair coin convergence
plt.subplot(1, 2, 1)
plt.plot(flip_numbers[:1000], running_proportion[:1000], linewidth=1.5, color=BLUE)
plt.axhline(y=0.5, color=RED, linestyle='--', linewidth=2, label='Theoretical P(H) = 0.5')
plt.fill_between(flip_numbers[:1000], 0.45, 0.55, alpha=0.2, color=RED, label='±0.05 band')
plt.xlabel('Number of Flips', fontsize=12)
plt.ylabel('Proportion of Heads', fontsize=12)
plt.title('Fair Coin: Convergence to 0.5', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0.3, 0.7)

# Add annotations at key points
plt.annotate(f'{running_proportion[9]:.3f}',
            xy=(10, running_proportion[9]), xytext=(80, 0.65),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
plt.annotate(f'{running_proportion[99]:.3f}',
            xy=(100, running_proportion[99]), xytext=(180, 0.35),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
plt.annotate(f'{running_proportion[999]:.3f}',
            xy=(1000, running_proportion[999]), xytext=(850, 0.55),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))

# Panel 2: Unfair coin convergence
plt.subplot(1, 2, 2)
plt.plot(flip_numbers[:1000], running_proportion_unfair[:1000],
         linewidth=1.5, color=GREEN)
plt.axhline(y=0.7, color=RED, linestyle='--', linewidth=2, label='Theoretical P(H) = 0.7')
plt.fill_between(flip_numbers[:1000], 0.65, 0.75, alpha=0.2, color=RED, label='±0.05 band')
plt.xlabel('Number of Flips', fontsize=12)
plt.ylabel('Proportion of Heads', fontsize=12)
plt.title('Unfair Coin (Biased): Convergence to 0.7', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 0.9)

# Add annotations
plt.annotate(f'{running_proportion_unfair[9]:.3f}',
            xy=(10, running_proportion_unfair[9]), xytext=(80, 0.55),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))
plt.annotate(f'{running_proportion_unfair[999]:.3f}',
            xy=(1000, running_proportion_unfair[999]), xytext=(850, 0.75),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-01-foundations/ch03-probability/diagrams/law_of_large_numbers.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Created: law_of_large_numbers.png")

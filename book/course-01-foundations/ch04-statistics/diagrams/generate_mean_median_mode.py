#!/usr/bin/env python3
"""Generate mean/median/mode distribution comparison diagram"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create figure with three distribution types
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Symmetric distribution
np.random.seed(42)
symmetric = np.random.normal(50, 10, 1000)
axes[0].hist(symmetric, bins=30, alpha=0.7, color='#2196F3', edgecolor='black')
mean_sym = np.mean(symmetric)
median_sym = np.median(symmetric)
axes[0].axvline(mean_sym, color='#F44336', linestyle='--', linewidth=2, label=f'Mean = {mean_sym:.1f}')
axes[0].axvline(median_sym, color='#4CAF50', linestyle='--', linewidth=2, label=f'Median = {median_sym:.1f}')
axes[0].set_title('Symmetric Distribution\nMean ≈ Median ≈ Mode', fontsize=12, weight='bold')
axes[0].set_xlabel('Value', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Right-skewed distribution
right_skewed = np.random.exponential(20, 1000) + 30
axes[1].hist(right_skewed, bins=30, alpha=0.7, color='#FF9800', edgecolor='black')
mean_right = np.mean(right_skewed)
median_right = np.median(right_skewed)
axes[1].axvline(mean_right, color='#F44336', linestyle='--', linewidth=2, label=f'Mean = {mean_right:.1f}')
axes[1].axvline(median_right, color='#4CAF50', linestyle='--', linewidth=2, label=f'Median = {median_right:.1f}')
axes[1].set_title('Right-Skewed Distribution\nMode < Median < Mean', fontsize=12, weight='bold')
axes[1].set_xlabel('Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Left-skewed distribution
left_skewed = 100 - np.random.exponential(20, 1000)
axes[2].hist(left_skewed, bins=30, alpha=0.7, color='#4CAF50', edgecolor='black')
mean_left = np.mean(left_skewed)
median_left = np.median(left_skewed)
axes[2].axvline(mean_left, color='#F44336', linestyle='--', linewidth=2, label=f'Mean = {mean_left:.1f}')
axes[2].axvline(median_left, color='#9C27B0', linestyle='--', linewidth=2, label=f'Median = {median_left:.1f}')
axes[2].set_title('Left-Skewed Distribution\nMean < Median < Mode', fontsize=12, weight='bold')
axes[2].set_xlabel('Value', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_median_mode.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated mean_median_mode.png")

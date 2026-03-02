#!/usr/bin/env python3
"""Generate visual comparison: Single Split vs 5-Fold Cross-Validation"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set style for consistent appearance
plt.style.use('default')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Left panel: Single Split
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 2)
ax1.axis('off')
ax1.set_title('Single Train/Test Split', fontsize=14, fontweight='bold')

# Training set (80%)
train_rect = patches.Rectangle((0, 0.5), 8, 0.8, linewidth=2,
                                edgecolor='#2196F3', facecolor='#90CAF9')
ax1.add_patch(train_rect)
ax1.text(4, 0.9, 'Training (80%)', ha='center', va='center', fontsize=11)

# Test set (20%)
test_rect = patches.Rectangle((8, 0.5), 2, 0.8, linewidth=2,
                               edgecolor='#F44336', facecolor='#EF9A9A')
ax1.add_patch(test_rect)
ax1.text(9, 0.9, 'Test\n(20%)', ha='center', va='center', fontsize=11)

ax1.text(5, 0.1, 'One estimate, high variance', ha='center',
         fontsize=10, style='italic')

# Right panel: 5-Fold Cross-Validation
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')
ax2.set_title('5-Fold Cross-Validation', fontsize=14, fontweight='bold')

fold_width = 2.0

for fold_idx in range(5):
    y_pos = 5 - fold_idx
    # Draw all 5 folds
    for i in range(5):
        x_pos = i * fold_width
        if i == fold_idx:
            # This fold is test
            rect = patches.Rectangle((x_pos, y_pos - 0.35), fold_width, 0.7,
                                      linewidth=1.5, edgecolor='#F44336',
                                      facecolor='#EF9A9A')
            if fold_idx == 0:  # Label only first row
                ax2.text(x_pos + fold_width/2, y_pos + 0.5, f'Fold {i+1}',
                         ha='center', fontsize=8)
        else:
            # This fold is training
            rect = patches.Rectangle((x_pos, y_pos - 0.35), fold_width, 0.7,
                                      linewidth=1, edgecolor='#2196F3',
                                      facecolor='#90CAF9', alpha=0.6)
            if fold_idx == 0:  # Label only first row
                ax2.text(x_pos + fold_width/2, y_pos + 0.5, f'Fold {i+1}',
                         ha='center', fontsize=8)
        ax2.add_patch(rect)

    # Label iteration
    ax2.text(-0.5, y_pos, f'Iter {fold_idx + 1}', ha='right', va='center', fontsize=9)

# Legend
ax2.text(5, 0.3, 'Five estimates averaged → Lower variance', ha='center',
         fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-04-ml/ch21-model-selection/diagrams/cross_validation_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: cross_validation_comparison.png")
plt.close()

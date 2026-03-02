"""
Walk-forward validation visualization
Demonstrates expanding window strategy for time series cross-validation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create walk-forward validation timeline visualization
fig, ax = plt.subplots(figsize=(14, 6))

# Define fold parameters
folds = 4
start_train = 0
initial_train_size = 60
test_size = 12
total_size = 100

colors_train = ['#2196F3', '#1976D2', '#1565C0', '#0D47A1']
colors_test = ['#FF9800', '#F57C00', '#E64A19', '#D84315']

for fold in range(folds):
    train_end = initial_train_size + fold * test_size
    test_start = train_end
    test_end = test_start + test_size

    # Draw train bar
    ax.barh(fold, train_end - start_train, left=start_train, height=0.5,
            color=colors_train[fold], alpha=0.8, label='Train' if fold == 0 else '',
            edgecolor='white', linewidth=2)

    # Draw test bar
    ax.barh(fold, test_size, left=test_start, height=0.5,
            color=colors_test[fold], alpha=0.8, label='Test' if fold == 0 else '',
            edgecolor='white', linewidth=2)

    # Add annotations
    ax.text(train_end / 2, fold, f'Train: {start_train}–{train_end}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    ax.text(test_start + test_size / 2, fold, f'Test: {test_start}–{test_end}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax.set_yticks(range(folds))
ax.set_yticklabels([f'Fold {i+1}' for i in range(folds)], fontsize=12)
ax.set_xlabel('Time Points', fontsize=13, fontweight='bold')
ax.set_title('Walk-Forward Validation: Expanding Window Strategy', fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(0, total_size)
ax.set_ylim(-0.5, folds - 0.5)

# Add explanatory text
fig.text(0.5, 0.02, 'Training set expands over time while always predicting future observations\nPrevents data leakage by maintaining temporal ordering',
         ha='center', fontsize=11, style='italic', color='#555555')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig('/home/chirag/ds-book/book/course-07-time-series/ch30-advanced-ts/diagrams/walk_forward_validation.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: walk_forward_validation.png")
plt.close()

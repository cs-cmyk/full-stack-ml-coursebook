#!/usr/bin/env python3
"""Generate Diagram 1: Scaled Dot-Product Attention Flow"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
purple = '#9C27B0'
gray = '#607D8B'

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, fontsize=12):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", edgecolor='black',
                          facecolor=color, linewidth=2, alpha=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white')

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color=color)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=10, color=color)

# Input sequence
ax.text(5, 13, 'Input Sequence: [x₁, x₂, x₃]', ha='center', fontsize=14, fontweight='bold')

# Linear projections
create_box(ax, 2.5, 11.5, 1.8, 0.6, 'Linear W_Q', blue, 11)
create_box(ax, 5, 11.5, 1.8, 0.6, 'Linear W_K', green, 11)
create_box(ax, 7.5, 11.5, 1.8, 0.6, 'Linear W_V', orange, 11)

# Arrows from input to projections
create_arrow(ax, 3.5, 12.8, 2.5, 11.9, color=gray)
create_arrow(ax, 5, 12.8, 5, 11.9, color=gray)
create_arrow(ax, 6.5, 12.8, 7.5, 11.9, color=gray)

# Q, K, V matrices
create_box(ax, 2.5, 10, 1.8, 0.6, 'Q (n×d_k)', blue, 11)
create_box(ax, 5, 10, 1.8, 0.6, 'K (n×d_k)', green, 11)
create_box(ax, 7.5, 10, 1.8, 0.6, 'V (n×d_v)', orange, 11)

# Arrows from projections to matrices
create_arrow(ax, 2.5, 11.1, 2.5, 10.4, color=blue)
create_arrow(ax, 5, 11.1, 5, 10.4, color=green)
create_arrow(ax, 7.5, 11.1, 7.5, 10.4, color=orange)

# Q @ K^T operation
create_box(ax, 3.75, 8.5, 2.2, 0.6, 'Q @ K^T', purple, 11)
ax.text(3.75, 7.9, '(n×n scores)', ha='center', fontsize=9, style='italic')

# Arrows to Q @ K^T
create_arrow(ax, 2.5, 9.6, 3.2, 8.9, color=blue)
create_arrow(ax, 5, 9.6, 4.3, 8.9, color=green)

# Scaling
create_box(ax, 3.75, 6.8, 2.2, 0.6, '÷ √(d_k)', purple, 11)
ax.text(3.75, 6.2, '(scaling)', ha='center', fontsize=9, style='italic')
create_arrow(ax, 3.75, 8.1, 3.75, 7.2, color=purple)

# Softmax
create_box(ax, 3.75, 5.1, 2.2, 0.6, 'softmax', purple, 11)
ax.text(3.75, 4.5, '(n×n weights)', ha='center', fontsize=9, style='italic')
create_arrow(ax, 3.75, 6.4, 3.75, 5.5, color=purple)

# Weighted sum with V
create_box(ax, 5, 3, 2.5, 0.6, 'weights @ V', purple, 11)

# Arrows to weighted sum
create_arrow(ax, 3.75, 4.7, 4.5, 3.4, color=purple)
create_arrow(ax, 7.5, 9.6, 6.5, 3.4, color=orange)

# Output
create_box(ax, 5, 1.5, 2.5, 0.6, 'Output (n×d_v)', purple, 11)
create_arrow(ax, 5, 2.6, 5, 1.9, color=purple)

# Add title
ax.text(5, 0.5, 'Scaled Dot-Product Attention Mechanism',
        ha='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch25-transformers/diagrams/attention_flow.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: attention_flow.png")

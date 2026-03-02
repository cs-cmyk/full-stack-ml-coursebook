#!/usr/bin/env python3
"""Generate Diagram 3: Complete Transformer Encoder Block"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
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
yellow = '#FFC107'
gray = '#607D8B'

# Helper functions
def create_box(ax, x, y, width, height, text, color, fontsize=11):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.15", edgecolor='black',
                          facecolor=color, linewidth=2, alpha=0.85)
    ax.add_patch(box)
    # Handle multi-line text
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', multialignment='center')

def create_arrow(ax, x1, y1, x2, y2, color='black', width=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=width, color=color)
    ax.add_patch(arrow)

def create_residual_connection(ax, x_start, y_start, y_end, side='left'):
    """Create a curved residual connection"""
    offset = -1.8 if side == 'left' else 1.8
    # Curved path
    arc_start_y = y_start
    arc_end_y = y_end

    # Draw the arc using multiple line segments
    x_arc = x_start + offset
    y_points = np.linspace(arc_start_y, arc_end_y, 50)
    x_curve = x_arc + 0.3 * np.sin((y_points - arc_start_y) / (arc_end_y - arc_start_y) * np.pi)

    for i in range(len(y_points) - 1):
        ax.plot([x_curve[i], x_curve[i+1]], [y_points[i], y_points[i+1]],
               color=yellow, linewidth=2.5, alpha=0.8)

    # Add arrow at the end
    create_arrow(ax, x_curve[-2], y_points[-2], x_start, y_end - 0.3, yellow, 2.5)

# Input
create_box(ax, 5, 12.5, 4, 0.7, 'Input:\nToken Embeddings +\nPositional Encoding', gray, 10)

# Multi-Head Self-Attention
create_box(ax, 5, 10.5, 3.5, 0.8, 'Multi-Head\nSelf-Attention', blue, 11)

# Arrow from input to attention
create_arrow(ax, 5, 12, 5, 11, gray, 2)

# Residual connection 1
create_residual_connection(ax, 5, 12, 9.3, side='left')

# Add operation (residual + attention output)
create_box(ax, 5, 9, 2, 0.6, 'Add', yellow, 11)

# Arrows to Add
create_arrow(ax, 5, 10, 5, 9.4, blue, 2)

# Layer Normalization 1
create_box(ax, 5, 7.8, 2.5, 0.6, 'Layer Norm', green, 11)

# Arrow to Layer Norm
create_arrow(ax, 5, 8.6, 5, 8.2, yellow, 2)

# Feed-Forward Network
create_box(ax, 5, 6.3, 3.5, 0.8, 'Position-wise\nFeed-Forward\nNetwork', orange, 11)

# Arrow to FFN
create_arrow(ax, 5, 7.4, 5, 6.8, green, 2)

# Residual connection 2
create_residual_connection(ax, 5, 7.8, 4.8, side='right')

# Add operation 2
create_box(ax, 5, 4.5, 2, 0.6, 'Add', yellow, 11)

# Arrow to Add 2
create_arrow(ax, 5, 5.8, 5, 4.9, orange, 2)

# Layer Normalization 2
create_box(ax, 5, 3.2, 2.5, 0.6, 'Layer Norm', green, 11)

# Arrow to Layer Norm 2
create_arrow(ax, 5, 4.1, 5, 3.6, yellow, 2)

# Output
create_box(ax, 5, 1.5, 4, 0.7, 'Output to Next Block\nor Classification Head', gray, 10)

# Arrow to output
create_arrow(ax, 5, 2.8, 5, 1.9, green, 2)

# Add labels
ax.text(2.2, 9, 'Residual\nConnection', ha='center', fontsize=9,
        style='italic', color=yellow, fontweight='bold')
ax.text(7.8, 4.5, 'Residual\nConnection', ha='center', fontsize=9,
        style='italic', color=yellow, fontweight='bold')

# Title
fig.suptitle('Transformer Encoder Block Architecture', fontsize=16, fontweight='bold', y=0.96)

# Add subtitle
ax.text(5, 0.5, 'This block is typically stacked N times (N=6-24 for modern transformers)',
        ha='center', fontsize=10, style='italic', color=gray)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch25-transformers/diagrams/encoder_block.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: encoder_block.png")

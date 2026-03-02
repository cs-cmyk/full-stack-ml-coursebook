#!/usr/bin/env python3
"""Generate Diagram 2: Multi-Head Attention Architecture"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
purple = '#9C27B0'
gray = '#607D8B'
red = '#F44336'

# Helper functions
def create_box(ax, x, y, width, height, text, color, fontsize=11):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.1", edgecolor='black',
                          facecolor=color, linewidth=2, alpha=0.8)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white', wrap=True)

def create_arrow(ax, x1, y1, x2, y2, color='black', style='->', width=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=20,
                           linewidth=width, color=color)
    ax.add_patch(arrow)

# Input
ax.text(6, 11, 'Input (batch, n, d_model)', ha='center', fontsize=13, fontweight='bold')

# Linear projections for Q, K, V
create_box(ax, 2.5, 9.5, 2, 0.6, 'Linear W_Q', blue, 11)
create_box(ax, 6, 9.5, 2, 0.6, 'Linear W_K', green, 11)
create_box(ax, 9.5, 9.5, 2, 0.6, 'Linear W_V', orange, 11)

# Arrows from input to projections
create_arrow(ax, 3.5, 10.7, 2.5, 9.9, gray)
create_arrow(ax, 6, 10.7, 6, 9.9, gray)
create_arrow(ax, 8.5, 10.7, 9.5, 9.9, gray)

# Reshape operations
create_box(ax, 2.5, 8, 2, 0.6, 'Reshape', blue, 10)
create_box(ax, 6, 8, 2, 0.6, 'Reshape', green, 10)
create_box(ax, 9.5, 8, 2, 0.6, 'Reshape', orange, 10)

ax.text(2.5, 7.4, '(batch, h, n, d_k)', ha='center', fontsize=8, style='italic')
ax.text(6, 7.4, '(batch, h, n, d_k)', ha='center', fontsize=8, style='italic')
ax.text(9.5, 7.4, '(batch, h, n, d_v)', ha='center', fontsize=8, style='italic')

# Arrows to reshape
create_arrow(ax, 2.5, 9.1, 2.5, 8.4, blue)
create_arrow(ax, 6, 9.1, 6, 8.4, green)
create_arrow(ax, 9.5, 9.1, 9.5, 8.4, orange)

# Multi-head attention boxes (representing h parallel heads)
head_colors = [purple, '#E91E63', '#00BCD4', '#8BC34A']
for i, color in enumerate(head_colors):
    x_offset = 2.5 + i * 1.5
    y = 6
    create_box(ax, x_offset, y, 1, 0.5, f'Head {i+1}', color, 9)

# Arrows to heads
create_arrow(ax, 2.5, 7.6, 2.5, 6.3, blue)
create_arrow(ax, 6, 7.6, 5, 6.3, green)
create_arrow(ax, 9.5, 7.6, 8, 6.3, orange)

# Scaled dot-product attention (central operation)
create_box(ax, 6, 4.5, 4, 0.7, 'Scaled Dot-Product Attention', purple, 11)
ax.text(6, 3.9, '(applied per head)', ha='center', fontsize=9, style='italic')

# Arrow to attention
create_arrow(ax, 5, 5.7, 6, 5.0, gray)

# Output from attention
ax.text(6, 3.2, '(batch, h, n, d_v)', ha='center', fontsize=9, style='italic')

# Reshape/Concatenate
create_box(ax, 6, 2.5, 2.5, 0.6, 'Reshape', purple, 11)
ax.text(6, 1.9, '(batch, n, h×d_v)', ha='center', fontsize=9, style='italic')

# Arrow to reshape
create_arrow(ax, 6, 4.1, 6, 2.9, purple)

# Linear output projection
create_box(ax, 6, 1, 2.5, 0.6, 'Linear W_O', red, 11)

# Arrow to output projection
create_arrow(ax, 6, 2.1, 6, 1.4, purple)

# Final output
ax.text(6, 0.3, 'Output (batch, n, d_model)', ha='center', fontsize=12, fontweight='bold')

# Arrow to final output
create_arrow(ax, 6, 0.6, 6, 0.5, red)

# Add annotations
ax.text(0.5, 6, 'h parallel\nheads', ha='center', fontsize=10,
        style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Title
fig.suptitle('Multi-Head Attention Architecture', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch25-transformers/diagrams/multihead_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: multihead_architecture.png")

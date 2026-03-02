#!/usr/bin/env python3
"""
Diagram 1: ML System Design Patterns Comparison
Creates a visual comparison of Batch, Real-Time, and Streaming inference patterns
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'
PURPLE = '#9C27B0'
GRAY = '#607D8B'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ML System Design Patterns Comparison', fontsize=16, fontweight='bold', y=0.98)

# Helper function to draw boxes and arrows
def draw_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor=color,
                         linewidth=1.5, alpha=0.8)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize,
           weight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=20,
                          linewidth=2, color='black',
                          connectionstyle="arc3,rad=0")
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.1, mid_y, label, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ===== Pattern 1: Batch Prediction =====
ax1 = axes[0]
ax1.set_xlim(0, 4)
ax1.set_ylim(0, 5)
ax1.axis('off')
ax1.set_title('Batch Prediction (Offline)', fontsize=13, fontweight='bold', pad=10)

draw_box(ax1, 0.5, 4, 3, 0.6, 'Data Warehouse', BLUE, 9)
draw_arrow(ax1, 2, 4, 2, 3.3, 'Daily Extract')
draw_box(ax1, 0.5, 2.7, 3, 0.6, 'Feature Computation', GREEN, 9)
draw_arrow(ax1, 2, 2.7, 2, 2)
draw_box(ax1, 0.5, 1.4, 3, 0.6, 'Model Inference\nn=50,000 samples', ORANGE, 9)
draw_arrow(ax1, 2, 1.4, 2, 0.7)
draw_box(ax1, 0.5, 0.1, 3, 0.6, 'Write Predictions\nto Database', PURPLE, 9)

# Latency indicator
ax1.text(2, -0.5, 'Latency: Hours to Days', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor=RED, alpha=0.3), weight='bold')

# ===== Pattern 2: Real-Time Inference =====
ax2 = axes[1]
ax2.set_xlim(0, 4)
ax2.set_ylim(0, 5)
ax2.axis('off')
ax2.set_title('Real-Time Inference (Online)', fontsize=13, fontweight='bold', pad=10)

draw_box(ax2, 0.5, 4, 3, 0.6, 'User Request', BLUE, 9)
draw_arrow(ax2, 2, 4, 2, 3.3, 'HTTP/gRPC')
draw_box(ax2, 0.5, 2.7, 3, 0.6, 'Feature Lookup\nOnline Store', GREEN, 9)
draw_arrow(ax2, 2, 2.7, 2, 2)
draw_box(ax2, 0.5, 1.4, 3, 0.6, 'Model Inference\nn=1 sample', ORANGE, 9)
draw_arrow(ax2, 2, 1.4, 2, 0.7)
draw_box(ax2, 0.5, 0.1, 3, 0.6, 'Return Response', PURPLE, 9)

# Latency indicator
ax2.text(2, -0.5, 'Latency: 10-100ms', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor=GREEN, alpha=0.3), weight='bold')

# ===== Pattern 3: Streaming Inference =====
ax3 = axes[2]
ax3.set_xlim(0, 4)
ax3.set_ylim(0, 5)
ax3.axis('off')
ax3.set_title('Streaming Inference', fontsize=13, fontweight='bold', pad=10)

draw_box(ax3, 0.5, 4, 3, 0.6, 'Event Stream\nKafka/Kinesis', BLUE, 9)
draw_arrow(ax3, 2, 4, 2, 3.3, 'Continuous')
draw_box(ax3, 0.5, 2.7, 3, 0.6, 'Stateful Processing\nWindowed Aggregates', GREEN, 9)
draw_arrow(ax3, 2, 2.7, 2, 2)
draw_box(ax3, 0.5, 1.4, 3, 0.6, 'Model Inference\nContinuous', ORANGE, 9)
draw_arrow(ax3, 2, 1.4, 2, 0.7)
draw_box(ax3, 0.5, 0.1, 3, 0.6, 'Output Stream\nor Data Store', PURPLE, 9)

# Latency indicator
ax3.text(2, -0.5, 'Latency: Seconds', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor=ORANGE, alpha=0.3), weight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-18/ch51/diagrams/diagram1_system_patterns.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Diagram 1 saved: diagram1_system_patterns.png")
plt.close()

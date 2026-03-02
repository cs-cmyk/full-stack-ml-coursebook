#!/usr/bin/env python3
"""
Diagram 2: Feature Store Architecture
Shows the dual-path architecture with offline and online stores
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

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')
fig.suptitle('Feature Store Architecture', fontsize=16, fontweight='bold')

# Helper functions
def draw_box(x, y, width, height, text, color, fontsize=9, alpha=0.8):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor=color,
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    # Split text on newlines for multi-line support
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize,
           weight='bold', multialignment='center')

def draw_arrow(x1, y1, x2, y2, label='', style='->', color='black', linewidth=2, linestyle='solid'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, mutation_scale=20,
                          linewidth=linewidth, color=color,
                          linestyle=linestyle,
                          connectionstyle="arc3,rad=0.1")
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y + 0.2, label, fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

def draw_section_box(x, y, width, height, title, color):
    """Draw a section container box"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor=color, facecolor='none',
                         linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height + 0.15, title,
           ha='center', va='bottom', fontsize=11,
           weight='bold', color=color)

# ===== Data Sources Section (Top) =====
draw_section_box(0.5, 9.5, 13, 2, 'Data Sources', BLUE)
draw_box(1, 10, 5.5, 1, 'Batch Data\nData Warehouse', BLUE, 10)
draw_box(7.5, 10, 5.5, 1, 'Streaming Data\nKafka', BLUE, 10)

# ===== Feature Computation Section =====
draw_section_box(0.5, 6.8, 13, 2.2, 'Feature Computation', GREEN)
draw_box(1, 7.3, 5.5, 1.2, 'Batch Feature Pipeline\nSpark/Airflow', GREEN, 10)
draw_box(7.5, 7.3, 5.5, 1.2, 'Streaming Feature Pipeline\nFlink/Beam', GREEN, 10)

# Arrows from data sources to computation
draw_arrow(3.75, 10, 3.75, 8.5)
draw_arrow(10.25, 10, 10.25, 8.5)

# ===== Feature Store Section =====
draw_section_box(0.5, 3.5, 13, 2.8, 'Feature Store', PURPLE)

# Feature Registry (center top)
draw_box(5, 5.5, 4, 0.7, 'Feature Registry\nMetadata & Definitions', GRAY, 9)

# Offline Store (left)
draw_box(1, 3.8, 5.5, 1.3, 'Offline Store\nParquet/BigQuery\nHistorical Features', ORANGE, 9)

# Online Store (right)
draw_box(7.5, 3.8, 5.5, 1.3, 'Online Store\nRedis/DynamoDB\nLatest Features', ORANGE, 9)

# Arrows from computation to stores
draw_arrow(3.75, 7.3, 3.75, 5.1)
draw_arrow(10.25, 7.3, 10.25, 5.1)

# Arrow from batch to online (materialize)
draw_arrow(6.5, 4.4, 7.5, 4.4, 'Materialize', color=PURPLE, linestyle='dashed')

# Arrows from registry to computation (dashed - defines)
draw_arrow(6, 5.5, 4.5, 8.5, 'Defines', color=GRAY, linestyle='dashed', linewidth=1.5)
draw_arrow(8, 5.5, 9.5, 8.5, 'Defines', color=GRAY, linestyle='dashed', linewidth=1.5)

# ===== Consumption Section =====
draw_section_box(0.5, 0.5, 13, 2.5, 'Consumption', RED)

# Training Pipeline (left)
draw_box(1, 1, 5.5, 1.2, 'Training Pipeline\nPoint-in-Time Queries', RED, 10)

# Serving API (right)
draw_box(7.5, 1, 5.5, 1.2, 'Serving API\nLow-Latency Lookup', RED, 10)

# Arrows from stores to consumption
draw_arrow(3.75, 3.8, 3.75, 2.2)
draw_arrow(10.25, 3.8, 10.25, 2.2)

# ===== Model in the middle =====
draw_box(5.5, 0.2, 3, 0.6, 'Model', GREEN, 10)

# Arrows from training to model and model to serving
draw_arrow(6.5, 1, 7, 0.8, 'Trains')
draw_arrow(8.5, 0.5, 7.5, 1)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-18/ch51/diagrams/diagram2_feature_store.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Diagram 2 saved: diagram2_feature_store.png")
plt.close()

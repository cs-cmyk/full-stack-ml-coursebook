#!/usr/bin/env python3
"""
Diagram 3: Model Serving Stack
Shows production model serving infrastructure with load balancing and observability
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

fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 13)
ax.axis('off')
fig.suptitle('Model Serving Stack', fontsize=16, fontweight='bold')

# Helper functions
def draw_box(x, y, width, height, text, color, fontsize=9, alpha=0.8):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor=color,
                         linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
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
        ax.text(mid_x + 0.2, mid_y + 0.1, label, fontsize=7,
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

# ===== Clients Section (Top) =====
draw_section_box(0.5, 10.5, 13, 1.8, 'Clients', BLUE)
draw_box(1.5, 11, 3, 0.8, 'Web Application', BLUE, 9)
draw_box(5.5, 11, 3, 0.8, 'Mobile App', BLUE, 9)
draw_box(9.5, 11, 3, 0.8, 'Backend Service', BLUE, 9)

# ===== Load Balancer =====
draw_box(5, 9, 4, 0.8, 'Load Balancer\nAPI Gateway', PURPLE, 10)

# Arrows from clients to load balancer
draw_arrow(3, 11, 6.5, 9.8)
draw_arrow(7, 11, 7, 9.8)
draw_arrow(11, 11, 7.5, 9.8)

# ===== Model Servers Section =====
draw_section_box(0.5, 5.8, 9, 2.7, 'Model Servers (Auto-scaled)', GREEN)

draw_box(1, 7.5, 2.7, 1.5, 'Model Server\nReplica 1\nTensorFlow\nServing/Triton', GREEN, 8)
draw_box(4.2, 7.5, 2.7, 1.5, 'Model Server\nReplica 2', GREEN, 8)
draw_box(7.4, 7.5, 2.0, 1.5, 'Model Server\nReplica N', GREEN, 8)

# Arrows from load balancer to model servers
draw_arrow(6, 9, 2.3, 9)
draw_arrow(7, 9, 5.5, 9)
draw_arrow(8, 9, 8.4, 9)

# ===== Supporting Services (middle) =====
# Model Repository
draw_box(10.5, 8.5, 3, 1.2, 'Model Repository\nS3 / MLflow\nRegistry', ORANGE, 9)

# Feature Store
draw_box(10.5, 6.5, 3, 1.2, 'Feature Store\nOnline Store', ORANGE, 9)

# Arrows from model repo to servers (dashed - load model)
draw_arrow(10.5, 9.1, 9.4, 8.5, 'Load Model', color=GRAY, linestyle='dashed', linewidth=1.5)
draw_arrow(10.5, 8.8, 6.9, 8.3, 'Load Model', color=GRAY, linestyle='dashed', linewidth=1.5)
draw_arrow(10.5, 8.6, 3.7, 8.1, 'Load Model', color=GRAY, linestyle='dashed', linewidth=1.5)

# Arrows from servers to feature store
draw_arrow(3.7, 7.5, 10.5, 7.1)
draw_arrow(5.5, 7.5, 10.5, 7.0)
draw_arrow(8.4, 7.5, 10.5, 6.9)

# ===== Observability Section (Bottom) =====
draw_section_box(0.5, 0.5, 9, 4.8, 'Observability', RED)

draw_box(1.5, 3.5, 2.5, 1.2, 'Logging\nSplunk/ELK', RED, 9)
draw_box(4.5, 3.5, 2.5, 1.2, 'Metrics\nPrometheus/\nDatadog', RED, 9)
draw_box(7.5, 3.5, 1.5, 1.2, 'Tracing\nJaeger', RED, 9)

# Arrows from model servers to observability
draw_arrow(2.3, 7.5, 2.3, 4.7)
draw_arrow(2.5, 7.5, 5.7, 4.7)
draw_arrow(2.7, 7.5, 8.2, 4.7)

# Add data flow descriptions
ax.text(5, 2.5, 'Logs: Request/response data, errors\nMetrics: Latency, throughput, CPU/GPU usage\nTraces: End-to-end request tracking',
       ha='center', va='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='white', edgecolor=RED, alpha=0.8))

# Add key features box
ax.text(11.5, 4.5, 'Key Features:\n\n• Horizontal scaling\n• Load distribution\n• Version control\n• Low-latency features\n• Full observability\n• Health monitoring',
       ha='center', va='center', fontsize=9,
       bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-18/ch51/diagrams/diagram3_model_serving.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Diagram 3 saved: diagram3_model_serving.png")
plt.close()

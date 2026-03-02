import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure for three-pass reading framework
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_pass1 = '#e8f4f8'
color_pass2 = '#b3d9e6'
color_pass3 = '#7cb5d1'

# Pass 1: Quick Scan
y_pos = 8
box1 = FancyBboxPatch((0.5, y_pos-0.8), 8.5, 1.6,
                       boxstyle="round,pad=0.1",
                       edgecolor='#2c5f7a', facecolor=color_pass1, linewidth=2)
ax.add_patch(box1)
ax.text(1, y_pos+0.5, 'PASS 1: Quick Scan (5-10 minutes)',
        fontsize=13, weight='bold', color='#1a3a4a')
ax.text(1, y_pos, 'Read: Title • Abstract • Intro (first 2¶) • Section headers • Conclusion • Figures',
        fontsize=9, color='#2c5f7a')
ax.text(1, y_pos-0.4, 'Goal: Should I read this paper?  →  Decision: Discard / Queue / Read now',
        fontsize=9, style='italic', color='#2c5f7a')

# Arrow 1->2
arrow1 = FancyArrowPatch((4.75, y_pos-0.9), (4.75, y_pos-1.5),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2, color='#2c5f7a')
ax.add_patch(arrow1)
ax.text(5, y_pos-1.2, 'If relevant', fontsize=8, style='italic', color='#2c5f7a')

# Pass 2: Focused Read
y_pos = 5
box2 = FancyBboxPatch((0.5, y_pos-0.8), 8.5, 1.6,
                       boxstyle="round,pad=0.1",
                       edgecolor='#2c5f7a', facecolor=color_pass2, linewidth=2)
ax.add_patch(box2)
ax.text(1, y_pos+0.5, 'PASS 2: Focused Read (30-60 minutes)',
        fontsize=13, weight='bold', color='#1a3a4a')
ax.text(1, y_pos, 'Read: Full intro • Methods (skip proofs) • Experiments • Ablations • Figures deeply',
        fontsize=9, color='#2c5f7a')
ax.text(1, y_pos-0.4, 'Goal: Understand contribution  →  Outcome: Can explain key idea & results',
        fontsize=9, style='italic', color='#2c5f7a')

# Arrow 2->3
arrow2 = FancyArrowPatch((4.75, y_pos-0.9), (4.75, y_pos-1.5),
                         arrowstyle='->', mutation_scale=20,
                         linewidth=2, color='#2c5f7a')
ax.add_patch(arrow2)
ax.text(5, y_pos-1.2, 'If implementing', fontsize=8, style='italic', color='#2c5f7a')

# Pass 3: Deep Dive
y_pos = 2
box3 = FancyBboxPatch((0.5, y_pos-0.8), 8.5, 1.6,
                       boxstyle="round,pad=0.1",
                       edgecolor='#2c5f7a', facecolor=color_pass3, linewidth=2)
ax.add_patch(box3)
ax.text(1, y_pos+0.5, 'PASS 3: Deep Dive (3-6 hours)',
        fontsize=13, weight='bold', color='#1a3a4a')
ax.text(1, y_pos, 'Read: Everything including appendix • Work through math • Check all details',
        fontsize=9, color='#2c5f7a')
ax.text(1, y_pos-0.4, 'Goal: Could I reproduce this?  →  Outcome: Implementation-ready understanding',
        fontsize=9, style='italic', color='#2c5f7a')

plt.title('The Three-Pass Reading Framework\nProgressive Depth: Most papers need only Pass 1 or 2',
          fontsize=15, weight='bold', pad=20, color='#1a3a4a')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch63/diagrams/three_pass_framework.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ three_pass_framework.png saved")

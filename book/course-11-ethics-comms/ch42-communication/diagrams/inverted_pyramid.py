"""
Inverted Pyramid Report Structure Diagram
Educational visualization showing how to structure data science reports
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors from the textbook palette
colors = {
    'executive': '#FF9800',       # orange (most important)
    'methodology': '#4CAF50',      # green (middle)
    'technical': '#2196F3'         # blue (detail)
}

# Draw inverted pyramid (trapezoids stacked)
# Each level represents decreasing width (more detail, fewer readers)
levels = [
    {'name': 'Executive Summary', 'color': colors['executive'],
     'points': [(1, 8.5), (9, 8.5), (8, 6.5), (2, 6.5)],
     'subtitle': '(Widest - Most Important)',
     'text': 'Key findings, recommendations,\nbusiness impact',
     'center': (5, 7.5),
     'border_width': 4},
    {'name': 'Methodology & Results', 'color': colors['methodology'],
     'points': [(2, 6.5), (8, 6.5), (7, 4.5), (3, 4.5)],
     'subtitle': '(Middle Layer)',
     'text': 'Approach, data sources,\nvisualizations, metrics',
     'center': (5, 5.5),
     'border_width': 3},
    {'name': 'Technical Appendix', 'color': colors['technical'],
     'points': [(3, 4.5), (7, 4.5), (6, 2.5), (4, 2.5)],
     'subtitle': '(Narrowest - Most Detail)',
     'text': 'Code, full tables,\nmathematical proofs, assumptions',
     'center': (5, 3.5),
     'border_width': 2}
]

# Draw each level
for level in levels:
    polygon = mpatches.Polygon(level['points'], closed=True,
                               facecolor=level['color'],
                               edgecolor='black',
                               linewidth=level['border_width'],
                               alpha=0.85)
    ax.add_patch(polygon)

    # Add title text (bold)
    ax.text(level['center'][0], level['center'][1] + 0.55, level['name'],
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='white', zorder=10)

    # Add subtitle
    ax.text(level['center'][0], level['center'][1] + 0.2, level['subtitle'],
            ha='center', va='center', fontsize=10,
            color='white', zorder=10, style='italic')

    # Add description text
    ax.text(level['center'][0], level['center'][1] - 0.35, level['text'],
            ha='center', va='center', fontsize=10,
            color='white', zorder=10, multialignment='center')

# Add title
ax.text(5, 9.5, 'The Inverted Pyramid Report Structure',
        ha='center', va='center', fontsize=18, fontweight='bold')

# Add explanation at bottom
explanation_text = [
    'Start with the most critical information (what was found, what should be done)',
    'and progressively add detail. Busy executives read only the top;',
    'technical reviewers read everything.'
]
y_pos = 1.6
for line in explanation_text:
    ax.text(5, y_pos, line,
            ha='center', va='center', fontsize=11, style='italic', color='#555555')
    y_pos -= 0.3

# Add side annotations with icons/markers
# Left side: Audience markers
ax.text(0.3, 7.5, '📊', ha='center', va='center', fontsize=20)
ax.text(0.3, 7.1, 'Executives', ha='center', va='center',
        fontsize=9, color='#555555', rotation=0)

ax.text(0.3, 5.5, '👥', ha='center', va='center', fontsize=20)
ax.text(0.3, 5.1, 'Managers', ha='center', va='center',
        fontsize=9, color='#555555', rotation=0)

ax.text(0.3, 3.5, '🔬', ha='center', va='center', fontsize=20)
ax.text(0.3, 3.1, 'Technicians', ha='center', va='center',
        fontsize=9, color='#555555', rotation=0)

# Right side: Reading depth arrow
ax.annotate('', xy=(9.5, 8), xytext=(9.5, 3),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#9C27B0'))
ax.text(9.8, 5.5, 'Increasing\nDepth &\nDetail', ha='left', va='center',
        fontsize=10, color='#9C27B0', fontweight='bold', rotation=-90)

# Add reading pattern markers
# Top: Most readers
ax.plot([1.5, 8.5], [8.8, 8.8], 'k-', linewidth=2, alpha=0.3)
ax.text(5, 9, '← Most readers start here →', ha='center', va='center',
        fontsize=9, color='#555555', style='italic')

# Bottom: Fewest readers
ax.plot([4.5, 5.5], [2.3, 2.3], 'k-', linewidth=2, alpha=0.3)
ax.text(5, 2.05, 'Fewest readers', ha='center', va='center',
        fontsize=9, color='#555555', style='italic')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-11-ethics-comms/ch42-communication/diagrams/inverted_pyramid.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Inverted pyramid diagram saved")
plt.close()

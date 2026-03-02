"""
Stakeholder Pyramid Diagram
Educational visualization showing information needs by organizational level
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors from the textbook palette
colors = {
    'executives': '#F44336',      # red
    'managers': '#4CAF50',         # green
    'experts': '#2196F3',          # blue
    'users': '#9C27B0'             # purple
}

# Draw inverted pyramid (trapezoids stacked)
# Each level: (x_left, x_right, y_bottom, y_top)
levels = [
    {'name': 'Executives', 'color': colors['executives'],
     'points': [(3, 9), (7, 9), (6.5, 7.5), (3.5, 7.5)],
     'text': 'Strategy, ROI, One-Number Summary',
     'center': (5, 8.3)},
    {'name': 'Managers', 'color': colors['managers'],
     'points': [(3.5, 7.5), (6.5, 7.5), (6, 6), (4, 6)],
     'text': 'Action Items, Resource Allocation, Timelines',
     'center': (5, 6.8)},
    {'name': 'Domain Experts', 'color': colors['experts'],
     'points': [(4, 6), (6, 6), (5.5, 4.5), (4.5, 4.5)],
     'text': 'Methodology, Assumptions, Limitations',
     'center': (5, 5.3)},
    {'name': 'End Users', 'color': colors['users'],
     'points': [(4.5, 4.5), (5.5, 4.5), (5.25, 3), (4.75, 3)],
     'text': 'Features, Usability, Support',
     'center': (5, 3.8)}
]

# Draw each level
for level in levels:
    polygon = mpatches.Polygon(level['points'], closed=True,
                               facecolor=level['color'],
                               edgecolor='black', linewidth=2.5, alpha=0.85)
    ax.add_patch(polygon)

    # Add title text (bold)
    ax.text(level['center'][0], level['center'][1] + 0.35, level['name'],
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white', zorder=10)

    # Add description text
    ax.text(level['center'][0], level['center'][1] - 0.2, level['text'],
            ha='center', va='center', fontsize=10,
            color='white', zorder=10, wrap=True)

# Add arrows showing flow
arrow_props = dict(arrowstyle='->', lw=2.5, color='#607D8B')
ax.annotate('', xy=(5, 7.3), xytext=(5, 7.6),
            arrowprops=arrow_props, zorder=5)
ax.annotate('', xy=(5, 5.8), xytext=(5, 6.1),
            arrowprops=arrow_props, zorder=5)
ax.annotate('', xy=(5, 4.3), xytext=(5, 4.6),
            arrowprops=arrow_props, zorder=5)

# Add title
ax.text(5, 9.7, 'The Stakeholder Pyramid',
        ha='center', va='center', fontsize=18, fontweight='bold')

# Add subtitle/explanation
ax.text(5, 2.2, 'Different organizational levels require different communication approaches.',
        ha='center', va='center', fontsize=11, style='italic', color='#555555')
ax.text(5, 1.9, 'Technical depth increases as the pyramid descends,',
        ha='center', va='center', fontsize=11, style='italic', color='#555555')
ax.text(5, 1.6, 'but executives at the top make the final decisions.',
        ha='center', va='center', fontsize=11, style='italic', color='#555555')

# Add side annotations
ax.text(8.5, 8.3, 'Decision\nMakers', ha='left', va='center',
        fontsize=10, color='#555555', style='italic')
ax.text(8.5, 3.8, 'Implementers', ha='left', va='center',
        fontsize=10, color='#555555', style='italic')

# Add arrow showing increasing technical detail
ax.annotate('', xy=(1.5, 3.5), xytext=(1.5, 8.5),
            arrowprops=dict(arrowstyle='<-', lw=2, color='#FF9800'))
ax.text(1, 6, 'Increasing\nTechnical\nDetail', ha='center', va='center',
        fontsize=9, color='#FF9800', fontweight='bold', rotation=90)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-11-ethics-comms/ch42-communication/diagrams/stakeholder_pyramid.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Stakeholder pyramid diagram saved")
plt.close()

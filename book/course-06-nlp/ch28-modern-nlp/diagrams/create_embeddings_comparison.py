"""
Create visualization comparing static and contextualized embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
np.random.seed(42)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color palette
blue = '#2196F3'
red = '#F44336'
gray = '#607D8B'

# Left panel: Static Embeddings
ax1 = axes[0]
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')

# All "bank" embeddings cluster at the same point
static_x, static_y = 0, 0
noise = 0.08  # Small noise for visibility

# River context (blue)
for i in range(4):
    x = static_x + np.random.randn() * noise
    y = static_y + np.random.randn() * noise
    ax1.scatter(x, y, s=300, c=blue, alpha=0.6, edgecolors='black', linewidth=2, zorder=3)

# Financial context (red) - overlapping
for i in range(4):
    x = static_x + np.random.randn() * noise
    y = static_y + np.random.randn() * noise
    ax1.scatter(x, y, s=300, c=red, alpha=0.6, edgecolors='black', linewidth=2, zorder=3)

# Add circle to show clustering
circle = Circle((static_x, static_y), 0.3, fill=False, edgecolor=gray,
                linewidth=2, linestyle='--', zorder=2)
ax1.add_patch(circle)

ax1.text(0, -1.5, 'All "bank" instances\nget identical vectors',
         fontsize=11, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax1.set_title('Static Embeddings (Word2Vec)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Dimension 1', fontsize=12)
ax1.set_ylabel('Dimension 2', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax1.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

# Right panel: Contextualized Embeddings
ax2 = axes[1]
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.set_aspect('equal')

# River context (blue) - clustered top-left
river_center = (-0.8, 0.8)
for i in range(4):
    x = river_center[0] + np.random.randn() * 0.15
    y = river_center[1] + np.random.randn() * 0.15
    ax2.scatter(x, y, s=300, c=blue, alpha=0.6, edgecolors='black', linewidth=2, zorder=3)

# Financial context (red) - clustered bottom-right
financial_center = (0.8, -0.8)
for i in range(4):
    x = financial_center[0] + np.random.randn() * 0.15
    y = financial_center[1] + np.random.randn() * 0.15
    ax2.scatter(x, y, s=300, c=red, alpha=0.6, edgecolors='black', linewidth=2, zorder=3)

# Add circles to show separate clusters
circle1 = Circle(river_center, 0.4, fill=False, edgecolor=blue,
                 linewidth=2, linestyle='--', zorder=2, alpha=0.7)
circle2 = Circle(financial_center, 0.4, fill=False, edgecolor=red,
                 linewidth=2, linestyle='--', zorder=2, alpha=0.7)
ax2.add_patch(circle1)
ax2.add_patch(circle2)

# Add labels
ax2.text(river_center[0], river_center[1] + 0.7, 'River\ncontext',
         fontsize=10, ha='center', va='bottom', color=blue, fontweight='bold')
ax2.text(financial_center[0], financial_center[1] - 0.7, 'Financial\ncontext',
         fontsize=10, ha='center', va='top', color=red, fontweight='bold')

ax2.text(0, -1.5, 'BERT creates different\nvectors based on context',
         fontsize=11, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax2.set_title('Contextualized Embeddings (BERT)', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Dimension 1', fontsize=12)
ax2.set_ylabel('Dimension 2', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
ax2.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

# Add legend
river_patch = mpatches.Patch(color=blue, label='"bank" in river context', alpha=0.6)
financial_patch = mpatches.Patch(color=red, label='"bank" in financial context', alpha=0.6)
fig.legend(handles=[river_patch, financial_patch], loc='upper center',
           ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/chirag/ds-book/book/course-06-nlp/ch28-modern-nlp/diagrams/static-vs-contextualized.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: static-vs-contextualized.png")

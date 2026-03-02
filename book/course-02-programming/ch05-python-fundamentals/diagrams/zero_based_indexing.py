"""
Zero-Based Indexing Visualization
Shows positive and negative indices for a list
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Define colors
color_blue = '#2196F3'
color_orange = '#FF9800'
color_gray = '#607D8B'

# List values
values = [10, 20, 30, 40, 50]
n = len(values)

# Draw list elements
box_width = 1.5
box_height = 1
start_x = 2
y_center = 3

for i, val in enumerate(values):
    x = start_x + i * box_width
    # Box
    rect = patches.FancyBboxPatch((x, y_center-box_height/2), box_width*0.95, box_height,
                                   boxstyle="round,pad=0.05",
                                   edgecolor=color_blue,
                                   facecolor=color_blue,
                                   linewidth=2.5,
                                   alpha=0.2)
    ax.add_patch(rect)
    # Value
    ax.text(x + box_width/2, y_center, str(val),
            ha='center', va='center', fontsize=18, weight='bold', color=color_blue)

# Positive indices (above)
ax.text(1, y_center + 1.2, 'Positive\nindices:', ha='right', va='center',
        fontsize=11, weight='bold', color=color_blue)
for i in range(n):
    x = start_x + i * box_width + box_width/2
    ax.text(x, y_center + 1.2, str(i), ha='center', va='center',
            fontsize=14, weight='bold', color=color_blue,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color_blue, linewidth=2))

# Negative indices (below)
ax.text(1, y_center - 1.2, 'Negative\nindices:', ha='right', va='center',
        fontsize=11, weight='bold', color=color_orange)
for i in range(n):
    x = start_x + i * box_width + box_width/2
    neg_idx = i - n
    ax.text(x, y_center - 1.2, str(neg_idx), ha='center', va='center',
            fontsize=14, weight='bold', color=color_orange,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color_orange, linewidth=2))

# Title
ax.text(6, 5.5, 'Python Zero-Based Indexing', ha='center', va='center',
        fontsize=16, weight='bold')

# List representation
list_text = 'features = [10, 20, 30, 40, 50]'
ax.text(6, 5, list_text, ha='center', va='center',
        fontsize=12, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

# Examples
example_y = 0.8
ax.text(2.5, example_y, 'features[0] = 10', ha='left', va='center',
        fontsize=11, family='monospace', color=color_blue)
ax.text(2.5, example_y - 0.4, 'features[-1] = 50', ha='left', va='center',
        fontsize=11, family='monospace', color=color_orange)

ax.text(6, example_y, 'features[2] = 30', ha='left', va='center',
        fontsize=11, family='monospace', color=color_blue)
ax.text(6, example_y - 0.4, 'features[-3] = 30', ha='left', va='center',
        fontsize=11, family='monospace', color=color_orange)

# Key insight
ax.text(6, 0, 'First element: index 0  •  Last element: index -1',
        ha='center', va='center', fontsize=11, style='italic', color=color_gray)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch05-python-fundamentals/diagrams/zero_based_indexing.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: zero_based_indexing.png")

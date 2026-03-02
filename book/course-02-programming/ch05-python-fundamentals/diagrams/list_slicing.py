"""
List Slicing Visualization
Shows how slicing syntax works with examples
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_blue = '#2196F3'
color_green = '#4CAF50'
color_orange = '#FF9800'
color_purple = '#9C27B0'
color_gray = '#607D8B'

# List values
values = [2.3, 1.8, 3.1, 2.7, 4.0, 1.5, 3.8]
n = len(values)

# Helper function to draw list
def draw_list(ax, y_pos, values, highlight_indices=None, highlight_color=color_blue):
    box_width = 1.2
    box_height = 0.8
    start_x = 1.5

    for i, val in enumerate(values):
        x = start_x + i * box_width
        # Determine if highlighted
        is_highlighted = highlight_indices and i in highlight_indices
        alpha = 0.4 if is_highlighted else 0.1
        color = highlight_color if is_highlighted else color_gray

        # Box
        rect = patches.FancyBboxPatch((x, y_pos-box_height/2), box_width*0.95, box_height,
                                       boxstyle="round,pad=0.05",
                                       edgecolor=color,
                                       facecolor=color,
                                       linewidth=2.5 if is_highlighted else 1.5,
                                       alpha=alpha)
        ax.add_patch(rect)
        # Value
        ax.text(x + box_width/2, y_pos, f'{val:.1f}',
                ha='center', va='center', fontsize=11,
                weight='bold' if is_highlighted else 'normal',
                color=color)
        # Index below
        ax.text(x + box_width/2, y_pos - 0.6, str(i),
                ha='center', va='center', fontsize=9, color=color_gray)

# Title
ax.text(6, 9.5, 'Python List Slicing: [start:end:step]', ha='center', va='center',
        fontsize=16, weight='bold')

# Original list
ax.text(0.5, 8.2, 'Original list:', ha='left', va='center',
        fontsize=11, weight='bold', color='black')
list_text = 'features = [2.3, 1.8, 3.1, 2.7, 4.0, 1.5, 3.8]'
ax.text(6, 7.8, list_text, ha='center', va='center',
        fontsize=10, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

# Example 1: features[1:4]
y1 = 6.8
ax.text(0.5, y1, 'features[1:4]', ha='left', va='center',
        fontsize=11, family='monospace', weight='bold', color=color_blue)
draw_list(ax, y1, values, highlight_indices=[1, 2, 3], highlight_color=color_blue)
ax.text(10.5, y1, '→  [1.8, 3.1, 2.7]', ha='left', va='center',
        fontsize=10, family='monospace', color=color_blue)
ax.text(10.5, y1-0.4, 'Indices 1, 2, 3\n(excludes 4)', ha='left', va='center',
        fontsize=8, style='italic', color=color_gray)

# Example 2: features[:3]
y2 = 5.3
ax.text(0.5, y2, 'features[:3]', ha='left', va='center',
        fontsize=11, family='monospace', weight='bold', color=color_green)
draw_list(ax, y2, values, highlight_indices=[0, 1, 2], highlight_color=color_green)
ax.text(10.5, y2, '→  [2.3, 1.8, 3.1]', ha='left', va='center',
        fontsize=10, family='monospace', color=color_green)
ax.text(10.5, y2-0.4, 'First 3 elements\n(start defaults to 0)', ha='left', va='center',
        fontsize=8, style='italic', color=color_gray)

# Example 3: features[4:]
y3 = 3.8
ax.text(0.5, y3, 'features[4:]', ha='left', va='center',
        fontsize=11, family='monospace', weight='bold', color=color_orange)
draw_list(ax, y3, values, highlight_indices=[4, 5, 6], highlight_color=color_orange)
ax.text(10.5, y3, '→  [4.0, 1.5, 3.8]', ha='left', va='center',
        fontsize=10, family='monospace', color=color_orange)
ax.text(10.5, y3-0.4, 'From index 4 to end\n(end defaults to len)', ha='left', va='center',
        fontsize=8, style='italic', color=color_gray)

# Example 4: features[::2]
y4 = 2.3
ax.text(0.5, y4, 'features[::2]', ha='left', va='center',
        fontsize=11, family='monospace', weight='bold', color=color_purple)
draw_list(ax, y4, values, highlight_indices=[0, 2, 4, 6], highlight_color=color_purple)
ax.text(10.5, y4, '→  [2.3, 3.1, 4.0, 3.8]', ha='left', va='center',
        fontsize=10, family='monospace', color=color_purple)
ax.text(10.5, y4-0.4, 'Every 2nd element\n(step = 2)', ha='left', va='center',
        fontsize=8, style='italic', color=color_gray)

# Key rule at bottom
rule_text = 'Key Rule: [start:end] includes start, excludes end  •  All parameters optional'
ax.text(6, 0.8, rule_text, ha='center', va='center',
        fontsize=11, weight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                 edgecolor=color_gray, linewidth=2))

# Common patterns
ax.text(6, 0.2, 'Common: features[:n] = first n  •  features[n:] = from n onward  •  features[-1] = last',
        ha='center', va='center', fontsize=9, style='italic', color=color_gray)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-02-programming/ch05-python-fundamentals/diagrams/list_slicing.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: list_slicing.png")

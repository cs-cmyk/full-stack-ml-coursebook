"""
Diagram 1: One-Hot Encoding Visual
Side-by-side transformation showing categorical to binary columns
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
gray = '#607D8B'
light_gray = '#E0E0E0'

# ============ LEFT SIDE: Original Data ============
# Title
ax.text(2.5, 5.5, 'BEFORE: Original Data', fontsize=14, weight='bold', ha='center')

# Original data table
original_data = [
    ['Sample', 'Color'],
    ['1', 'Red'],
    ['2', 'Blue'],
    ['3', 'Green'],
    ['4', 'Red'],
    ['5', 'Blue']
]

# Draw table background
table_left = 0.5
table_top = 5.0
col_width = 1.5
row_height = 0.6

# Header
for i, text in enumerate(original_data[0]):
    rect = FancyBboxPatch((table_left + i * col_width, table_top - row_height),
                          col_width, row_height,
                          boxstyle="round,pad=0.05",
                          edgecolor=blue, facecolor=blue, linewidth=2)
    ax.add_patch(rect)
    ax.text(table_left + i * col_width + col_width/2, table_top - row_height/2,
           text, ha='center', va='center', fontsize=11, weight='bold', color='white')

# Data rows
for row_idx, row in enumerate(original_data[1:], 1):
    for col_idx, text in enumerate(row):
        y_pos = table_top - (row_idx + 1) * row_height

        # Highlight categorical values
        if col_idx == 1:  # Color column
            if text == 'Red':
                color = '#FFCDD2'
            elif text == 'Blue':
                color = '#BBDEFB'
            else:
                color = '#C8E6C9'
        else:
            color = light_gray

        rect = FancyBboxPatch((table_left + col_idx * col_width, y_pos),
                              col_width, row_height,
                              boxstyle="round,pad=0.05",
                              edgecolor=gray, facecolor=color, linewidth=1)
        ax.add_patch(rect)
        ax.text(table_left + col_idx * col_width + col_width/2, y_pos + row_height/2,
               text, ha='center', va='center', fontsize=10)

# Annotation
ax.text(2.5, 0.8, '1 column with\n3 unique categories',
        ha='center', va='center', fontsize=10, style='italic', color=gray,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=gray, linewidth=1))

# ============ ARROW ============
arrow = FancyArrowPatch((4.5, 3.0), (6.5, 3.0),
                       arrowstyle='->', mutation_scale=30,
                       linewidth=3, color=orange)
ax.add_patch(arrow)
ax.text(5.5, 3.5, 'One-Hot\nEncode', ha='center', va='center',
       fontsize=11, weight='bold', color=orange)

# ============ RIGHT SIDE: Encoded Data ============
# Title
ax.text(10, 5.5, 'AFTER: One-Hot Encoded', fontsize=14, weight='bold', ha='center')

# Encoded data table
encoded_data = [
    ['Sample', 'is_Red', 'is_Blue', 'is_Green'],
    ['1', '1', '0', '0'],
    ['2', '0', '1', '0'],
    ['3', '0', '0', '1'],
    ['4', '1', '0', '0'],
    ['5', '0', '1', '0']
]

# Draw encoded table
table_left2 = 7.5
col_width2 = 1.2

# Header
for i, text in enumerate(encoded_data[0]):
    rect = FancyBboxPatch((table_left2 + i * col_width2, table_top - row_height),
                          col_width2, row_height,
                          boxstyle="round,pad=0.05",
                          edgecolor=green, facecolor=green, linewidth=2)
    ax.add_patch(rect)
    ax.text(table_left2 + i * col_width2 + col_width2/2, table_top - row_height/2,
           text, ha='center', va='center', fontsize=9, weight='bold', color='white')

# Data rows with highlighting
for row_idx, row in enumerate(encoded_data[1:], 1):
    for col_idx, text in enumerate(row):
        y_pos = table_top - (row_idx + 1) * row_height

        # Highlight 1s in green
        if col_idx > 0 and text == '1':
            color = '#A5D6A7'
        elif col_idx > 0:
            color = light_gray
        else:
            color = light_gray

        rect = FancyBboxPatch((table_left2 + col_idx * col_width2, y_pos),
                              col_width2, row_height,
                              boxstyle="round,pad=0.05",
                              edgecolor=gray, facecolor=color, linewidth=1)
        ax.add_patch(rect)
        ax.text(table_left2 + col_idx * col_width2 + col_width2/2, y_pos + row_height/2,
               text, ha='center', va='center', fontsize=10,
               weight='bold' if text == '1' else 'normal')

# Highlight example: Row 1 (Red)
ax.annotate('', xy=(table_left2 + 0.5 * col_width2, table_top - 2 * row_height - row_height/2),
           xytext=(table_left2 - 0.5, table_top - 2 * row_height - row_height/2),
           arrowprops=dict(arrowstyle='->', color=orange, lw=2))
ax.text(table_left2 - 1.2, table_top - 2 * row_height - row_height/2,
       'Red → [1, 0, 0]', fontsize=9, color=orange, weight='bold',
       ha='right', va='center')

# Annotation
ax.text(10, 0.8, '3 binary columns\n(one per category)',
        ha='center', va='center', fontsize=10, style='italic', color=gray,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=gray, linewidth=1))

# ============ KEY INSIGHT BOX ============
insight_box = FancyBboxPatch((0.5, -0.5), 13, 0.7,
                            boxstyle="round,pad=0.1",
                            edgecolor=blue, facecolor='#E3F2FD', linewidth=2)
ax.add_patch(insight_box)
ax.text(7, 0.15, '✓ Each category becomes its own binary indicator column',
       ha='center', va='center', fontsize=11, weight='bold', color=blue)
ax.text(7, -0.15, 'Only ONE column is "1" per row (mutually exclusive categories)',
       ha='center', va='center', fontsize=10, color=blue)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-03-eda-features/ch14-categorical-features/diagrams/01_one_hot_encoding_visual.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Diagram 1 saved: 01_one_hot_encoding_visual.png")

"""
Diagram 5: Categorical Encoding Decision Tree
Decision flowchart for choosing the right encoding strategy
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
purple = '#9C27B0'
gray = '#607D8B'
yellow = '#FFC107'

def draw_box(ax, x, y, width, height, text, bg_color, edge_color, text_color='black', fontsize=10, bold=True):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor=edge_color, facecolor=bg_color, linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
           weight=weight, color=text_color, multialignment='center')

def draw_diamond(ax, x, y, width, height, text, bg_color, edge_color, fontsize=9):
    """Draw a diamond (decision node) with text"""
    diamond = Polygon([
        (x, y + height/2),  # top
        (x + width/2, y),   # right
        (x, y - height/2),  # bottom
        (x - width/2, y)    # left
    ], edgecolor=edge_color, facecolor=bg_color, linewidth=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
           weight='bold', color='black', multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, label='', color=gray, style='-', lw=2):
    """Draw an arrow with optional label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=lw, color=color, linestyle=style)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, ha='center', va='bottom',
               fontsize=8, weight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, linewidth=1))

# ============ TITLE ============
draw_box(ax, 7, 11.3, 10, 0.6, 'Categorical Encoding Decision Tree',
        blue, blue, 'white', fontsize=14)

# ============ START NODE ============
draw_box(ax, 7, 10, 3, 0.6, 'Categorical Feature',
        '#E0E0E0', gray, fontsize=11)

# Arrow down
draw_arrow(ax, 7, 9.7, 7, 9.2)

# ============ DECISION 1: Binary? ============
draw_diamond(ax, 7, 8.7, 2.5, 0.8, 'Only 2\ncategories?', '#FFF9C4', yellow)

# YES branch (right)
draw_arrow(ax, 8.3, 8.7, 10, 8.7, 'YES', green)
draw_box(ax, 11.5, 8.7, 2.5, 0.7, 'Binary Encoding\n(0/1 single column)',
        '#C8E6C9', green, fontsize=9)
ax.text(11.5, 8.1, 'Example: Yes/No,\nMale/Female', ha='center', va='top',
       fontsize=7, style='italic', color=gray)

# NO branch (down)
draw_arrow(ax, 7, 8.2, 7, 7.7, 'NO', gray)

# ============ DECISION 2: Inherent Order? ============
draw_diamond(ax, 7, 7.2, 2.8, 0.8, 'Inherent\norder?', '#E1BEE7', purple)

# YES branch (right) - Ordinal
draw_arrow(ax, 8.5, 7.2, 10.5, 7.2, 'YES', green)
draw_box(ax, 12, 7.2, 2.8, 0.8, 'Ordinal Encoding\n(preserve order: 1,2,3...)',
        '#C8E6C9', green, fontsize=9)
ax.text(12, 6.5, 'Example: Small < Med < Large,\nHS < Bachelor < Master < PhD',
       ha='center', va='top', fontsize=7, style='italic', color=gray)

# NO branch (down) - Nominal
draw_arrow(ax, 7, 6.7, 7, 6.2, 'NO\n(Nominal)', gray)

# Label: Nominal path
draw_box(ax, 7, 5.8, 2.2, 0.4, 'Nominal Variable',
        '#FFE0B2', orange, fontsize=9)

# Arrow down
draw_arrow(ax, 7, 5.6, 7, 5.1)

# ============ DECISION 3: Cardinality ============
draw_diamond(ax, 7, 4.6, 2.5, 0.8, 'Cardinality\n< 15?', '#BBDEFB', blue)

# YES branch (left) - Low cardinality
draw_arrow(ax, 5.8, 4.6, 3.5, 4.6, 'YES', green)
draw_box(ax, 2, 4.6, 2.5, 0.8, 'One-Hot Encoding\n(K or K-1 binary columns)',
        '#C8E6C9', green, fontsize=9)
ax.text(2, 3.9, 'Best for: Linear models,\nNN, SVM (distance-based)',
       ha='center', va='top', fontsize=7, style='italic', color=gray)

# NO branch (down) - High cardinality
draw_arrow(ax, 7, 4.1, 7, 3.6, 'NO\n(High Card.)', orange)

# Label: High cardinality path
draw_box(ax, 7, 3.2, 2.8, 0.4, 'High Cardinality\n(50+ categories)',
        '#FFCDD2', red, fontsize=9)

# Arrow down splits into 3
draw_arrow(ax, 7, 3.0, 7, 2.6)

# Split into 3 strategies
x_positions = [2.5, 7, 11.5]
strategies = [
    ('Frequency\nEncoding', 'Replace with\ncount/percentage', '#E1BEE7', purple),
    ('Grouped\nTop-N + Other', 'Keep top N,\ngroup rest as "Other"', '#FFE0B2', orange),
    ('Target\nEncoding', 'Replace with\nmean target value', '#FFCDD2', red)
]

for i, (x_pos, (title, desc, bg, edge)) in enumerate(zip(x_positions, strategies)):
    # Arrow to strategy
    if i == 0:
        draw_arrow(ax, 6.5, 2.5, x_pos, 2.1, '', gray, '-', 1.5)
    elif i == 1:
        draw_arrow(ax, 7, 2.4, x_pos, 2.1, '', gray, '-', 1.5)
    else:
        draw_arrow(ax, 7.5, 2.5, x_pos, 2.1, '', gray, '-', 1.5)

    # Strategy box
    draw_box(ax, x_pos, 1.6, 2.2, 0.7, title, bg, edge, fontsize=9)
    ax.text(x_pos, 1.15, desc, ha='center', va='top',
           fontsize=7, style='italic', color=gray, multialignment='center')

# ============ SPECIAL CASE: Tree-Based Models ============
y_special = 0.3
special_box = FancyBboxPatch((0.5, y_special - 0.5), 13, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor=purple, facecolor='#F3E5F5', linewidth=2)
ax.add_patch(special_box)

ax.text(7, y_special + 0.15, '🌲 Special Case: Tree-Based Models (Random Forest, XGBoost, Decision Trees)',
       ha='center', va='center', fontsize=10, weight='bold', color=purple)
ax.text(7, y_special - 0.15, 'Label encoding acceptable for nominal + high-cardinality features (trees split on thresholds, not distances)',
       ha='center', va='center', fontsize=8, color=purple, style='italic')

# ============ KEY QUESTIONS BOX ============
y_key = -0.8
key_box = FancyBboxPatch((0.5, y_key - 0.8), 6, 1.3,
                        boxstyle="round,pad=0.1",
                        edgecolor=blue, facecolor='#E3F2FD', linewidth=2)
ax.add_patch(key_box)

ax.text(3.5, y_key + 0.45, 'Key Questions to Ask:',
       ha='center', va='center', fontsize=11, weight='bold', color=blue)

questions = [
    "1. Binary? → Use 0/1",
    "2. Inherent order? → Ordinal encoding",
    "3. Low cardinality (<15)? → One-hot",
    "4. High cardinality? → Frequency/Grouping/Target",
    "5. Tree-based model? → Label encoding OK"
]

for i, question in enumerate(questions):
    ax.text(3.5, y_key + 0.15 - i * 0.25, question,
           ha='center', va='center', fontsize=8, color=gray)

# ============ COMPARISON TABLE ============
y_table = -0.8
table_box = FancyBboxPatch((7, y_table - 0.8), 6.5, 1.3,
                          boxstyle="round,pad=0.1",
                          edgecolor=green, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(table_box)

ax.text(10.25, y_table + 0.45, 'Quick Reference:',
       ha='center', va='center', fontsize=11, weight='bold', color=green)

reference = [
    "✓ Nominal + Low card → One-Hot",
    "✓ Ordinal → Ordinal Encoding",
    "✓ High card + Linear → Frequency/Grouped",
    "✓ High card + Trees → Label/Target OK",
    "⚠ Never: Label encoding nominal with linear!"
]

for i, ref in enumerate(reference):
    color = red if '⚠' in ref else gray
    ax.text(10.25, y_table + 0.15 - i * 0.25, ref,
           ha='center', va='center', fontsize=8, color=color,
           weight='bold' if '⚠' in ref else 'normal')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-03-eda-features/ch14-categorical-features/diagrams/05_encoding_decision_tree.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Diagram 5 saved: 05_encoding_decision_tree.png")

"""
Diagram 2: Ordinal vs Nominal Encoding Comparison
Shows when label encoding is wrong vs. when it's appropriate
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
gray = '#607D8B'
light_gray = '#E0E0E0'

# ============ TOP HALF: NOMINAL (Wrong vs Right) ============
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 6)
ax1.axis('off')

# Title
title_box = FancyBboxPatch((0.5, 5.2), 13, 0.6,
                           boxstyle="round,pad=0.1",
                           edgecolor=blue, facecolor=blue, linewidth=2)
ax1.add_patch(title_box)
ax1.text(7, 5.5, 'NOMINAL VARIABLES (No Inherent Order)',
        ha='center', va='center', fontsize=14, weight='bold', color='white')

# Example: Colors
ax1.text(7, 4.7, 'Example: Color (Red, Blue, Green)',
        ha='center', va='top', fontsize=12, weight='bold', color=gray)

# ---- WRONG WAY ----
ax1.text(3, 4.0, '❌ WRONG: Label Encoding',
        ha='center', va='center', fontsize=12, weight='bold', color=red)

# Show label encoding
categories_wrong = ['Red', 'Blue', 'Green']
values_wrong = [1, 2, 3]
colors_cat = ['#EF5350', '#42A5F5', '#66BB6A']

y_pos = 3.3
for i, (cat, val, col) in enumerate(zip(categories_wrong, values_wrong, colors_cat)):
    x_pos = 1.5 + i * 1.2
    # Category box
    rect = FancyBboxPatch((x_pos - 0.4, y_pos - 0.2), 0.8, 0.4,
                          boxstyle="round,pad=0.05",
                          edgecolor=gray, facecolor=col, linewidth=2, alpha=0.7)
    ax1.add_patch(rect)
    ax1.text(x_pos, y_pos, cat, ha='center', va='center', fontsize=10, weight='bold')

    # Arrow
    ax1.annotate('', xy=(x_pos, y_pos - 0.5), xytext=(x_pos, y_pos - 0.25),
                arrowprops=dict(arrowstyle='->', color=gray, lw=2))

    # Encoded value
    circle = Circle((x_pos, y_pos - 0.75), 0.25, edgecolor=red, facecolor=light_gray, linewidth=2)
    ax1.add_patch(circle)
    ax1.text(x_pos, y_pos - 0.75, str(val), ha='center', va='center',
            fontsize=12, weight='bold', color=red)

# Problems
problems = [
    "2 > 1 implies Blue > Red?",
    "Green - Red = 2?",
    "(Red + Green)/2 = Blue?"
]
y_problem = 1.8
for i, problem in enumerate(problems):
    ax1.text(3, y_problem - i * 0.35, f"• {problem}",
            ha='center', va='center', fontsize=9, color=red, style='italic')

# ---- RIGHT WAY ----
ax1.text(10.5, 4.0, '✓ CORRECT: One-Hot Encoding',
        ha='center', va='center', fontsize=12, weight='bold', color=green)

# Show one-hot encoding
y_pos2 = 3.3
categories = ['Red', 'Blue', 'Green']
for i, (cat, col) in enumerate(zip(categories, colors_cat)):
    x_pos = 8.5 + i * 1.8

    # Category
    rect = FancyBboxPatch((x_pos - 0.35, y_pos2 - 0.15), 0.7, 0.3,
                          boxstyle="round,pad=0.05",
                          edgecolor=gray, facecolor=col, linewidth=2, alpha=0.7)
    ax1.add_patch(rect)
    ax1.text(x_pos, y_pos2, cat, ha='center', va='center', fontsize=9, weight='bold')

    # Arrow
    ax1.annotate('', xy=(x_pos, y_pos2 - 0.5), xytext=(x_pos, y_pos2 - 0.2),
                arrowprops=dict(arrowstyle='->', color=gray, lw=2))

    # One-hot vector
    vector_y = y_pos2 - 1.0
    for j in range(3):
        val = 1 if j == i else 0
        box_color = '#A5D6A7' if val == 1 else light_gray
        rect = FancyBboxPatch((x_pos - 0.35 + j * 0.24, vector_y - 0.12), 0.2, 0.24,
                              boxstyle="round,pad=0.02",
                              edgecolor=gray, facecolor=box_color, linewidth=1)
        ax1.add_patch(rect)
        ax1.text(x_pos - 0.35 + j * 0.24 + 0.1, vector_y, str(val),
                ha='center', va='center', fontsize=9, weight='bold' if val == 1 else 'normal')

# Benefits
benefits = [
    "No false ordering",
    "Each category independent",
    "Works with all models"
]
y_benefit = 1.8
for i, benefit in enumerate(benefits):
    ax1.text(10.5, y_benefit - i * 0.35, f"✓ {benefit}",
            ha='center', va='center', fontsize=9, color=green, weight='bold')

# Divider
ax1.plot([0.5, 13.5], [0.8, 0.8], color=gray, linewidth=2, linestyle='--')

# Key insight
ax1.text(7, 0.4, 'Key: Nominal categories have NO mathematical relationship → One-hot encode',
        ha='center', va='center', fontsize=10, weight='bold', color=blue,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor=blue, linewidth=2))

# ============ BOTTOM HALF: ORDINAL (Label Encoding is OK) ============
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 6)
ax2.axis('off')

# Title
title_box2 = FancyBboxPatch((0.5, 5.2), 13, 0.6,
                            boxstyle="round,pad=0.1",
                            edgecolor=green, facecolor=green, linewidth=2)
ax2.add_patch(title_box2)
ax2.text(7, 5.5, 'ORDINAL VARIABLES (Meaningful Order)',
        ha='center', va='center', fontsize=14, weight='bold', color='white')

# Example: T-shirt sizes
ax2.text(7, 4.7, 'Example: T-Shirt Size (Small, Medium, Large, XL)',
        ha='center', va='top', fontsize=12, weight='bold', color=gray)

# Show ordinal encoding
ax2.text(7, 4.0, '✓ CORRECT: Ordinal Encoding (Preserves Order)',
        ha='center', va='center', fontsize=12, weight='bold', color=green)

# Categories with meaningful order
sizes = ['Small', 'Medium', 'Large', 'XL']
size_values = [1, 2, 3, 4]
size_colors = ['#FFF9C4', '#FFE082', '#FFB74D', '#FF9800']

y_base = 3.0
for i, (size, val, col) in enumerate(zip(sizes, size_values, size_colors)):
    x_pos = 2 + i * 2.8

    # Size box
    rect = FancyBboxPatch((x_pos - 0.5, y_base - 0.2), 1.0, 0.4,
                          boxstyle="round,pad=0.05",
                          edgecolor=gray, facecolor=col, linewidth=2)
    ax2.add_patch(rect)
    ax2.text(x_pos, y_base, size, ha='center', va='center', fontsize=10, weight='bold')

    # Arrow
    ax2.annotate('', xy=(x_pos, y_base - 0.5), xytext=(x_pos, y_base - 0.25),
                arrowprops=dict(arrowstyle='->', color=green, lw=2))

    # Encoded value
    circle = Circle((x_pos, y_base - 0.75), 0.25, edgecolor=green, facecolor=col, linewidth=2)
    ax2.add_patch(circle)
    ax2.text(x_pos, y_base - 0.75, str(val), ha='center', va='center',
            fontsize=12, weight='bold', color=gray)

# Show ordering is meaningful
arrow_order = FancyArrowPatch((2.5, 1.5), (11.5, 1.5),
                             arrowstyle='<->', mutation_scale=20,
                             linewidth=2, color=green)
ax2.add_patch(arrow_order)
ax2.text(7, 1.8, 'Increasing Size →', ha='center', va='bottom',
        fontsize=10, weight='bold', color=green)
ax2.text(7, 1.2, '(Order is meaningful!)', ha='center', va='top',
        fontsize=9, style='italic', color=green)

# Why it works
reasons = [
    "✓ Small < Medium < Large < XL (true relationship)",
    "✓ Distance matters: Large (3) is between Medium (2) and XL (4)",
    "✓ Model can learn: 'larger sizes predict higher values'",
    "✓ Preserves information with just 1 column (not 4)"
]
y_reason = 0.6
for i, reason in enumerate(reasons):
    ax2.text(7, y_reason - i * 0.25, reason,
            ha='center', va='center', fontsize=9, color=gray)

# ============ FINAL INSIGHT ============
fig.text(0.5, 0.02,
        'Decision Rule: Can you meaningfully AVERAGE two categories? → If YES, ordinal (use label encoding) | If NO, nominal (use one-hot)',
        ha='center', va='center', fontsize=11, weight='bold', color=blue,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor=blue, linewidth=2))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('/home/chirag/ds-book/book/course-03-eda-features/ch14-categorical-features/diagrams/02_ordinal_vs_nominal.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Diagram 2 saved: 02_ordinal_vs_nominal.png")

"""
Diagram 3: High-Cardinality Problem Visualization
Comparison of different encoding strategies for high-cardinality features
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
purple = '#9C27B0'
gray = '#607D8B'

# ============ LEFT PLOT: Number of Features ============
strategies = ['One-Hot\nEncoding', 'Frequency\nEncoding', 'Grouped\nTop-20', 'Target\nEncoding']
n_features = [500, 1, 21, 1]
colors_strategies = [red, green, orange, purple]

bars = ax1.bar(strategies, n_features, color=colors_strategies, alpha=0.8, edgecolor=gray, linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, n_features):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{val}',
            ha='center', va='bottom', fontsize=12, weight='bold')

ax1.set_ylabel('Number of Features Created', fontsize=12, weight='bold')
ax1.set_xlabel('Encoding Strategy', fontsize=12, weight='bold')
ax1.set_title('Feature Dimensionality Comparison\n(50 unique categories)', fontsize=13, weight='bold')
ax1.set_ylim(0, 550)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotation for one-hot problem
ax1.annotate('Curse of\nDimensionality!',
            xy=(0, 500), xytext=(-0.3, 400),
            arrowprops=dict(arrowstyle='->', color=red, lw=2),
            fontsize=10, weight='bold', color=red,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=red, linewidth=2))

# Highlight the sweet spot
ax1.annotate('Good balance:\nModest features\n+ High performance',
            xy=(2, 21), xytext=(2.5, 150),
            arrowprops=dict(arrowstyle='->', color=orange, lw=2),
            fontsize=9, weight='bold', color=orange,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=orange, linewidth=2))

# ============ RIGHT PLOT: Performance vs Training Time ============
# Simulated data based on realistic expectations
performance_r2 = [0.72, 0.68, 0.75, 0.78]  # R² scores
training_time = [8.5, 0.3, 1.2, 0.5]  # seconds

# Create scatter plot
for i, (strategy, r2, time, color) in enumerate(zip(strategies, performance_r2, training_time, colors_strategies)):
    # Size proportional to features (inverse for visibility)
    size = 3000 / (n_features[i] + 1)
    ax2.scatter(time, r2, s=size, color=color, alpha=0.7, edgecolors=gray, linewidth=2, label=strategy)

    # Add labels
    offset_x = 0.3 if i != 0 else 1.5
    offset_y = 0.01 if i % 2 == 0 else -0.01
    ax2.annotate(strategy.replace('\n', ' '),
                xy=(time, r2), xytext=(time + offset_x, r2 + offset_y),
                fontsize=9, weight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1.5))

ax2.set_xlabel('Training Time (seconds)', fontsize=12, weight='bold')
ax2.set_ylabel('Model Performance (R² Score)', fontsize=12, weight='bold')
ax2.set_title('Performance vs. Training Time Trade-off', fontsize=13, weight='bold')
ax2.set_xlim(-0.5, 10)
ax2.set_ylim(0.65, 0.80)
ax2.grid(True, alpha=0.3, linestyle='--')

# Add optimal region
from matplotlib.patches import Rectangle
optimal_region = Rectangle((0, 0.73), 2, 0.06,
                          alpha=0.15, facecolor=green, edgecolor=green, linewidth=2, linestyle='--')
ax2.add_patch(optimal_region)
ax2.text(1, 0.793, 'Optimal Zone:\nHigh R², Fast Training',
        ha='center', va='bottom', fontsize=9, style='italic', color=green, weight='bold')

# ============ COMPARISON TABLE (Bottom) ============
fig.text(0.5, -0.02, 'Strategy Comparison Summary', ha='center', fontsize=13, weight='bold')

table_data = [
    ['Strategy', 'Features', 'Training Time', 'R² Score', 'Best Use Case'],
    ['One-Hot', '500', 'Slow (8.5s)', '0.72', 'Low cardinality only (<15 cats)'],
    ['Frequency', '1', 'Very Fast (0.3s)', '0.68', 'When frequency is predictive'],
    ['Grouped Top-20', '21', 'Fast (1.2s)', '0.75', 'Most common categories matter'],
    ['Target', '1', 'Very Fast (0.5s)', '0.78', 'Strong target correlation (careful: leakage!)']
]

# Create table
table_y = -0.10
cell_height = 0.025
row_widths = [0.18, 0.12, 0.18, 0.12, 0.40]
colors_table = ['white', '#FFCDD2', '#C8E6C9', '#FFE0B2', '#E1BEE7']

for row_idx, row in enumerate(table_data):
    x_start = 0.05
    for col_idx, (cell, width) in enumerate(zip(row, row_widths)):
        # Header row
        if row_idx == 0:
            bg_color = gray
            text_color = 'white'
            text_weight = 'bold'
        else:
            bg_color = colors_table[row_idx]
            text_color = 'black'
            text_weight = 'normal'

        # Draw cell
        fig.add_artist(plt.Rectangle((x_start, table_y - row_idx * cell_height),
                                     width, cell_height,
                                     facecolor=bg_color, edgecolor=gray, linewidth=1,
                                     transform=fig.transFigure, clip_on=False))

        # Add text
        fig.text(x_start + width/2, table_y - row_idx * cell_height + cell_height/2,
                cell, ha='center', va='center', fontsize=8,
                weight=text_weight, color=text_color, transform=fig.transFigure)

        x_start += width

plt.tight_layout(rect=[0, 0.08, 1, 0.98])
plt.savefig('/home/chirag/ds-book/book/course-03-eda-features/ch14-categorical-features/diagrams/03_high_cardinality_comparison.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Diagram 3 saved: 03_high_cardinality_comparison.png")

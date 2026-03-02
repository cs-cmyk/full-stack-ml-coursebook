"""
LSTM architecture diagram
Visualizes the LSTM cell structure with gates
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'LSTM Cell Architecture', fontsize=18, fontweight='bold', ha='center')
ax.text(5, 9, 'Long Short-Term Memory for Time Series', fontsize=12, ha='center', style='italic', color='#555')

# Color scheme
color_forget = '#F44336'
color_input = '#4CAF50'
color_output = '#2196F3'
color_cell = '#FF9800'
color_tanh = '#9C27B0'

# Cell state line (horizontal)
ax.plot([0.5, 9.5], [6.5, 6.5], 'k-', linewidth=3, alpha=0.3)
ax.text(0.5, 7, 'Cell State (Ct-1)', fontsize=10, ha='right', fontweight='bold')
ax.text(9.5, 7, 'Cell State (Ct)', fontsize=10, ha='left', fontweight='bold')

# Inputs at bottom
ax.annotate('', xy=(2, 2), xytext=(2, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(2, 0.2, 'xt\n(input)', fontsize=10, ha='center', fontweight='bold')

ax.annotate('', xy=(3.5, 2), xytext=(3.5, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(3.5, 0.2, 'ht-1\n(prev hidden)', fontsize=10, ha='center', fontweight='bold')

# Forget Gate
forget_box = FancyBboxPatch((1.5, 2), 1.5, 1, boxstyle="round,pad=0.1",
                            edgecolor=color_forget, facecolor=color_forget, alpha=0.3, linewidth=2)
ax.add_patch(forget_box)
ax.text(2.25, 2.5, 'σ', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(2.25, 1.8, 'Forget Gate', fontsize=9, ha='center', fontweight='bold')
ax.text(2.25, 3.2, 'ft = σ(Wf·[ht-1, xt] + bf)', fontsize=8, ha='center', family='monospace')

# Arrow from forget gate to cell state
ax.annotate('', xy=(2.25, 6.5), xytext=(2.25, 3.3),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_forget))
ax.plot([2.25, 2.25], [6.3, 6.7], 'o', color=color_forget, markersize=12)
ax.text(1.8, 5, '×', fontsize=14, fontweight='bold', color=color_forget)

# Input Gate
input_box = FancyBboxPatch((4, 2), 1.5, 1, boxstyle="round,pad=0.1",
                          edgecolor=color_input, facecolor=color_input, alpha=0.3, linewidth=2)
ax.add_patch(input_box)
ax.text(4.75, 2.5, 'σ', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(4.75, 1.8, 'Input Gate', fontsize=9, ha='center', fontweight='bold')
ax.text(4.75, 3.2, 'it = σ(Wi·[ht-1, xt] + bi)', fontsize=8, ha='center', family='monospace')

# Candidate Cell State
candidate_box = FancyBboxPatch((6, 2), 1.5, 1, boxstyle="round,pad=0.1",
                              edgecolor=color_tanh, facecolor=color_tanh, alpha=0.3, linewidth=2)
ax.add_patch(candidate_box)
ax.text(6.75, 2.5, 'tanh', fontsize=13, ha='center', va='center', fontweight='bold')
ax.text(6.75, 1.8, 'Candidate', fontsize=9, ha='center', fontweight='bold')
ax.text(6.75, 3.2, 'C̃t = tanh(WC·[ht-1, xt] + bC)', fontsize=8, ha='center', family='monospace')

# Arrows from input gate and candidate to cell state
ax.annotate('', xy=(5.5, 6.5), xytext=(4.75, 3.3),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_input))
ax.annotate('', xy=(5.7, 6.5), xytext=(6.75, 3.3),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_tanh))
ax.plot([5.5, 5.7], [6.5, 6.5], 'o', color=color_input, markersize=12)
ax.text(5.5, 5, '×', fontsize=14, fontweight='bold', color=color_input)

# Cell State Update (addition)
ax.plot([3.5, 3.5], [6.3, 6.7], '+', color='black', markersize=16, markeredgewidth=3)

# Output Gate
output_box = FancyBboxPatch((7.5, 4), 1.5, 1, boxstyle="round,pad=0.1",
                           edgecolor=color_output, facecolor=color_output, alpha=0.3, linewidth=2)
ax.add_patch(output_box)
ax.text(8.25, 4.5, 'σ', fontsize=16, ha='center', va='center', fontweight='bold')
ax.text(8.25, 3.8, 'Output Gate', fontsize=9, ha='center', fontweight='bold')
ax.text(8.25, 5.2, 'ot = σ(Wo·[ht-1, xt] + bo)', fontsize=8, ha='center', family='monospace')

# Arrow from cell state to output
ax.annotate('', xy=(7.5, 6.5), xytext=(7, 6.5),
            arrowprops=dict(arrowstyle='->', lw=2, color=color_cell))
ax.text(7.3, 6.8, 'tanh', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=color_tanh, linewidth=1.5))

# Output hidden state
ax.annotate('', xy=(8.25, 8), xytext=(8.25, 6),
            arrowprops=dict(arrowstyle='->', lw=2.5, color=color_output))
ax.text(8.25, 8.5, 'ht = ot ⊙ tanh(Ct)', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=color_output, alpha=0.2, edgecolor=color_output, linewidth=2))

# Legend
legend_y = 0.8
ax.text(0.5, legend_y + 0.3, 'Key Operations:', fontsize=11, fontweight='bold')
ax.text(0.5, legend_y, '• σ: Sigmoid (0 to 1) - controls flow', fontsize=9)
ax.text(0.5, legend_y - 0.3, '• tanh: Hyperbolic tangent (-1 to 1)', fontsize=9)
ax.text(0.5, legend_y - 0.6, '• ×: Element-wise multiplication', fontsize=9)
ax.text(0.5, legend_y - 0.9, '• +: Element-wise addition', fontsize=9)
ax.text(0.5, legend_y - 1.2, '• ⊙: Hadamard product', fontsize=9)

# Add explanatory box
explanation = (
    "Memory Flow:\n"
    "1. Forget: Decide what to remove from memory\n"
    "2. Input: Decide what new info to store\n"
    "3. Update: Modify cell state\n"
    "4. Output: Decide what to output based on memory"
)
ax.text(9.8, 4, explanation, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9, edgecolor='#CCCCCC', linewidth=1.5))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-07-time-series/ch30-advanced-ts/diagrams/lstm_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: lstm_architecture.png")
plt.close()

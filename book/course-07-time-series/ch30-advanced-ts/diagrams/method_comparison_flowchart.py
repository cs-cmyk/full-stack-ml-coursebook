"""
Decision flowchart for selecting the right forecasting method
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.7, 'Forecasting Method Selection Flowchart', fontsize=17, fontweight='bold', ha='center')

# Helper function for boxes
def add_box(ax, x, y, w, h, text, color, shape='round'):
    if shape == 'round':
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                            edgecolor=color, facecolor=color, alpha=0.3, linewidth=2.5)
    else:  # diamond for decision
        box = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                                      edgecolor=color, facecolor=color, alpha=0.2, linewidth=2)
    ax.add_patch(box)

    # Split text into lines for better formatting
    if '\n' in text:
        for i, line in enumerate(text.split('\n')):
            offset = (len(text.split('\n')) - 1) * 0.08
            ax.text(x, y + offset - i*0.16, line, fontsize=9.5, ha='center', va='center', fontweight='bold')
    else:
        ax.text(x, y, text, fontsize=10, ha='center', va='center', fontweight='bold')

def add_arrow(ax, x1, y1, x2, y2, label='', color='black'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color=color))
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x + 0.2, mid_y, label, fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

# Start node
add_box(ax, 5, 9, 2, 0.5, 'Time Series\nForecasting Problem', '#333333', 'round')

# First decision: Data size
add_arrow(ax, 5, 8.75, 5, 8.3)
add_box(ax, 5, 8, 2, 0.5, 'Sample Size?', '#607D8B')
ax.text(5, 7.85, '(n observations)', fontsize=8, ha='center', style='italic', color='#555')

# Small data branch (< 100)
add_arrow(ax, 4.2, 7.75, 2, 7.3, '<100', '#F44336')
add_box(ax, 2, 7, 1.5, 0.5, 'Simple\nBaselines', '#F44336', 'round')
ax.text(2, 6.6, 'Naive, Seasonal Naive,\nExp. Smoothing', fontsize=7.5, ha='center', style='italic')

# Medium data branch (100-10k)
add_arrow(ax, 5, 7.75, 5, 7.3, '100-10k')
add_box(ax, 5, 7, 1.8, 0.5, 'Interpretability\nRequired?', '#FF9800')

# High interpretability branch
add_arrow(ax, 4.2, 6.75, 3.2, 6.3, 'HIGH', '#4CAF50')
add_box(ax, 3.2, 6, 1.5, 0.5, 'Multiple\nSeasonalities?', '#4CAF50')

add_arrow(ax, 3.2, 5.75, 2.5, 5.3, 'Yes')
add_box(ax, 2.5, 5, 1.3, 0.45, 'Prophet', '#2196F3', 'round')
ax.text(2.5, 4.6, '✓ Trend+Seasons\n✓ Holidays\n✓ Interpretable', fontsize=7, ha='center', style='italic')

add_arrow(ax, 3.2, 5.75, 3.9, 5.3, 'No')
add_box(ax, 3.9, 5, 1.3, 0.45, 'SARIMA', '#2196F3', 'round')
ax.text(3.9, 4.6, '✓ Statistical\n✓ Single season\n✓ Small-medium', fontsize=7, ha='center', style='italic')

# Medium interpretability branch
add_arrow(ax, 5.9, 6.75, 6.8, 6.3, 'MED', '#FF9800')
add_box(ax, 6.8, 6, 1.5, 0.5, 'Many External\nVariables?', '#FF9800')

add_arrow(ax, 6.8, 5.75, 6, 5.3, 'Yes')
add_box(ax, 6, 5, 1.3, 0.45, 'XGBoost', '#4CAF50', 'round')
ax.text(6, 4.6, '✓ Features\n✓ Non-linear\n✓ Fast', fontsize=7, ha='center', style='italic')

add_arrow(ax, 6.8, 5.75, 7.6, 5.3, 'No')
add_box(ax, 7.6, 5, 1.3, 0.45, 'Prophet or\nXGBoost', '#4CAF50', 'round')
ax.text(7.6, 4.6, 'Test both,\ncompare CV scores', fontsize=7, ha='center', style='italic')

# Large data branch (> 10k)
add_arrow(ax, 5.8, 7.75, 7.5, 7.3, '>10k', '#9C27B0')
add_box(ax, 7.5, 7, 1.8, 0.5, 'Complex\nSequences?', '#9C27B0')

add_arrow(ax, 7.5, 6.75, 8.3, 6.3, 'Yes')
add_box(ax, 8.3, 6, 1.3, 0.45, 'LSTM/GRU', '#9C27B0', 'round')
ax.text(8.3, 5.6, '✓ Long sequences\n✓ Non-linear\n✓ Many vars', fontsize=7, ha='center', style='italic')

add_arrow(ax, 7.5, 6.75, 6.7, 6.3, 'No')
add_box(ax, 6.7, 6, 1.3, 0.45, 'XGBoost or\nLightGBM', '#FF9800', 'round')
ax.text(6.7, 5.6, '✓ Fast\n✓ Feature importance\n✓ Robust', fontsize=7, ha='center', style='italic')

# Special considerations box
special_box = mpatches.FancyBboxPatch((0.3, 0.5), 3.5, 3.2,
                                     edgecolor='#F44336', facecolor='#FFF3E0',
                                     alpha=0.7, linewidth=2, linestyle='--')
ax.add_patch(special_box)
ax.text(2.05, 3.5, 'Special Considerations', fontsize=11, ha='center', fontweight='bold', color='#F44336')

considerations = [
    "• Strong trend extrapolation:",
    "  → Use Prophet or SARIMA",
    "  → Trees cannot extrapolate!",
    "",
    "• Multiple seasonal periods:",
    "  → Prophet (e.g., daily+weekly+yearly)",
    "",
    "• Business stakeholder reporting:",
    "  → Prophet (interpretable components)",
    "",
    "• Real-time inference speed critical:",
    "  → XGBoost or simple baseline",
    "",
    "• Irregular time intervals:",
    "  → Tree models with engineered features"
]

y_start = 3.2
for i, line in enumerate(considerations):
    ax.text(0.5, y_start - i*0.18, line, fontsize=8, ha='left', va='top',
            family='monospace' if '→' in line else 'sans-serif')

# Validation reminder box
valid_box = mpatches.FancyBboxPatch((6, 0.5), 3.7, 1.5,
                                   edgecolor='#2196F3', facecolor='#E3F2FD',
                                   alpha=0.8, linewidth=2, linestyle='--')
ax.add_patch(valid_box)
ax.text(7.85, 1.85, 'Always Validate!', fontsize=11, ha='center', fontweight='bold', color='#2196F3')

validation = [
    "✓ Use walk-forward cross-validation",
    "✓ Compare to naive baselines",
    "✓ Report MAE/MAPE on test set",
    "✓ Check for data leakage",
    "✓ Test on multiple datasets"
]

y_valid = 1.6
for i, line in enumerate(validation):
    ax.text(6.2, y_valid - i*0.18, line, fontsize=8.5, ha='left', va='top')

# Bottom note
ax.text(5, 0.15, 'Decision is data-dependent: Always implement multiple methods and compare empirically!',
        fontsize=10, ha='center', style='italic', fontweight='bold', color='#333',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, edgecolor='#F57C00', linewidth=2))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-07-time-series/ch30-advanced-ts/diagrams/method_comparison_flowchart.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: method_comparison_flowchart.png")
plt.close()

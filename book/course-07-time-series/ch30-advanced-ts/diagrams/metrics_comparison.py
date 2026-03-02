"""
Comparison of evaluation metrics for time series forecasting
Shows characteristics and use cases for MAE, MAPE, SMAPE, RMSE
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time Series Evaluation Metrics Comparison', fontsize=16, fontweight='bold', y=0.98)

# Generate example data
np.random.seed(42)
y_true = np.array([100, 120, 110, 130, 125, 140, 135, 150, 145, 160])
y_pred_good = y_true + np.random.normal(0, 5, len(y_true))
y_pred_bad = y_true + np.random.normal(0, 15, len(y_true))

# Also test edge cases
y_true_with_zero = np.array([100, 50, 10, 5, 1, 0.1, 0])
y_pred_with_zero = y_true_with_zero + np.random.normal(0, 2, len(y_true_with_zero))

# Subplot 1: MAE (Mean Absolute Error)
ax1 = axes[0, 0]
mae_good = np.mean(np.abs(y_true - y_pred_good))
mae_bad = np.mean(np.abs(y_true - y_pred_bad))

ax1.plot(y_true, 'o-', color='black', linewidth=2, markersize=8, label='Actual', markerfacecolor='black')
ax1.plot(y_pred_good, 's--', color='#4CAF50', linewidth=2, markersize=6, label=f'Good (MAE={mae_good:.1f})')
ax1.plot(y_pred_bad, '^--', color='#F44336', linewidth=2, markersize=6, label=f'Bad (MAE={mae_bad:.1f})')

ax1.set_title('MAE: Mean Absolute Error', fontsize=13, fontweight='bold')
ax1.set_xlabel('Time', fontsize=11)
ax1.set_ylabel('Value', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(alpha=0.3)

# Add formula and properties
formula_text = (
    'Formula: MAE = (1/n) Σ|yᵢ - ŷᵢ|\n\n'
    '✓ Same units as target\n'
    '✓ Interpretable\n'
    '✓ Robust to outliers\n'
    '✓ Works with zeros\n'
    '⚠ Not scale-independent'
)
ax1.text(0.97, 0.5, formula_text, transform=ax1.transAxes, fontsize=8.5,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8, edgecolor='#4CAF50', linewidth=2))

# Subplot 2: MAPE (Mean Absolute Percentage Error)
ax2 = axes[0, 1]

# Calculate MAPE (avoiding zeros)
def calc_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_good = calc_mape(y_true, y_pred_good)
mape_bad = calc_mape(y_true, y_pred_bad)

ax2.plot(y_true, 'o-', color='black', linewidth=2, markersize=8, label='Actual', markerfacecolor='black')
ax2.plot(y_pred_good, 's--', color='#4CAF50', linewidth=2, markersize=6, label=f'Good (MAPE={mape_good:.1f}%)')
ax2.plot(y_pred_bad, '^--', color='#F44336', linewidth=2, markersize=6, label=f'Bad (MAPE={mape_bad:.1f}%)')

ax2.set_title('MAPE: Mean Absolute Percentage Error', fontsize=13, fontweight='bold')
ax2.set_xlabel('Time', fontsize=11)
ax2.set_ylabel('Value', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(alpha=0.3)

formula_text = (
    'Formula: MAPE = (100/n) Σ|yᵢ - ŷᵢ|/|yᵢ|\n\n'
    '✓ Scale-independent (%)\n'
    '✓ Easy to explain\n'
    '✗ Undefined at zero\n'
    '✗ Asymmetric (penalizes\n   over-forecasts more)\n'
    '✗ Infinite with zeros'
)
ax2.text(0.97, 0.5, formula_text, transform=ax2.transAxes, fontsize=8.5,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8, edgecolor='#FF9800', linewidth=2))

# Subplot 3: SMAPE (Symmetric MAPE)
ax3 = axes[1, 0]

def calc_smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

smape_good = calc_smape(y_true, y_pred_good)
smape_bad = calc_smape(y_true, y_pred_bad)

ax3.plot(y_true, 'o-', color='black', linewidth=2, markersize=8, label='Actual', markerfacecolor='black')
ax3.plot(y_pred_good, 's--', color='#4CAF50', linewidth=2, markersize=6, label=f'Good (SMAPE={smape_good:.1f}%)')
ax3.plot(y_pred_bad, '^--', color='#F44336', linewidth=2, markersize=6, label=f'Bad (SMAPE={smape_bad:.1f}%)')

ax3.set_title('SMAPE: Symmetric Mean Absolute Percentage Error', fontsize=13, fontweight='bold')
ax3.set_xlabel('Time', fontsize=11)
ax3.set_ylabel('Value', fontsize=11)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(alpha=0.3)

formula_text = (
    'Formula: SMAPE = (200/n) Σ|yᵢ - ŷᵢ|/(|yᵢ| + |ŷᵢ|)\n\n'
    '✓ More symmetric than MAPE\n'
    '✓ Bounded [0, 200%]\n'
    '✓ Scale-independent\n'
    '⚠ Still problematic near zero\n'
    '⚠ Can be unstable'
)
ax3.text(0.97, 0.5, formula_text, transform=ax3.transAxes, fontsize=8.5,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8, edgecolor='#9C27B0', linewidth=2))

# Subplot 4: Comparison table and guidelines
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

# Create comparison table
comparison_data = [
    ['Metric', 'Units', 'Outliers', 'Zeros', 'Scale-Indep', 'Use When'],
    ['MAE', 'Original', 'Robust', '✓ OK', '✗ No', 'Same scale data\nInterpretability needed'],
    ['RMSE', 'Original', 'Sensitive', '✓ OK', '✗ No', 'Penalize large errors\nGaussian errors'],
    ['MAPE', 'Percent', 'Moderate', '✗ Fails', '✓ Yes', 'Compare across scales\nNo zeros in data'],
    ['SMAPE', 'Percent', 'Moderate', '⚠ Poor', '✓ Yes', 'More symmetric errors\nFew near-zero values'],
]

# Draw table
col_widths = [0.12, 0.13, 0.15, 0.12, 0.15, 0.33]
row_height = 0.12
table_start_y = 0.85

for i, row in enumerate(comparison_data):
    y = table_start_y - i * row_height
    x = 0.05

    for j, cell in enumerate(row):
        # Header row
        if i == 0:
            color = '#2196F3'
            text_color = 'white'
            fontweight = 'bold'
            fontsize = 10
        else:
            color = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
            text_color = 'black'
            fontweight = 'normal'
            fontsize = 9

        rect = Rectangle((x, y - row_height), col_widths[j], row_height,
                        facecolor=color, edgecolor='#CCCCCC', linewidth=1)
        ax4.add_patch(rect)

        # Add text
        ax4.text(x + col_widths[j]/2, y - row_height/2, cell,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=text_color, wrap=True)

        x += col_widths[j]

# Add guidelines box
guidelines = (
    "📊 Recommendation Guidelines:\n\n"
    "1️⃣ Default Choice: Use MAE\n"
    "   • Most robust and interpretable\n"
    "   • Works in all scenarios\n\n"
    "2️⃣ Cross-Scale Comparison: Use MAPE\n"
    "   • Only if no zeros exist\n"
    "   • Compare models across datasets\n\n"
    "3️⃣ Always Report Multiple Metrics\n"
    "   • MAE for absolute error\n"
    "   • MAPE/SMAPE for relative error\n\n"
    "4️⃣ Compare to Naive Baseline\n"
    "   • Validates model adds value\n"
    "   • Seasonal naive for seasonality"
)

ax4.text(0.5, 0.27, guidelines, ha='center', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9,
                 edgecolor='#F57C00', linewidth=2.5),
        family='sans-serif', linespacing=1.5)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-07-time-series/ch30-advanced-ts/diagrams/metrics_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: metrics_comparison.png")
plt.close()

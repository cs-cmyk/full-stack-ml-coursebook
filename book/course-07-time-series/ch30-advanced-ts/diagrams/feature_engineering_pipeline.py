"""
Feature engineering pipeline for time series with tree-based models
Shows transformation from time series to supervised learning
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Feature Engineering Pipeline for Tree-Based Time Series Models',
        fontsize=16, fontweight='bold', ha='center')

# Colors
color_raw = '#607D8B'
color_lag = '#2196F3'
color_rolling = '#4CAF50'
color_temporal = '#FF9800'
color_cyclical = '#9C27B0'
color_final = '#F44336'

# Raw Time Series (top)
raw_box = FancyBboxPatch((0.5, 8), 2, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=color_raw, facecolor=color_raw, alpha=0.3, linewidth=2.5)
ax.add_patch(raw_box)
ax.text(1.5, 8.4, 'Raw Time Series', fontsize=12, ha='center', fontweight='bold')
ax.text(1.5, 7.7, '[t₁, t₂, t₃, ..., tₙ]', fontsize=9, ha='center', family='monospace')

# Arrow down
ax.annotate('', xy=(1.5, 7.5), xytext=(1.5, 7.8),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Feature Engineering Box (central)
feature_box = Rectangle((0.2, 2.5), 9.6, 4.5, edgecolor='#333333',
                        facecolor='#F5F5F5', alpha=0.5, linewidth=2.5, linestyle='--')
ax.add_patch(feature_box)
ax.text(5, 6.8, 'Feature Engineering Layer', fontsize=13, ha='center',
        fontweight='bold', style='italic', color='#333333')

# LAG FEATURES
y_lag = 5.5
lag_box = FancyBboxPatch((0.5, y_lag), 1.8, 0.9, boxstyle="round,pad=0.08",
                        edgecolor=color_lag, facecolor=color_lag, alpha=0.25, linewidth=2)
ax.add_patch(lag_box)
ax.text(1.4, y_lag + 0.45, 'Lag Features', fontsize=11, ha='center', fontweight='bold', color=color_lag)
lag_text = 'lag_1 = yₜ₋₁\nlag_2 = yₜ₋₂\nlag_12 = yₜ₋₁₂'
ax.text(1.4, y_lag + 0.15, lag_text, fontsize=8, ha='center', va='center', family='monospace')

# ROLLING STATISTICS
y_roll = 5.5
roll_box = FancyBboxPatch((2.5, y_roll), 2.2, 0.9, boxstyle="round,pad=0.08",
                         edgecolor=color_rolling, facecolor=color_rolling, alpha=0.25, linewidth=2)
ax.add_patch(roll_box)
ax.text(3.6, y_roll + 0.45, 'Rolling Statistics', fontsize=11, ha='center', fontweight='bold', color=color_rolling)
roll_text = 'roll_mean_3\nroll_std_7\nroll_mean_12'
ax.text(3.6, y_roll + 0.15, roll_text, fontsize=8, ha='center', va='center', family='monospace')

# TEMPORAL FEATURES
y_temp = 5.5
temp_box = FancyBboxPatch((5, y_temp), 2.2, 0.9, boxstyle="round,pad=0.08",
                         edgecolor=color_temporal, facecolor=color_temporal, alpha=0.25, linewidth=2)
ax.add_patch(temp_box)
ax.text(6.1, y_temp + 0.45, 'Temporal Features', fontsize=11, ha='center', fontweight='bold', color=color_temporal)
temp_text = 'month\nday_of_week\nquarter\nis_weekend'
ax.text(6.1, y_temp + 0.1, temp_text, fontsize=8, ha='center', va='center', family='monospace')

# CYCLICAL ENCODING
y_cyc = 5.5
cyc_box = FancyBboxPatch((7.5, y_cyc), 2, 0.9, boxstyle="round,pad=0.08",
                        edgecolor=color_cyclical, facecolor=color_cyclical, alpha=0.25, linewidth=2)
ax.add_patch(cyc_box)
ax.text(8.5, y_cyc + 0.45, 'Cyclical Encoding', fontsize=11, ha='center', fontweight='bold', color=color_cyclical)
cyc_text = 'month_sin\nmonth_cos\n(Dec → Jan)'
ax.text(8.5, y_cyc + 0.1, cyc_text, fontsize=8, ha='center', va='center', family='monospace')

# Example transformation in middle
y_example = 4.2
ax.text(5, y_example + 0.4, 'Example Transformation', fontsize=11, ha='center',
        fontweight='bold', style='italic', color='#333')

# Before table
before_text = (
    "Time Series Format:\n"
    "───────────────────\n"
    "  t    y\n"
    "  1   100\n"
    "  2   105\n"
    "  3   110"
)
ax.text(2, y_example - 0.3, before_text, fontsize=8, ha='center', va='top',
        family='monospace', bbox=dict(boxstyle='round', facecolor='white',
        edgecolor='#CCCCCC', linewidth=1.5, alpha=0.9))

# Arrow
ax.annotate('', xy=(4.5, y_example - 0.2), xytext=(3.5, y_example - 0.2),
            arrowprops=dict(arrowstyle='->', lw=3, color='#333'))
ax.text(4, y_example, 'Transform', fontsize=9, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# After table
after_text = (
    "Supervised Learning Format:\n"
    "─────────────────────────────────────\n"
    "  lag_1  lag_2  month  y_target\n"
    "  100    NaN    1      105\n"
    "  105    100    1      110\n"
    "  110    105    1      115"
)
ax.text(7.5, y_example - 0.3, after_text, fontsize=8, ha='center', va='top',
        family='monospace', bbox=dict(boxstyle='round', facecolor='white',
        edgecolor='#CCCCCC', linewidth=1.5, alpha=0.9))

# Arrows from features to combined
for x in [1.4, 3.6, 6.1, 8.5]:
    ax.annotate('', xy=(5, 2.5), xytext=(x, y_lag - 0.1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#666', alpha=0.6))

# Combined feature matrix
combined_box = FancyBboxPatch((3, 1.3), 4, 1, boxstyle="round,pad=0.1",
                            edgecolor=color_final, facecolor=color_final, alpha=0.3, linewidth=2.5)
ax.add_patch(combined_box)
ax.text(5, 1.8, 'Combined Feature Matrix', fontsize=12, ha='center', fontweight='bold', color=color_final)
matrix_text = 'X = [lag_1, lag_2, ..., roll_mean_3, ..., month_sin, ...]'
ax.text(5, 1.5, matrix_text, fontsize=8.5, ha='center', family='monospace')

# Arrow to model
ax.annotate('', xy=(5, 0.8), xytext=(5, 1.3),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Model box
model_box = FancyBboxPatch((3.5, 0.1), 3, 0.6, boxstyle="round,pad=0.1",
                          edgecolor='#333', facecolor='#333', alpha=0.8, linewidth=2)
ax.add_patch(model_box)
ax.text(5, 0.4, 'XGBoost / LightGBM / Random Forest', fontsize=11, ha='center',
        fontweight='bold', color='white')

# Side notes
notes = [
    "Key Principles:",
    "• Create lag features for autocorrelation",
    "• Rolling stats capture trends",
    "• Date features encode calendar patterns",
    "• Cyclical encoding preserves periodicity",
    "• Drop NaN rows after feature creation",
    "• Split data BEFORE engineering (avoid leakage)"
]

y_note = 9.2
for i, note in enumerate(notes):
    style = 'italic' if i == 0 else 'normal'
    weight = 'bold' if i == 0 else 'normal'
    ax.text(7.5, y_note - i*0.25, note, fontsize=9, ha='left',
            style=style, fontweight=weight, color='#333')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-07-time-series/ch30-advanced-ts/diagrams/feature_engineering_pipeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: feature_engineering_pipeline.png")
plt.close()

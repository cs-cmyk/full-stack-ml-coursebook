#!/usr/bin/env python3
"""Generate outlier detection diagram"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Set random seed and style
np.random.seed(42)
sns.set_style('whitegrid')

# Load data
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# Focus on median income feature
feature_name = 'MedInc'
data = df[feature_name].values

# METHOD 1: IQR Method
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = (data < lower_bound) | (data > upper_bound)
n_outliers_iqr = np.sum(outliers_iqr)

# METHOD 2: Z-Score Method
mean = np.mean(data)
std = np.std(data, ddof=1)
z_scores = (data - mean) / std
threshold = 3.0
outliers_zscore = np.abs(z_scores) > threshold
n_outliers_zscore = np.sum(outliers_zscore)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Box plot showing outliers (IQR method)
box_parts = axes[0, 0].boxplot(data, vert=True, patch_artist=True, widths=0.5)
box_parts['boxes'][0].set_facecolor('#2196F3')
box_parts['boxes'][0].set_alpha(0.7)
box_parts['medians'][0].set_color('#F44336')
box_parts['medians'][0].set_linewidth(2)
axes[0, 0].axhline(lower_bound, color='#FF9800', linestyle='--', linewidth=2,
                    label=f'Lower fence = {lower_bound:.2f}')
axes[0, 0].axhline(upper_bound, color='#FF9800', linestyle='--', linewidth=2,
                    label=f'Upper fence = {upper_bound:.2f}')
axes[0, 0].set_ylabel(f'{feature_name}', fontsize=12)
axes[0, 0].set_title(f'Box Plot: IQR Outlier Detection\n{n_outliers_iqr} outliers',
                      fontsize=12, weight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Histogram with outlier regions shaded (Z-score method)
axes[0, 1].hist(data, bins=50, alpha=0.7, color='#2196F3', edgecolor='black')
axes[0, 1].axvline(mean - threshold*std, color='#F44336', linestyle='--', linewidth=2,
                    label=f'μ - {threshold}σ')
axes[0, 1].axvline(mean + threshold*std, color='#F44336', linestyle='--', linewidth=2,
                    label=f'μ + {threshold}σ')
axes[0, 1].axvline(mean, color='#4CAF50', linestyle='-', linewidth=2, label='Mean')
axes[0, 1].set_xlabel(f'{feature_name}', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title(f'Histogram: Z-Score Outlier Detection\n{n_outliers_zscore} outliers beyond ±{threshold}σ',
                      fontsize=12, weight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3. Scatter plot: index vs value (both methods)
indices = np.arange(len(data))
colors = np.array(['#2196F3'] * len(data))
colors[outliers_iqr & outliers_zscore] = '#F44336'      # Both methods
colors[outliers_iqr & ~outliers_zscore] = '#FF9800'     # IQR only
colors[~outliers_iqr & outliers_zscore] = '#9C27B0'     # Z-score only

axes[1, 0].scatter(indices, data, c=colors, alpha=0.5, s=10)
axes[1, 0].axhline(mean, color='#4CAF50', linestyle='-', linewidth=1, label='Mean')
axes[1, 0].axhline(upper_bound, color='#FF9800', linestyle='--', linewidth=1,
                    label='IQR fences')
axes[1, 0].axhline(lower_bound, color='#FF9800', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('Index', fontsize=12)
axes[1, 0].set_ylabel(f'{feature_name}', fontsize=12)
axes[1, 0].set_title('Scatter Plot: Outliers by Both Methods', fontsize=12, weight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 4. Impact of outlier removal on statistics
data_no_outliers = data[~outliers_iqr]
stats_comparison = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
    'With Outliers': [
        len(data),
        np.mean(data),
        np.median(data),
        np.std(data, ddof=1),
        np.min(data),
        np.max(data)
    ],
    'Without Outliers': [
        len(data_no_outliers),
        np.mean(data_no_outliers),
        np.median(data_no_outliers),
        np.std(data_no_outliers, ddof=1),
        np.min(data_no_outliers),
        np.max(data_no_outliers)
    ]
})
stats_comparison['Change (%)'] = 100 * (stats_comparison['Without Outliers'] -
                                         stats_comparison['With Outliers']) / stats_comparison['With Outliers']

# Format the table values
table_data = []
for idx, row in stats_comparison.iterrows():
    stat = row['Statistic']
    with_out = f"{row['With Outliers']:.2f}" if stat != 'Count' else f"{int(row['With Outliers'])}"
    without_out = f"{row['Without Outliers']:.2f}" if stat != 'Count' else f"{int(row['Without Outliers'])}"
    change = f"{row['Change (%)']:.2f}%"
    table_data.append([stat, with_out, without_out, change])

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=table_data,
                         colLabels=['Statistic', 'With Outliers', 'Without Outliers', 'Change (%)'],
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
# Color header
for i in range(4):
    table[(0, i)].set_facecolor('#607D8B')
    table[(0, i)].set_text_props(weight='bold', color='white')

axes[1, 1].set_title('Impact of Outlier Removal (IQR method)', fontsize=12, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated outlier_detection.png")

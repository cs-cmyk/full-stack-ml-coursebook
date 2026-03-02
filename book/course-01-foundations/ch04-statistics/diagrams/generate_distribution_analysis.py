#!/usr/bin/env python3
"""Generate comprehensive distribution analysis diagram"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Set style and random seed
sns.set_style('whitegrid')
np.random.seed(42)

# Load data
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# Focus on MedInc feature
income = df['MedInc']

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with mean and median lines
axes[0, 0].hist(income, bins=50, alpha=0.7, color='#2196F3', edgecolor='black')
mean_val = income.mean()
median_val = income.median()
axes[0, 0].axvline(mean_val, color='#F44336', linestyle='--', linewidth=2,
                    label=f'Mean = {mean_val:.2f}')
axes[0, 0].axvline(median_val, color='#4CAF50', linestyle='--', linewidth=2,
                    label=f'Median = {median_val:.2f}')
axes[0, 0].set_xlabel('Median Income ($10,000s)', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Histogram: Distribution of Median Income', fontsize=12, weight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plot
box_parts = axes[0, 1].boxplot(income, vert=True, patch_artist=True, widths=0.5)
box_parts['boxes'][0].set_facecolor('#2196F3')
box_parts['boxes'][0].set_alpha(0.7)
box_parts['medians'][0].set_color('#F44336')
box_parts['medians'][0].set_linewidth(2)
axes[0, 1].set_ylabel('Median Income ($10,000s)', fontsize=12)
axes[0, 1].set_title('Box Plot: Identifying Outliers', fontsize=12, weight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Add annotations for box plot components
q1, median, q3 = np.percentile(income, [25, 50, 75])
iqr = q3 - q1
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr
axes[0, 1].text(1.15, q1, f'Q1 = {q1:.2f}', fontsize=10)
axes[0, 1].text(1.15, median, f'Median = {median:.2f}', fontsize=10, color='#F44336')
axes[0, 1].text(1.15, q3, f'Q3 = {q3:.2f}', fontsize=10)

# 3. Multiple features comparison
features_to_plot = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
bp = df[features_to_plot].boxplot(ax=axes[1, 0], patch_artist=True, return_type='dict')
for patch in bp['boxes']:
    patch.set_facecolor('#2196F3')
    patch.set_alpha(0.7)
axes[1, 0].set_ylabel('Value (various units)', fontsize=12)
axes[1, 0].set_title('Box Plots: Comparing Multiple Features', fontsize=12, weight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Skewness visualization
skew_values = df[features_to_plot].skew()
colors = ['#4CAF50' if abs(s) < 0.5 else '#FF9800' if abs(s) < 1 else '#F44336'
          for s in skew_values]
axes[1, 1].bar(features_to_plot, skew_values, color=colors, edgecolor='black', alpha=0.8)
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1, 1].axhline(0.5, color='#FF9800', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 1].axhline(-0.5, color='#FF9800', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 1].set_ylabel('Skewness', fontsize=12)
axes[1, 1].set_title('Skewness: Measuring Distribution Asymmetry', fontsize=12, weight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated distribution_analysis.png")

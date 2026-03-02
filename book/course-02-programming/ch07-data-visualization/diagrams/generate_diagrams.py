#!/usr/bin/env python3
"""
Generate Educational Diagrams for Chapter 7: Data Visualization
Creates clean, pedagogical visualizations for the textbook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure consistent styling
sns.set_style('whitegrid')
sns.set_palette('colorblind')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Color palette (consistent with textbook)
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Load California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseValue'] = housing.target
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# ============================================================
# DIAGRAM 1: Univariate Analysis - Distribution Exploration
# ============================================================
print("\n[1/5] Creating univariate analysis diagram...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Univariate Analysis: Distribution Exploration',
             fontsize=16, fontweight='bold', y=0.995)

# Top-left: Histogram of Median Income
ax1 = axes[0, 0]
ax1.hist(df['MedInc'], bins=30, color=COLORS['blue'], edgecolor='black', alpha=0.7)
mean_val = df['MedInc'].mean()
median_val = df['MedInc'].median()
ax1.axvline(mean_val, color=COLORS['red'], linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f}')
ax1.axvline(median_val, color=COLORS['orange'], linestyle='--', linewidth=2,
           label=f'Median: {median_val:.2f}')
ax1.set_xlabel('Median Income (tens of thousands $)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('A) Histogram: Median Income Distribution', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Top-right: Box plot of Median Income
ax2 = axes[0, 1]
bp = ax2.boxplot(df['MedInc'], vert=False, patch_artist=True, widths=0.5,
                 boxprops=dict(facecolor=COLORS['blue'], edgecolor='black', alpha=0.7),
                 medianprops=dict(color=COLORS['red'], linewidth=2),
                 whiskerprops=dict(color='black', linewidth=1.5),
                 capprops=dict(color='black', linewidth=1.5),
                 flierprops=dict(marker='o', markerfacecolor=COLORS['red'],
                                markersize=4, alpha=0.5))
ax2.set_xlabel('Median Income (tens of thousands $)', fontsize=12)
ax2.set_title('B) Box Plot: Median Income (with Outliers)', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.set_yticks([])

# Bottom-left: Histogram of House Values
ax3 = axes[1, 0]
ax3.hist(df['MedHouseValue'], bins=30, color=COLORS['green'], edgecolor='black', alpha=0.7)
mean_val = df['MedHouseValue'].mean()
median_val = df['MedHouseValue'].median()
ax3.axvline(mean_val, color=COLORS['red'], linestyle='--', linewidth=2,
           label=f'Mean: {mean_val:.2f}')
ax3.axvline(median_val, color=COLORS['orange'], linestyle='--', linewidth=2,
           label=f'Median: {median_val:.2f}')
ax3.set_xlabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('C) Histogram: House Value Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Bottom-right: Kernel Density Estimate (KDE)
ax4 = axes[1, 1]
sns.kdeplot(data=df, x='MedHouseValue', fill=True, color=COLORS['purple'], alpha=0.6, ax=ax4)
ax4.axvline(df['MedHouseValue'].mean(), color=COLORS['red'], linestyle='--',
           linewidth=2, label='Mean')
ax4.axvline(df['MedHouseValue'].median(), color=COLORS['orange'], linestyle='--',
           linewidth=2, label='Median')
ax4.set_xlabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('D) KDE: Smooth Density Estimate', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('univariate_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: univariate_analysis.png")
plt.close()

# ============================================================
# DIAGRAM 2: Bivariate Analysis - Exploring Relationships
# ============================================================
print("[2/5] Creating bivariate analysis diagram...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Bivariate Analysis: Exploring Relationships',
             fontsize=16, fontweight='bold')

# Left: Scatter plot with color gradient
ax1 = axes[0]
scatter = ax1.scatter(df['MedInc'], df['MedHouseValue'],
                     c=df['HouseAge'], cmap='viridis',
                     alpha=0.4, s=8, edgecolors='none')
ax1.set_xlabel('Median Income (tens of thousands $)', fontsize=12)
ax1.set_ylabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax1.set_title('A) Income vs. House Value (colored by House Age)',
             fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('House Age (years)', fontsize=11)
ax1.grid(alpha=0.3)

# Right: Hexbin plot for density visualization
ax2 = axes[1]
hexbin = ax2.hexbin(df['MedInc'], df['MedHouseValue'],
                    gridsize=30, cmap='YlOrRd', mincnt=1, alpha=0.8)
ax2.set_xlabel('Median Income (tens of thousands $)', fontsize=12)
ax2.set_ylabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax2.set_title('B) Income vs. House Value (hexbin density)',
             fontsize=13, fontweight='bold')
cbar2 = plt.colorbar(hexbin, ax=ax2)
cbar2.set_label('Count of houses', fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bivariate_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: bivariate_analysis.png")
plt.close()

# ============================================================
# DIAGRAM 3: Categorical Comparison - Box and Violin Plots
# ============================================================
print("[3/5] Creating categorical comparison diagram...")

# Create categorical bins for house age
df['AgeCategory'] = pd.cut(df['HouseAge'], bins=[0, 15, 30, 100],
                            labels=['New (0-15)', 'Middle (15-30)', 'Old (30+)'])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Categorical Comparison: House Value by Age Category',
             fontsize=16, fontweight='bold')

# Left: Box plot
ax1 = axes[0]
sns.boxplot(data=df, x='AgeCategory', y='MedHouseValue', palette='Set2', ax=ax1)
ax1.set_xlabel('House Age Category', fontsize=12)
ax1.set_ylabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax1.set_title('A) Box Plot: Distribution by Category', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Middle: Violin plot
ax2 = axes[1]
sns.violinplot(data=df, x='AgeCategory', y='MedHouseValue', palette='Set2', ax=ax2)
ax2.set_xlabel('House Age Category', fontsize=12)
ax2.set_ylabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax2.set_title('B) Violin Plot: Full Distribution Shape', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Right: Strip plot with point plot overlay
ax3 = axes[2]
# Sample data for strip plot
df_sample = df.sample(n=1000, random_state=42)
sns.stripplot(data=df_sample, x='AgeCategory', y='MedHouseValue',
              alpha=0.2, size=3, color=COLORS['gray'], ax=ax3)
sns.pointplot(data=df, x='AgeCategory', y='MedHouseValue',
              color=COLORS['red'], markers='D', scale=1.2,
              errorbar=('ci', 95), ax=ax3, legend=False)
ax3.set_xlabel('House Age Category', fontsize=12)
ax3.set_ylabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax3.set_title('C) Strip Plot + Mean with 95% CI', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('categorical_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: categorical_comparison.png")
plt.close()

# ============================================================
# DIAGRAM 4: Multivariate Visualization - Correlations
# ============================================================
print("[4/5] Creating multivariate analysis diagram...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Multivariate Analysis: Feature Correlations',
             fontsize=16, fontweight='bold')

# Left: Correlation heatmap
ax1 = axes[0]
features_for_corr = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                     'Population', 'AveOccup', 'MedHouseValue']
corr_matrix = df[features_for_corr].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1.5, linecolor='white',
            cbar_kws={'label': 'Correlation Coefficient'}, ax=ax1,
            vmin=-0.3, vmax=0.7)
ax1.set_title('A) Correlation Heatmap: Feature Relationships',
             fontsize=13, fontweight='bold')

# Right: Scatter matrix of top correlated features
ax2 = axes[1]
# Select top 3 features correlated with target
top_features = corr_matrix['MedHouseValue'].abs().sort_values(ascending=False).index[1:4]
df_sample = df.sample(n=2000, random_state=42)

# Create custom scatter with size encoding
scatter = ax2.scatter(df_sample['MedInc'], df_sample['MedHouseValue'],
                     c=df_sample['AveRooms'], s=20, cmap='plasma',
                     alpha=0.5, edgecolors='none')
ax2.set_xlabel('Median Income (tens of thousands $)', fontsize=12)
ax2.set_ylabel('Median House Value (hundreds of thousands $)', fontsize=12)
ax2.set_title('B) Top Predictors: Income vs. Value\n(colored by Avg Rooms)',
             fontsize=13, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Average Rooms', fontsize=11)
ax2.grid(alpha=0.3)

# Add correlation text
corr_val = df['MedInc'].corr(df['MedHouseValue'])
ax2.text(0.05, 0.95, f'Correlation: r = {corr_val:.3f}',
        transform=ax2.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('multivariate_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: multivariate_analysis.png")
plt.close()

# ============================================================
# DIAGRAM 5: Publication-Quality Figure
# ============================================================
print("[5/5] Creating publication-quality diagram...")

fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot with multiple encodings
scatter = ax.scatter(df['MedInc'], df['MedHouseValue'],
                    c=df['HouseAge'], s=df['AveRooms']*3,
                    cmap='plasma', alpha=0.3, edgecolors='none')

# Customizations
ax.set_xlabel('Median Income (tens of thousands $)', fontsize=13, fontweight='bold')
ax.set_ylabel('Median House Value (hundreds of thousands $)', fontsize=13, fontweight='bold')
ax.set_title('California Housing: Income vs. Value\n(Size = Avg Rooms, Color = House Age)',
             fontsize=15, fontweight='bold', pad=15)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('House Age (years)', fontsize=12, fontweight='bold')

# Add linear trend line
z = np.polyfit(df['MedInc'], df['MedHouseValue'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['MedInc'].min(), df['MedInc'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5,
       label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')

# Statistics box
corr = df['MedInc'].corr(df['MedHouseValue'])
ax.text(0.05, 0.95, f'n = {len(df):,} houses\nCorrelation: r = {corr:.3f}',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                 edgecolor=COLORS['blue'], linewidth=2))

ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(left=-0.5)
ax.set_ylim(bottom=-0.2)

plt.tight_layout()
plt.savefig('publication_quality.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: publication_quality.png")
plt.close()

# ============================================================
# BONUS: Chart Type Comparison
# ============================================================
print("[BONUS] Creating chart type comparison diagram...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Common Chart Types: When to Use Each',
             fontsize=16, fontweight='bold', y=0.995)

# Prepare sample data
sample_data = df.sample(n=500, random_state=42)

# 1. Histogram
ax = axes[0, 0]
ax.hist(sample_data['MedInc'], bins=20, color=COLORS['blue'],
       edgecolor='black', alpha=0.7)
ax.set_title('Histogram\nDistribution of one variable', fontweight='bold')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.grid(axis='y', alpha=0.3)

# 2. Box Plot
ax = axes[0, 1]
bp = ax.boxplot([sample_data[sample_data['AgeCategory'] == cat]['MedHouseValue'].values
                 for cat in ['New (0-15)', 'Middle (15-30)', 'Old (30+)']],
                labels=['New', 'Middle', 'Old'], patch_artist=True)
for patch, color in zip(bp['boxes'], [COLORS['green'], COLORS['orange'], COLORS['red']]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Box Plot\nCompare distributions across groups', fontweight='bold')
ax.set_ylabel('Value')
ax.grid(axis='y', alpha=0.3)

# 3. Scatter Plot
ax = axes[0, 2]
ax.scatter(sample_data['MedInc'], sample_data['MedHouseValue'],
          alpha=0.5, s=20, color=COLORS['purple'])
ax.set_title('Scatter Plot\nRelationship between two variables', fontweight='bold')
ax.set_xlabel('Variable 1')
ax.set_ylabel('Variable 2')
ax.grid(alpha=0.3)

# 4. Line Plot
ax = axes[1, 0]
time_series = sample_data.sort_values('HouseAge').head(50)
ax.plot(time_series['HouseAge'], time_series['MedHouseValue'],
       marker='o', markersize=4, linewidth=2, color=COLORS['blue'])
ax.set_title('Line Plot\nTrend over time or ordered data', fontweight='bold')
ax.set_xlabel('Time/Order')
ax.set_ylabel('Value')
ax.grid(alpha=0.3)

# 5. Bar Chart
ax = axes[1, 1]
cat_counts = df['AgeCategory'].value_counts().sort_index()
bars = ax.bar(range(len(cat_counts)), cat_counts.values,
             color=[COLORS['green'], COLORS['orange'], COLORS['red']], alpha=0.7)
ax.set_xticks(range(len(cat_counts)))
ax.set_xticklabels(['New', 'Middle', 'Old'], rotation=0)
ax.set_title('Bar Chart\nCounts or values by category', fontweight='bold')
ax.set_ylabel('Count')
ax.grid(axis='y', alpha=0.3)

# 6. Heatmap (small correlation matrix)
ax = axes[1, 2]
small_corr = df[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseValue']].corr()
im = ax.imshow(small_corr, cmap='coolwarm', aspect='auto', vmin=-0.5, vmax=1)
ax.set_xticks(range(len(small_corr)))
ax.set_yticks(range(len(small_corr)))
ax.set_xticklabels(small_corr.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(small_corr.columns, fontsize=9)
ax.set_title('Heatmap\nMultiple variable correlations', fontweight='bold')
# Add values
for i in range(len(small_corr)):
    for j in range(len(small_corr)):
        text = ax.text(j, i, f'{small_corr.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig('chart_types_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✓ Saved: chart_types_comparison.png")
plt.close()

print("\n" + "="*60)
print("✓ All diagrams generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. univariate_analysis.png")
print("  2. bivariate_analysis.png")
print("  3. categorical_comparison.png")
print("  4. multivariate_analysis.png")
print("  5. publication_quality.png")
print("  6. chart_types_comparison.png (BONUS)")
print("\nAll files saved to current directory.")

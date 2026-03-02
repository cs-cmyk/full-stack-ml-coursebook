#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 12: Correlation and Relationships
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.datasets import fetch_california_housing

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12
np.random.seed(42)

print("Generating diagrams for Chapter 12: Correlation and Relationships")
print("=" * 70)

# ============================================
# DIAGRAM 1: Correlation Patterns (3x3 grid)
# ============================================
print("\n1. Generating correlation_patterns.png...")

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Understanding Correlation: Visual Patterns for Different r Values',
             fontsize=16, fontweight='bold', y=0.995)

# Define correlation values to demonstrate
correlations = [1.0, 0.9, 0.7, 0.5, 0.3, 0.0, -0.3, -0.7, -0.95]

# Generate data for each correlation level
n = 100

for idx, target_r in enumerate(correlations):
    ax = axes[idx // 3, idx % 3]

    # Generate correlated data
    if target_r == 0:
        x = np.random.randn(n)
        y = np.random.randn(n)
    else:
        x = np.random.randn(n)
        y = target_r * x + np.sqrt(1 - target_r**2) * np.random.randn(n)

    # Calculate actual correlation
    actual_r, _ = pearsonr(x, y)

    # Create scatter plot
    ax.scatter(x, y, alpha=0.6, s=30, color='#2196F3', edgecolors='#1976D2', linewidth=0.5)

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color='#F44336', linewidth=2, alpha=0.7,
            label=f'r = {actual_r:.2f}')

    # Formatting
    ax.set_xlabel('Variable X', fontsize=10)
    ax.set_ylabel('Variable Y', fontsize=10)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add interpretation text
    if actual_r > 0.7:
        strength = "Strong Positive"
    elif actual_r > 0.3:
        strength = "Moderate Positive"
    elif actual_r > -0.3:
        strength = "Weak/None"
    elif actual_r > -0.7:
        strength = "Moderate Negative"
    else:
        strength = "Strong Negative"

    ax.set_title(strength, fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('correlation_patterns.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ correlation_patterns.png saved (3x3 grid showing r from +1.0 to -0.95)")

# ============================================
# DIAGRAM 2: Income vs House Value
# ============================================
print("\n2. Generating income_vs_value.png...")

# Load California Housing dataset
housing_data = fetch_california_housing()
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['MedHouseVal'] = housing_data.target

feature_x = 'MedInc'
feature_y = 'MedHouseVal'

# Compute correlation
r, p_value = pearsonr(df[feature_x], df[feature_y])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot with regression line
axes[0].scatter(df[feature_x], df[feature_y], alpha=0.3, s=10,
                color='#2196F3', edgecolors='none')
# Add regression line
z = np.polyfit(df[feature_x], df[feature_y], 1)
p = np.poly1d(z)
x_line = np.linspace(df[feature_x].min(), df[feature_x].max(), 100)
axes[0].plot(x_line, p(x_line), color='#F44336', linewidth=2.5, alpha=0.8,
            label=f'r = {r:.3f}')
axes[0].set_xlabel('Median Income (in $10,000s)', fontsize=12)
axes[0].set_ylabel('Median House Value (in $100,000s)', fontsize=12)
axes[0].set_title('Income vs. House Value: Strong Positive Correlation',
                 fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Hexbin plot for better visualization with many points
hb = axes[1].hexbin(df[feature_x], df[feature_y], gridsize=50,
                    cmap='Blues', mincnt=1)
axes[1].plot(x_line, p(x_line), color='#F44336', linewidth=2.5, alpha=0.9,
            label=f'r = {r:.3f}')
axes[1].set_xlabel('Median Income (in $10,000s)', fontsize=12)
axes[1].set_ylabel('Median House Value (in $100,000s)', fontsize=12)
axes[1].set_title('Hexbin Density Plot (Better for Large Datasets)',
                 fontsize=13, fontweight='bold')
cb = plt.colorbar(hb, ax=axes[1])
cb.set_label('Count', fontsize=11)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('income_vs_value.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ income_vs_value.png saved (scatter + hexbin plots)")

# ============================================
# DIAGRAM 3: Correlation Heatmap
# ============================================
print("\n3. Generating correlation_heatmap.png...")

# Compute correlation matrix for all features
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize=(12, 9))

# Create heatmap with annotations
sns.heatmap(corr_matrix,
           annot=True,           # Show correlation values
           fmt='.2f',            # Format to 2 decimal places
           cmap='coolwarm',      # Diverging colormap (blue-white-red)
           center=0,             # Center colormap at zero
           vmin=-1,              # Minimum value
           vmax=1,               # Maximum value
           square=True,          # Square cells
           linewidths=0.5,       # Grid lines between cells
           linecolor='white',
           cbar_kws={'label': 'Correlation Coefficient',
                    'shrink': 0.8})

plt.title('Correlation Matrix: California Housing Dataset',
         fontsize=15, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ correlation_heatmap.png saved (9x9 heatmap)")

print("\n" + "=" * 70)
print("All diagrams generated successfully!")
print("=" * 70)
print("\nGenerated files:")
print("  1. correlation_patterns.png  - 3x3 grid showing correlation patterns")
print("  2. income_vs_value.png       - Scatter and hexbin plots")
print("  3. correlation_heatmap.png   - Full correlation matrix heatmap")

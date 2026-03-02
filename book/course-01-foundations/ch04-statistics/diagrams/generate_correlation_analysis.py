#!/usr/bin/env python3
"""Generate correlation analysis diagram"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set random seed and style
np.random.seed(42)
sns.set_style('whitegrid')

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Select two features
x = df['petal length (cm)'].values
y = df['petal width (cm)'].values

# Manual correlation calculation
mean_x = np.mean(x)
mean_y = np.mean(y)
covariance = np.sum((x - mean_x) * (y - mean_y)) / len(x)
std_x = np.std(x, ddof=1)
std_y = np.std(y, ddof=1)
correlation_manual = covariance / (std_x * std_y)

# Compute full correlation matrix
corr_matrix = df.iloc[:, :-1].corr()  # Exclude target column

# Visualize correlation matrix as heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Correlation Matrix Heatmap\n(Iris Features)', fontsize=12, weight='bold')

# Scatter plot of highly correlated pair
axes[1].scatter(x, y, alpha=0.6, s=50, color='#2196F3', edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Petal Length (cm)', fontsize=12)
axes[1].set_ylabel('Petal Width (cm)', fontsize=12)
axes[1].set_title(f'Scatter Plot: Petal Length vs Width\nr = {correlation_manual:.4f}',
                  fontsize=12, weight='bold')
axes[1].grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
axes[1].plot(x, p(x), "#F44336", linestyle='--', linewidth=2, label='Linear fit')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated correlation_analysis.png")

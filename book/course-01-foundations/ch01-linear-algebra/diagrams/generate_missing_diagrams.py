#!/usr/bin/env python3
"""
Generate missing diagrams for Chapter 1: Linear Algebra
Based on specifications in content.md
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.datasets import load_iris, fetch_california_housing
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Use consistent color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

print("="*60)
print("Generating diagrams for Chapter 1: Linear Algebra")
print("="*60)

# ============================================================================
# DIAGRAM 1: Vectors Dual View
# ============================================================================
print("\n[1/2] Generating vectors_dual_view.png...")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Geometric view - vectors as arrows
ax1.set_xlim(-0.5, 4)
ax1.set_ylim(-0.5, 4)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Draw vectors as arrows
v1 = np.array([3, 2])
v2 = np.array([1, 3])

arrow1 = FancyArrowPatch((0, 0), tuple(v1),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=3, color=COLORS['red'])
arrow2 = FancyArrowPatch((0, 0), tuple(v2),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=3, color=COLORS['blue'])

ax1.add_patch(arrow1)
ax1.add_patch(arrow2)
ax1.text(v1[0]+0.1, v1[1]+0.1, 'v₁ = [3, 2]', fontsize=12,
         color=COLORS['red'], fontweight='bold')
ax1.text(v2[0]+0.1, v2[1]+0.1, 'v₂ = [1, 3]', fontsize=12,
         color=COLORS['blue'], fontweight='bold')
ax1.set_xlabel('x₁', fontsize=12, fontweight='bold')
ax1.set_ylabel('x₂', fontsize=12, fontweight='bold')
ax1.set_title('Geometric View: Vectors as Arrows', fontsize=14, fontweight='bold')

# Right plot: Data science view - vectors as data points
iris = load_iris()
X = iris.data[:, :2]  # Use only first 2 features for visualization

ax2.scatter(X[:, 0], X[:, 1], c=COLORS['purple'], alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
# Highlight three specific samples
sample_indices = [0, 50, 100]
for idx in sample_indices:
    ax2.scatter(X[idx, 0], X[idx, 1], c=COLORS['red'], s=200,
               edgecolors='black', linewidth=2, zorder=5)
    ax2.annotate(f'Sample {idx}\n[{X[idx, 0]:.1f}, {X[idx, 1]:.1f}]',
                xy=(X[idx, 0], X[idx, 1]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5',
                                       facecolor='wheat', alpha=0.9),
                fontweight='bold')

ax2.set_xlabel('Sepal Length (cm)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sepal Width (cm)', fontsize=12, fontweight='bold')
ax2.set_title('Data Science View: Vectors as Data Points', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vectors_dual_view.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Saved: vectors_dual_view.png")
print("  - Left: Geometric view (vectors as arrows)")
print("  - Right: Data science view (Iris samples as points)")

# ============================================================================
# DIAGRAM 2: Covariance Heatmap
# ============================================================================
print("\n[2/2] Generating covariance_heatmap.png...")

# Load California Housing dataset
housing = fetch_california_housing()
X_housing = housing.data[:1000]  # Use first 1000 samples for speed
feature_names_housing = housing.feature_names

# Standardize the data
mean_vals = np.mean(X_housing, axis=0)
std_vals = np.std(X_housing, axis=0)
X_centered = X_housing - mean_vals
X_scaled = X_centered / std_vals

# Compute covariance matrix
cov_matrix = np.cov(X_scaled.T)  # Transpose: want covariance between features

# Visualize covariance matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=feature_names_housing,
            yticklabels=feature_names_housing,
            center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Covariance'},
            linewidths=0.5, linecolor='white')
plt.title('Covariance Matrix of California Housing Features\n(Standardized Data)',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('covariance_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Saved: covariance_heatmap.png")
print("  - Shows feature correlations")
print("  - Red = positive correlation")
print("  - Blue = negative correlation")
print("  - Diagonal = variances (all ≈1 after standardization)")

print("\n" + "="*60)
print("All diagrams generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. vectors_dual_view.png (150 DPI, 14×6 inches)")
print("  2. covariance_heatmap.png (150 DPI, 10×8 inches)")
print("\nColor palette used:")
for name, color in COLORS.items():
    print(f"  - {name}: {color}")
print("="*60)

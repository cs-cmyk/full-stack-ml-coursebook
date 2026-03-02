#!/usr/bin/env python3
"""
Create distribution_fitting.png - showing actual vs fitted distribution and synthetic data
Uses Iris dataset sepal length
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_iris

# Set random seed for reproducibility
np.random.seed(42)

# Color palette
BLUE = '#2196F3'
RED = '#F44336'
ORANGE = '#FF9800'

# Load Iris dataset
iris = load_iris()
sepal_length = iris.data[:, 0]  # First column is sepal length

# Calculate statistics
mu = sepal_length.mean()
sigma = sepal_length.std()

# Fit normal distribution
fitted_dist = stats.norm(loc=mu, scale=sigma)

# Generate synthetic data from fitted distribution
synthetic_samples = fitted_dist.rvs(size=150, random_state=42)

# Create figure
plt.figure(figsize=(14, 5))

# Panel 1: Actual data with fitted PDF
plt.subplot(1, 2, 1)
plt.hist(sepal_length, bins=20, density=True, alpha=0.6,
         color=BLUE, edgecolor='black', linewidth=1.2, label='Actual Data')

# Plot fitted distribution
x = np.linspace(sepal_length.min() - 0.5, sepal_length.max() + 0.5, 200)
plt.plot(x, fitted_dist.pdf(x), color=RED, linewidth=3,
         label=f'Fitted Normal(μ={mu:.2f}, σ={sigma:.2f})')

# Add mean and standard deviation markers
plt.axvline(mu, color=RED, linestyle='--', linewidth=2, alpha=0.8, label=f'Mean μ = {mu:.2f}')
plt.axvline(mu + sigma, color=ORANGE, linestyle=':', linewidth=2, alpha=0.8, label='μ ± σ')
plt.axvline(mu - sigma, color=ORANGE, linestyle=':', linewidth=2, alpha=0.8)

# Shade the ±1σ region
plt.axvspan(mu - sigma, mu + sigma, alpha=0.1, color=ORANGE, label='±1σ region (~68%)')

plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Iris Sepal Length: Actual Data vs Fitted Normal Distribution', fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

# Add text box with statistics
textstr = f'Sample Size: {len(sepal_length)}\nMean: {mu:.3f} cm\nStd Dev: {sigma:.3f} cm'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: Actual vs Synthetic data comparison
plt.subplot(1, 2, 2)
plt.hist(sepal_length, bins=20, alpha=0.6, color=BLUE,
         edgecolor='black', linewidth=1.2, label=f'Actual Data (μ={mu:.2f}, σ={sigma:.2f})')
plt.hist(synthetic_samples, bins=20, alpha=0.6, color=RED,
         edgecolor='black', linewidth=1.2,
         label=f'Synthetic Data (μ={synthetic_samples.mean():.2f}, σ={synthetic_samples.std():.2f})')

plt.xlabel('Sepal Length (cm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Actual vs Synthetic Data Comparison', fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

# Add text box with comparison
prob_above_6_fitted = 1 - fitted_dist.cdf(6.0)
prob_above_6_actual = (sepal_length > 6.0).mean()
comparison_text = (f'P(X > 6.0 cm):\n'
                  f'Fitted: {prob_above_6_fitted:.1%}\n'
                  f'Actual: {prob_above_6_actual:.1%}\n'
                  f'Difference: {abs(prob_above_6_fitted - prob_above_6_actual):.1%}')
plt.text(0.02, 0.98, comparison_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-01-foundations/ch03-probability/diagrams/distribution_fitting.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Created: distribution_fitting.png")

#!/usr/bin/env python3
"""
Create expected_value_variance.png - showing house price distributions with E[X] and variance
Uses California Housing dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Set random seed for reproducibility
np.random.seed(42)

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'

# Load California Housing data
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df['MedHouseVal'] = housing.target * 100000  # Convert to dollars

# Get house prices
house_prices = housing_df['MedHouseVal']

# Calculate expected value and variance
E_X = house_prices.mean()  # Expected value
sigma_X = house_prices.std()  # Standard deviation

# Create income categories as proxy for regions
housing_df['income_category'] = pd.cut(housing_df['MedInc'],
                                       bins=[0, 3, 5, 10],
                                       labels=['Low', 'Medium', 'High'])

# Create figure
plt.figure(figsize=(14, 5))

# Panel 1: Expected Value as Balance Point
plt.subplot(1, 2, 1)
plt.hist(house_prices, bins=50, density=True, alpha=0.7,
         color=BLUE, edgecolor='black', linewidth=0.8)

# Mark the expected value
plt.axvline(E_X, color=RED, linewidth=3, linestyle='-',
           label=f'E[X] = ${E_X/1000:.0f}K')

# Mark ±1 standard deviation
plt.axvline(E_X + sigma_X, color=ORANGE, linewidth=2.5, linestyle='--',
            label=f'E[X] ± σ')
plt.axvline(E_X - sigma_X, color=ORANGE, linewidth=2.5, linestyle='--')

# Shade the ±1σ region
plt.axvspan(E_X - sigma_X, E_X + sigma_X, alpha=0.15, color=ORANGE,
            label='±1σ region')

# Calculate and show the probability within ±1σ
prob_within_1sigma = ((house_prices >= E_X - sigma_X) &
                      (house_prices <= E_X + sigma_X)).mean()

plt.xlabel('House Price ($)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Expected Value E[X] as the Balance Point', fontsize=13, fontweight='bold')
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3)

# Add text box with statistics
textstr = (f'E[X] = ${E_X:,.0f}\n'
          f'σ = ${sigma_X:,.0f}\n'
          f'P(E[X]-σ < X < E[X]+σ) = {prob_within_1sigma:.1%}')
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Add annotation pointing to the mean
plt.annotate('Expected Value:\nCenter of the distribution',
            xy=(E_X, 0.000002), xytext=(E_X + 100000, 0.000004),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2),
            fontsize=10, color=RED, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 2: Distributions by Income Region
plt.subplot(1, 2, 2)

colors = {'Low': GREEN, 'Medium': ORANGE, 'High': RED}
for category in ['Low', 'Medium', 'High']:
    subset = housing_df[housing_df['income_category'] == category]['MedHouseVal']
    cat_mean = subset.mean()
    cat_std = subset.std()

    plt.hist(subset, bins=30, alpha=0.5, label=f'{category} (μ=${cat_mean/1000:.0f}K, σ=${cat_std/1000:.0f}K)',
            density=True, color=colors[category], edgecolor='black', linewidth=0.5)

    # Mark mean for each category
    plt.axvline(cat_mean, color=colors[category], linewidth=2, linestyle='--', alpha=0.8)

plt.xlabel('House Price ($)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.title('Price Distributions by Income Region', fontsize=13, fontweight='bold')
plt.legend(fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3)

# Add interpretive text
interpretation = ('Higher income regions:\n'
                 '• Higher E[X] (more expensive)\n'
                 '• Higher σ (more variability)\n'
                 '• More uncertainty in predictions')
plt.text(0.02, 0.98, interpretation, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-01-foundations/ch03-probability/diagrams/expected_value_variance.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Created: expected_value_variance.png")

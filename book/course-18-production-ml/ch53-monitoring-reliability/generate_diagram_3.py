import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# Change to the chapter directory
os.chdir('/home/chirag/ds-book/book/course-18/ch53')

# Use consistent color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Load California Housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Split into baseline and current
X_baseline, X_current = train_test_split(X, test_size=0.3, random_state=42)

# Simulate drift by shifting MedInc
X_current_drifted = X_current.copy()
X_current_drifted['MedInc'] = X_current_drifted['MedInc'] + 2.0

# Visualize CDFs to show KS statistic and Wasserstein distance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# Get MedInc values
baseline_medinc = X_baseline['MedInc'].values
current_medinc = X_current_drifted['MedInc'].values

# Sort values for CDF plotting
baseline_sorted = np.sort(baseline_medinc)
current_sorted = np.sort(current_medinc)

# Empirical CDFs
baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
current_cdf = np.arange(1, len(current_sorted) + 1) / len(current_sorted)

# Plot CDFs with KS statistic
axes[0].plot(baseline_sorted, baseline_cdf, label='Baseline', color=colors['blue'], linewidth=2)
axes[0].plot(current_sorted, current_cdf, label='Current', color=colors['orange'], linewidth=2)

# Find and plot KS statistic (maximum vertical distance)
x_min = min(baseline_sorted.min(), current_sorted.min())
x_max = max(baseline_sorted.max(), current_sorted.max())
x_common = np.linspace(x_min, x_max, 1000)

baseline_cdf_interp = interp1d(baseline_sorted, baseline_cdf,
                                bounds_error=False, fill_value=(0, 1))
current_cdf_interp = interp1d(current_sorted, current_cdf,
                               bounds_error=False, fill_value=(0, 1))

cdf_diff = np.abs(baseline_cdf_interp(x_common) - current_cdf_interp(x_common))
max_diff_idx = np.argmax(cdf_diff)
max_diff_x = x_common[max_diff_idx]

axes[0].vlines(max_diff_x,
               min(baseline_cdf_interp(max_diff_x), current_cdf_interp(max_diff_x)),
               max(baseline_cdf_interp(max_diff_x), current_cdf_interp(max_diff_x)),
               colors=colors['red'], linewidth=3, label=f'KS Stat = {cdf_diff.max():.3f}')
axes[0].set_xlabel('MedInc', fontsize=12)
axes[0].set_ylabel('Cumulative Probability', fontsize=12)
axes[0].set_title('KS Test: Maximum Vertical Distance Between CDFs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(alpha=0.3)

# Plot CDFs with shaded area for Wasserstein distance
axes[1].plot(baseline_sorted, baseline_cdf, label='Baseline', color=colors['blue'], linewidth=2)
axes[1].plot(current_sorted, current_cdf, label='Current', color=colors['orange'], linewidth=2)
axes[1].fill_between(x_common,
                     baseline_cdf_interp(x_common),
                     current_cdf_interp(x_common),
                     alpha=0.3, color=colors['red'],
                     label=f'Wasserstein = {wasserstein_distance(baseline_medinc, current_medinc):.3f}')
axes[1].set_xlabel('MedInc', fontsize=12)
axes[1].set_ylabel('Cumulative Probability', fontsize=12)
axes[1].set_title('Wasserstein Distance: Area Between CDFs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/ks_wasserstein.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/ks_wasserstein.png")
plt.close()

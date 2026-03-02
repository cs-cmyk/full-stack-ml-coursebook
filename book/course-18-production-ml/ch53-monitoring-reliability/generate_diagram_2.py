import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
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
y = pd.Series(data.target, name='MedHouseValue')

# Split into "training" (baseline) and "production" (current) sets
X_baseline, X_current = train_test_split(X, test_size=0.3, random_state=42)

def calculate_psi(baseline, current, n_bins=10):
    """Calculate Population Stability Index between two distributions."""
    # Create bins based on baseline quantiles
    bin_edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)

    # Digitize both distributions
    baseline_binned = np.digitize(baseline, bin_edges[1:-1])
    current_binned = np.digitize(current, bin_edges[1:-1])

    # Calculate percentages in each bin
    baseline_counts = np.bincount(baseline_binned, minlength=n_bins)
    current_counts = np.bincount(current_binned, minlength=n_bins)

    baseline_pcts = baseline_counts / len(baseline)
    current_pcts = current_counts / len(current)

    # Add small constant to avoid division by zero
    epsilon = 0.0001
    baseline_pcts = baseline_pcts + epsilon
    current_pcts = current_pcts + epsilon

    # Calculate PSI
    psi = np.sum((current_pcts - baseline_pcts) * np.log(current_pcts / baseline_pcts))

    return psi, bin_edges, baseline_pcts, current_pcts

# Calculate PSI for each feature
psi_results = {}
for feature in X.columns:
    psi, bin_edges, baseline_pcts, current_pcts = calculate_psi(
        X_baseline[feature].values,
        X_current[feature].values
    )
    psi_results[feature] = {
        'psi': psi,
        'bin_edges': bin_edges,
        'baseline_pcts': baseline_pcts,
        'current_pcts': current_pcts
    }

# Get top drift feature
psi_df = pd.DataFrame([
    {'Feature': feature, 'PSI': results['psi']}
    for feature, results in psi_results.items()
]).sort_values('PSI', ascending=False)

# Visualize drift for the feature with highest PSI
top_drift_feature = psi_df.iloc[0]['Feature']
results = psi_results[top_drift_feature]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# Histogram comparison
axes[0].hist(X_baseline[top_drift_feature], bins=30, alpha=0.6,
             label='Baseline', color=colors['blue'], density=True)
axes[0].hist(X_current[top_drift_feature], bins=30, alpha=0.6,
             label='Current', color=colors['orange'], density=True)
axes[0].set_xlabel(top_drift_feature, fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title(f'Distribution Comparison: {top_drift_feature}', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(alpha=0.3)

# Bar chart of bin percentages
bin_indices = np.arange(len(results['baseline_pcts']))
width = 0.35
axes[1].bar(bin_indices - width/2, results['baseline_pcts'], width,
            label='Baseline', alpha=0.8, color=colors['blue'])
axes[1].bar(bin_indices + width/2, results['current_pcts'], width,
            label='Current', alpha=0.8, color=colors['orange'])
axes[1].set_xlabel('Bin', fontsize=12)
axes[1].set_ylabel('Percentage', fontsize=12)
axes[1].set_title(f'PSI Bin Comparison: {top_drift_feature}\nPSI = {results["psi"]:.6f}',
                  fontsize=14, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('diagrams/psi_example.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/psi_example.png")
plt.close()

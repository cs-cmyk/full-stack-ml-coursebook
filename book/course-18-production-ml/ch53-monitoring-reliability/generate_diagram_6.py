import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp, wasserstein_distance
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

np.random.seed(42)

# 1. Generate synthetic baseline (training) data
n_baseline = 10000
X_baseline, y_baseline = make_classification(
    n_samples=n_baseline, n_features=5, n_informative=4, n_redundant=0,
    n_clusters_per_class=2, class_sep=1.5, random_state=42
)

# Create meaningful feature names
feature_names = ['transaction_amount', 'merchant_category', 'time_of_day',
                 'user_age', 'days_since_signup']
df_baseline = pd.DataFrame(X_baseline, columns=feature_names)
df_baseline['is_fraud'] = y_baseline

# Train baseline model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(df_baseline[feature_names], df_baseline['is_fraud'])
baseline_accuracy = accuracy_score(df_baseline['is_fraud'],
                                   model.predict(df_baseline[feature_names]))

# 2. Generate current data with intentional drift
n_current = 5000

# Start with similar data
X_current, y_current = make_classification(
    n_samples=n_current, n_features=5, n_informative=4, n_redundant=0,
    n_clusters_per_class=2, class_sep=1.5, random_state=123
)

df_current = pd.DataFrame(X_current, columns=feature_names)

# Apply data drift: increase transaction_amount by 30%
df_current['transaction_amount'] = df_current['transaction_amount'] * 1.3

# Shift merchant_category distribution
df_current['merchant_category'] = df_current['merchant_category'] + 0.5

# Apply concept drift: rotate relationship between features and fraud
fraud_score = (
    df_current['transaction_amount'] * 0.5 +
    df_current['time_of_day'] * 0.3 -
    df_current['user_age'] * 0.2
)
y_current = (fraud_score > fraud_score.median()).astype(int)
df_current['is_fraud'] = y_current

# Evaluate model on current data
current_predictions = model.predict(df_current[feature_names])
current_accuracy = accuracy_score(df_current['is_fraud'], current_predictions)

# 3. Calculate drift metrics for all features
def calculate_psi(baseline, current, n_bins=10):
    bin_edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    baseline_binned = np.digitize(baseline, bin_edges[1:-1])
    current_binned = np.digitize(current, bin_edges[1:-1])
    baseline_counts = np.bincount(baseline_binned, minlength=n_bins)
    current_counts = np.bincount(current_binned, minlength=n_bins)
    baseline_pcts = baseline_counts / len(baseline) + 0.0001
    current_pcts = current_counts / len(current) + 0.0001
    psi = np.sum((current_pcts - baseline_pcts) * np.log(current_pcts / baseline_pcts))
    return psi

drift_metrics = []
for feature in feature_names:
    baseline_vals = df_baseline[feature].values
    current_vals = df_current[feature].values

    psi = calculate_psi(baseline_vals, current_vals)
    ks_stat, ks_pval = ks_2samp(baseline_vals, current_vals)
    wass = wasserstein_distance(baseline_vals, current_vals)

    drift_metrics.append({
        'Feature': feature,
        'PSI': psi,
        'KS_Stat': ks_stat,
        'KS_pval': ks_pval,
        'Wasserstein': wass
    })

drift_df = pd.DataFrame(drift_metrics).sort_values('PSI', ascending=False)

# 4. Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

# (a) Feature distributions for top 2 drifted features
for idx, feature in enumerate(drift_df['Feature'].head(2)):
    ax = axes[0, idx]
    ax.hist(df_baseline[feature], bins=30, alpha=0.6, label='Baseline',
            density=True, color=colors['blue'])
    ax.hist(df_current[feature], bins=30, alpha=0.6, label='Current',
            density=True, color=colors['orange'])
    psi_val = drift_df[drift_df['Feature'] == feature]['PSI'].values[0]
    ax.set_title(f'{feature}\nPSI = {psi_val:.3f}', fontweight='bold', fontsize=12)
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# (b) Drift metrics comparison
ax = axes[1, 0]
x = np.arange(len(feature_names))
width = 0.25
ax.bar(x - width, drift_df['PSI'], width, label='PSI', alpha=0.8, color=colors['blue'])
ax.bar(x, drift_df['KS_Stat'], width, label='KS Stat', alpha=0.8, color=colors['orange'])
# Normalize Wasserstein for visualization
wass_norm = drift_df['Wasserstein'] / drift_df['Wasserstein'].max()
ax.bar(x + width, wass_norm, width, label='Wasserstein (norm)', alpha=0.8, color=colors['green'])
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Drift Score', fontsize=12)
ax.set_title('Drift Metrics Comparison', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(drift_df['Feature'], rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# (c) Model performance over time (simulated)
time_steps = 30
accuracies = []
fraud_rates = []

for t in range(time_steps):
    # Gradual drift: interpolate between baseline and current
    alpha = t / time_steps
    X_mixed = (1 - alpha) * X_baseline[:n_current] + alpha * X_current
    # Gradual concept drift
    if t < 15:
        y_mixed = y_baseline[:n_current]
    else:
        transition = (t - 15) / 15
        y_mixed = ((1 - transition) * y_baseline[:n_current] +
                   transition * y_current).astype(int)

    preds = model.predict(X_mixed)
    acc = accuracy_score(y_mixed, preds)
    accuracies.append(acc)
    fraud_rates.append(y_mixed.mean())

ax = axes[1, 1]
ax.plot(accuracies, linewidth=2.5, color=colors['blue'], label='Accuracy')
ax.axhline(y=0.85, color=colors['red'], linestyle='--', linewidth=2,
           label='Alert Threshold')
ax.set_xlabel('Time Step (days)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Performance Degradation Over Time', fontweight='bold', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig('diagrams/exercise1_drift_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/exercise1_drift_analysis.png")
plt.close()

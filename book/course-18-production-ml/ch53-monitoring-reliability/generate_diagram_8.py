import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
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

# 1. Generate time-series dataset with daily snapshots over 60 days
n_samples_per_day = 500
n_days = 60
feature_names = ['user_tenure', 'session_duration', 'cart_value',
                 'items_viewed', 'conversion']

all_data = []

for day in range(1, n_days + 1):
    # Base distributions
    user_tenure = np.random.gamma(shape=2, scale=5, size=n_samples_per_day)
    session_duration = np.random.lognormal(mean=4, sigma=1, size=n_samples_per_day)
    cart_value = np.random.gamma(shape=3, scale=20, size=n_samples_per_day)
    items_viewed = np.random.poisson(lam=5, size=n_samples_per_day)

    # 3. Introduce gradual drift starting at day 40
    if day >= 40:
        days_since_drift = day - 40
        # Increase cart_value by 2% per day
        cart_value = cart_value * (1 + 0.02 * days_since_drift)
        # Decrease session_duration by 1% per day
        session_duration = session_duration * (1 - 0.01 * days_since_drift)

    # Generate conversion based on features
    conversion_prob = 1 / (1 + np.exp(-(
        -3 +
        0.02 * user_tenure +
        0.0001 * session_duration +
        0.01 * cart_value +
        0.1 * items_viewed
    )))
    conversion = np.random.binomial(1, conversion_prob)

    day_data = pd.DataFrame({
        'day': day,
        'user_tenure': user_tenure,
        'session_duration': session_duration,
        'cart_value': cart_value,
        'items_viewed': items_viewed,
        'conversion': conversion
    })
    all_data.append(day_data)

df_full = pd.concat(all_data, ignore_index=True)

# 2. Designate baseline and monitoring periods
baseline_data = df_full[df_full['day'] <= 30]
monitoring_days = range(31, n_days + 1)

# Simple drift detection using KS test
def detect_drift(baseline, current, feature_list):
    """Simplified drift detection using KS test"""
    drift_scores = {}
    drift_detected_flags = {}

    for feature in feature_list:
        if feature == 'conversion':
            continue
        ks_stat, p_value = ks_2samp(baseline[feature], current[feature])
        drift_scores[feature] = ks_stat
        drift_detected_flags[feature] = p_value < 0.05

    drift_share = sum(drift_detected_flags.values()) / len(drift_detected_flags)
    n_drifted = sum(drift_detected_flags.values())

    return drift_share, n_drifted, drift_scores, drift_detected_flags

# 4 & 5. Daily drift detection with rolling 7-day windows
monitoring_results = []

for current_day in monitoring_days:
    # Get last 7 days of data ending on current_day
    start_day = max(31, current_day - 6)
    current_window = df_full[(df_full['day'] >= start_day) &
                             (df_full['day'] <= current_day)]

    # Run drift detection
    drift_share, n_drifted, drift_scores, drift_flags = detect_drift(
        baseline_data, current_window, feature_names
    )

    # Calculate model performance on current window
    current_conversion_rate = current_window['conversion'].mean()
    baseline_conversion_rate = baseline_data['conversion'].mean()
    performance_drop = baseline_conversion_rate - current_conversion_rate

    monitoring_results.append({
        'day': current_day,
        'drift_share': drift_share,
        'n_drifted': n_drifted,
        'cart_value_drift': drift_flags.get('cart_value', False),
        'cart_value_score': drift_scores.get('cart_value', 0),
        'session_duration_drift': drift_flags.get('session_duration', False),
        'session_duration_score': drift_scores.get('session_duration', 0),
        'conversion_rate': current_conversion_rate,
        'performance_drop': performance_drop
    })

monitoring_df = pd.DataFrame(monitoring_results)

# 6. Implement alerting logic with three severity levels
def determine_alert_level(row):
    if row['drift_share'] > 0.6 or row['performance_drop'] > 0.05:
        return 'CRITICAL'
    elif row['drift_share'] > 0.4:
        return 'WARNING'
    elif row['drift_share'] > 0.2:
        return 'INFO'
    else:
        return 'OK'

monitoring_df['alert_level'] = monitoring_df.apply(determine_alert_level, axis=1)

# Create comprehensive dashboard
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.patch.set_facecolor('white')

# (a) Drift share over time
axes[0].plot(monitoring_df['day'], monitoring_df['drift_share'],
            linewidth=2.5, color=colors['blue'], marker='o', markersize=5)
axes[0].axhline(y=0.2, color=colors['green'], linestyle='--', linewidth=2,
               label='INFO threshold')
axes[0].axhline(y=0.4, color=colors['orange'], linestyle='--', linewidth=2,
               label='WARNING threshold')
axes[0].axhline(y=0.6, color=colors['red'], linestyle='--', linewidth=2,
               label='CRITICAL threshold')
axes[0].axvline(x=40, color=colors['purple'], linestyle=':', linewidth=2.5,
               label='Drift onset (Day 40)')
axes[0].set_xlabel('Day', fontsize=12)
axes[0].set_ylabel('Drift Share', fontsize=12)
axes[0].set_title('Dataset Drift Share Over Time', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0, 1])

# (b) Per-feature drift scores
axes[1].plot(monitoring_df['day'], monitoring_df['cart_value_score'],
            linewidth=2.5, marker='o', label='cart_value', color=colors['red'], markersize=5)
axes[1].plot(monitoring_df['day'], monitoring_df['session_duration_score'],
            linewidth=2.5, marker='s', label='session_duration', color=colors['blue'], markersize=5)
axes[1].axvline(x=40, color=colors['purple'], linestyle=':', linewidth=2.5,
               label='Drift onset')
axes[1].set_xlabel('Day', fontsize=12)
axes[1].set_ylabel('Drift Score (KS Statistic)', fontsize=12)
axes[1].set_title('Feature-Level Drift Scores Over Time', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

# (c) Model performance over time
baseline_conv = baseline_data['conversion'].mean()
axes[2].plot(monitoring_df['day'], monitoring_df['conversion_rate'],
            linewidth=2.5, color=colors['green'], marker='o', label='Conversion Rate', markersize=5)
axes[2].axhline(y=baseline_conv, color=colors['gray'], linestyle='--', linewidth=2,
               label=f'Baseline ({baseline_conv:.3f})')
axes[2].axhline(y=baseline_conv * 0.95, color=colors['red'], linestyle='--', linewidth=2,
               label='5% drop threshold')
axes[2].axvline(x=40, color=colors['purple'], linestyle=':', linewidth=2.5,
               label='Drift onset')
axes[2].set_xlabel('Day', fontsize=12)
axes[2].set_ylabel('Conversion Rate', fontsize=12)
axes[2].set_title('Model Performance (Conversion Rate) Over Time', fontweight='bold', fontsize=14)
axes[2].legend(fontsize=11)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/exercise3_monitoring_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/exercise3_monitoring_dashboard.png")
plt.close()

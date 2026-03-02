"""
Code Review Test Script for Chapter 16: Time Features
Tests all code blocks from content.md in sequence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BLOCK 1: Time Features Visualization")
print("="*80)

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))

# 1. Linear vs Cyclical Encoding
ax1 = plt.subplot(2, 3, 1)
hours_linear = np.arange(24)
ax1.plot(hours_linear, hours_linear, 'o-', linewidth=2, markersize=8, color='steelblue')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.axhline(y=23, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(12, 11.5, 'Distance = 23', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.set_xlabel('Hour (Linear)', fontsize=11)
ax1.set_ylabel('Numeric Value', fontsize=11)
ax1.set_title('A. Linear Encoding Problem', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Cyclical Encoding Circle
ax2 = plt.subplot(2, 3, 2, projection='polar')
hours = np.arange(24)
angles = 2 * np.pi * hours / 24
colors = plt.cm.twilight(hours / 24)
ax2.scatter(angles, np.ones(24), c=colors, s=100, zorder=3)
for i, hour in enumerate([0, 6, 12, 18, 23]):
    idx = np.where(hours == hour)[0][0]
    ax2.annotate(f'{hour}h', xy=(angles[idx], 1), xytext=(angles[idx], 1.15),
                ha='center', fontsize=10, fontweight='bold')
ax2.set_ylim(0, 1.3)
ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_title('B. Cyclical Encoding (Circle)', fontsize=12, fontweight='bold', pad=20)

# 3. Sine and Cosine Waves
ax3 = plt.subplot(2, 3, 3)
hours_extended = np.linspace(0, 48, 200)
hour_sin = np.sin(2 * np.pi * hours_extended / 24)
hour_cos = np.cos(2 * np.pi * hours_extended / 24)
ax3.plot(hours_extended, hour_sin, label='sin(2π·h/24)', linewidth=2, color='coral')
ax3.plot(hours_extended, hour_cos, label='cos(2π·h/24)', linewidth=2, color='teal')
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax3.axvline(x=23, color='red', linestyle='--', alpha=0.5)
ax3.text(11.5, 0.5, 'Hour 0 and 23\nare close!', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax3.set_xlabel('Hour', fontsize=11)
ax3.set_ylabel('Feature Value', fontsize=11)
ax3.set_title('C. Sine-Cosine Transformation', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# 4. Lag Features Timeline
ax4 = plt.subplot(2, 3, 4)
days = np.arange(20)
np.random.seed(42)  # For reproducibility
values = 50 + 10*np.sin(2*np.pi*days/7) + np.random.normal(0, 2, 20)
ax4.plot(days, values, 'o-', label='Original', linewidth=2, markersize=6, color='blue')
ax4.plot(days, np.roll(values, 1), 'o--', label='Lag-1 (shift 1)',
         linewidth=2, markersize=5, alpha=0.7, color='green')
ax4.plot(days, np.roll(values, 7), 's--', label='Lag-7 (shift 7)',
         linewidth=2, markersize=5, alpha=0.7, color='orange')
# Highlight prediction point
ax4.axvline(x=10, color='red', linestyle=':', linewidth=2)
ax4.text(10.5, 65, 'Predict\nhere', fontsize=10, color='red', fontweight='bold')
ax4.set_xlabel('Day', fontsize=11)
ax4.set_ylabel('Value', fontsize=11)
ax4.set_title('D. Lag Features', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Rolling Window
ax5 = plt.subplot(2, 3, 5)
days2 = np.arange(30)
np.random.seed(42)  # For reproducibility
values2 = 50 + 15*np.sin(2*np.pi*days2/10) + np.random.normal(0, 5, 30)
rolling_7 = np.convolve(values2, np.ones(7)/7, mode='same')
ax5.plot(days2, values2, 'o-', label='Original (noisy)', linewidth=1.5,
         markersize=4, alpha=0.6, color='gray')
ax5.plot(days2, rolling_7, linewidth=3, label='Rolling 7-day mean', color='blue')
# Show window
window_start = 15
ax5.axvspan(window_start-3, window_start+3, alpha=0.2, color='yellow')
ax5.text(window_start, 25, 'Window', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax5.set_xlabel('Day', fontsize=11)
ax5.set_ylabel('Value', fontsize=11)
ax5.set_title('E. Rolling Window Features', fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Temporal Train/Test Split
ax6 = plt.subplot(2, 3, 6)
ax6.barh([1], [75], left=[0], height=0.3, color='green', alpha=0.7, label='Training (past)')
ax6.barh([1], [25], left=[75], height=0.3, color='orange', alpha=0.7, label='Test (future)')
ax6.barh([0], [50], left=[0], height=0.3, color='red', alpha=0.3)
ax6.barh([0], [50], left=[50], height=0.3, color='red', alpha=0.3)
ax6.text(50, 0, '✗ RANDOM SPLIT\n(leakage!)', ha='center', va='center',
         fontsize=11, fontweight='bold', color='darkred')
ax6.text(37.5, 1, 'TRAIN', ha='center', va='center', fontsize=11, fontweight='bold')
ax6.text(87.5, 1, 'TEST', ha='center', va='center', fontsize=11, fontweight='bold')
ax6.arrow(37.5, 1.5, 45, 0, head_width=0.1, head_length=3, fc='black', ec='black')
ax6.text(60, 1.7, 'Time flows →', ha='center', fontsize=10)
ax6.set_xlim(0, 100)
ax6.set_ylim(-0.5, 2)
ax6.set_xlabel('Data Timeline (%)', fontsize=11)
ax6.set_yticks([0, 1])
ax6.set_yticklabels(['Wrong', 'Correct'])
ax6.set_title('F. Temporal Split (No Leakage)', fontsize=12, fontweight='bold')
ax6.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/time_features_overview.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Block 1 passed: Visualization created successfully")

print("\n" + "="*80)
print("BLOCK 2: Complete Time Feature Engineering Example")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic hourly sales data with realistic patterns
def generate_sales_data(n_days=365):
    """Generate synthetic hourly sales data with temporal patterns"""
    hours = pd.date_range('2023-01-01', periods=n_days*24, freq='h')
    df = pd.DataFrame({'timestamp': hours})

    # Extract time components for pattern generation
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Base sales pattern
    base_sales = 100

    # Hourly pattern (morning and afternoon peaks)
    hourly_pattern = 30 * np.sin(2 * np.pi * (df['hour'] - 6) / 12)

    # Weekly pattern (weekends busier)
    weekly_pattern = 20 * (df['day_of_week'].isin([5, 6])).astype(int)

    # Monthly/seasonal pattern (summer higher)
    seasonal_pattern = 15 * np.sin(2 * np.pi * (df['month'] - 3) / 12)

    # Trend (gradual growth over year)
    trend = np.linspace(0, 20, len(df))

    # Random noise
    noise = np.random.normal(0, 10, len(df))

    # Combine all patterns
    df['sales'] = (base_sales + hourly_pattern + weekly_pattern +
                   seasonal_pattern + trend + noise)
    df['sales'] = df['sales'].clip(lower=0)  # No negative sales

    return df[['timestamp', 'sales']]

# Generate data
df_raw = generate_sales_data(n_days=365)
print("Generated sales data:")
print(df_raw.head())
print(f"\nShape: {df_raw.shape}")
print(f"Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")

# 1. BASIC TEMPORAL FEATURE EXTRACTION
df = df_raw.copy()

# Extract datetime components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['week_of_year'] = df['timestamp'].dt.isocalendar().week

# Derived features
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_morning'] = (df['hour'] >= 6) & (df['hour'] < 12)
df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 18)
df['is_evening'] = (df['hour'] >= 18) & (df['hour'] < 24)
df['is_night'] = (df['hour'] >= 0) & (df['hour'] < 6)

print("\nBasic temporal features extracted:")
print(df[['timestamp', 'sales', 'hour', 'day_of_week', 'is_weekend']].head(10))

# 2. CYCLICAL ENCODING
# Hour (24-hour cycle)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Day of week (7-day cycle)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Month (12-month cycle)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print("\nCyclical features created:")
print(df[['hour', 'hour_sin', 'hour_cos']].head(10))
print(f"\nHour 0: sin={df.loc[0, 'hour_sin']:.3f}, cos={df.loc[0, 'hour_cos']:.3f}")
print(f"Hour 23: sin={df.loc[23, 'hour_sin']:.3f}, cos={df.loc[23, 'hour_cos']:.3f}")
print("Note: Hour 0 and 23 are close in cyclical space!")

# 3. LAG FEATURES
# Create lag features: yesterday same hour, last week same hour
df['sales_lag_1'] = df['sales'].shift(1)      # 1 hour ago
df['sales_lag_24'] = df['sales'].shift(24)    # Yesterday same hour
df['sales_lag_168'] = df['sales'].shift(168)  # Last week same hour

print("\nLag features created:")
print(df[['timestamp', 'sales', 'sales_lag_1', 'sales_lag_24', 'sales_lag_168']].iloc[168:172])

# 4. ROLLING WINDOW FEATURES
# Rolling statistics: daily and weekly averages
df['sales_rolling_24_mean'] = df['sales'].rolling(window=24, min_periods=1).mean()
df['sales_rolling_24_std'] = df['sales'].rolling(window=24, min_periods=1).std()
df['sales_rolling_168_mean'] = df['sales'].rolling(window=168, min_periods=1).mean()

print("\nRolling features created:")
print(df[['timestamp', 'sales', 'sales_rolling_24_mean', 'sales_rolling_168_mean']].iloc[168:172])

# 5. PREPARE FOR MODELING
# Drop rows with NaN (from lags and rolling features)
df_clean = df.dropna().reset_index(drop=True)
print(f"\nRows after dropping NaN: {len(df_clean)} (started with {len(df)})")

# Select features for modeling
feature_cols = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos',
    'sales_lag_1', 'sales_lag_24', 'sales_lag_168',
    'sales_rolling_24_mean', 'sales_rolling_24_std', 'sales_rolling_168_mean'
]

X = df_clean[feature_cols]
y = df_clean['sales']

# 6. TEMPORAL TRAIN/TEST SPLIT (Critical: no random split!)
# Use first 80% for training, last 20% for testing
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTemporal split:")
print(f"Training set: {len(X_train)} samples (first 80%)")
print(f"Test set: {len(X_test)} samples (last 20%)")
print(f"Training period: {df_clean.loc[0, 'timestamp']} to {df_clean.loc[split_idx-1, 'timestamp']}")
print(f"Test period: {df_clean.loc[split_idx, 'timestamp']} to {df_clean.loc[len(df_clean)-1, 'timestamp']}")

# 7. TRAIN MODEL
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 8. EVALUATE
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\n{'='*50}")
print("MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"Training R²:  {train_r2:.4f} | RMSE: {train_rmse:.2f}")
print(f"Test R²:      {test_r2:.4f} | RMSE: {test_rmse:.2f}")

# 9. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# 10. VISUALIZE PREDICTIONS
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot last week of test set
test_week_idx = slice(-168, None)  # Last 168 hours = 1 week
timestamps_test = df_clean.loc[split_idx:, 'timestamp'].values[test_week_idx]
y_test_week = y_test.values[test_week_idx]
y_pred_week = y_pred_test[test_week_idx]

axes[0].plot(timestamps_test, y_test_week, label='Actual Sales',
             linewidth=2, alpha=0.7, color='blue')
axes[0].plot(timestamps_test, y_pred_week, label='Predicted Sales',
             linewidth=2, alpha=0.7, color='red', linestyle='--')
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Sales', fontsize=12)
axes[0].set_title('Actual vs Predicted Sales (Last Week of Test Set)', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=11)
axes[0].grid(True, alpha=0.3)

# Residuals plot
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.3, s=10)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Sales', fontsize=12)
axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/time_features_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to diagrams/time_features_predictions.png")
print("✓ Block 2 passed: Complete time feature engineering example ran successfully")

print("\n" + "="*80)
print("SUMMARY: All code blocks executed successfully!")
print("="*80)

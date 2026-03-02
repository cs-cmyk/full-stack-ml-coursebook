import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'
PURPLE = '#9C27B0'
GRAY = '#607D8B'

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

# Feature engineering
df = df_raw.copy()

# Extract datetime components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['week_of_year'] = df['timestamp'].dt.isocalendar().week

# Derived features
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Lag features
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_lag_24'] = df['sales'].shift(24)
df['sales_lag_168'] = df['sales'].shift(168)

# Rolling window features
df['sales_rolling_24_mean'] = df['sales'].rolling(window=24, min_periods=1).mean()
df['sales_rolling_24_std'] = df['sales'].rolling(window=24, min_periods=1).std()
df['sales_rolling_168_mean'] = df['sales'].rolling(window=168, min_periods=1).mean()

# Drop rows with NaN
df_clean = df.dropna().reset_index(drop=True)

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

# Temporal train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Training R²:  {train_r2:.4f} | RMSE: {train_rmse:.2f}")
print(f"Test R²:      {test_r2:.4f} | RMSE: {test_rmse:.2f}")

# Visualize predictions
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot last week of test set
test_week_idx = slice(-168, None)  # Last 168 hours = 1 week
timestamps_test = df_clean.loc[split_idx:, 'timestamp'].values[test_week_idx]
y_test_week = y_test.values[test_week_idx]
y_pred_week = y_pred_test[test_week_idx]

axes[0].plot(timestamps_test, y_test_week, label='Actual Sales',
             linewidth=2, alpha=0.8, color=BLUE)
axes[0].plot(timestamps_test, y_pred_week, label='Predicted Sales',
             linewidth=2, alpha=0.8, color=RED, linestyle='--')
axes[0].set_xlabel('Date', fontsize=13)
axes[0].set_ylabel('Sales', fontsize=13)
axes[0].set_title('Actual vs Predicted Sales (Last Week of Test Set)', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Residuals plot
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, alpha=0.3, s=10, color=PURPLE)
axes[1].axhline(y=0, color=RED, linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Sales', fontsize=13)
axes[1].set_ylabel('Residuals (Actual - Predicted)', fontsize=13)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('time_features_predictions.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: time_features_predictions.png")

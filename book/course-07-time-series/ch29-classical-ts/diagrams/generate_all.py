"""
Generate all diagrams for Chapter 29: Classical Time Series Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.datasets import get_rdataset
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B',
    'dark_blue': '#2E86AB',
    'dark_purple': '#A23B72',
    'dark_green': '#6A994E'
}

# Load Air Passengers dataset
air_passengers = get_rdataset('AirPassengers')
df = air_passengers.data
df['time'] = pd.date_range(start='1949-01', periods=len(df), freq='M')
df = df.set_index('time')
df = df.rename(columns={'value': 'passengers'})

print("Generating diagrams for Chapter 29...")
print(f"Loaded dataset: {df.shape}")

# ============================================================================
# DIAGRAM 1: Components (Main visualization - Figure 29.1)
# ============================================================================
print("\n1. Generating components.png...")

# Perform STL decomposition
stl = STL(df['passengers'], seasonal=13, trend=15)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Original series
axes[0].plot(df.index, df['passengers'], color=COLORS['blue'], linewidth=1.5)
axes[0].set_ylabel('Original\nSeries', fontweight='bold', fontsize=12)
axes[0].set_title('Time Series Decomposition: Air Passengers (1949-1960)',
                   fontsize=14, fontweight='bold', pad=20)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=11)

# Trend component
axes[1].plot(df.index, result.trend, color=COLORS['purple'], linewidth=2)
axes[1].set_ylabel('Trend\nComponent', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

# Seasonal component
axes[2].plot(df.index, result.seasonal, color=COLORS['orange'], linewidth=1.5)
axes[2].set_ylabel('Seasonal\nComponent', fontweight='bold', fontsize=12)
axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(labelsize=11)

# Residual component
axes[3].plot(df.index, result.resid, color=COLORS['dark_green'], linewidth=1)
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
axes[3].set_ylabel('Residual\nComponent', fontweight='bold', fontsize=12)
axes[3].set_xlabel('Year', fontweight='bold', fontsize=12)
axes[3].grid(True, alpha=0.3)
axes[3].tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('components.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ components.png saved")

# ============================================================================
# DIAGRAM 2: Raw Series
# ============================================================================
print("2. Generating raw_series.png...")

plt.figure(figsize=(12, 4))
plt.plot(df.index, df['passengers'], linewidth=1.5, color=COLORS['blue'])
plt.title('Air Passengers: Monthly Totals (1949-1960)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12, fontweight='bold')
plt.ylabel('Number of Passengers (thousands)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('raw_series.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ raw_series.png saved")

# ============================================================================
# DIAGRAM 3: Decomposition (detailed version)
# ============================================================================
print("3. Generating decomposition.png...")

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Original series
axes[0].plot(df.index, df['passengers'], color=COLORS['dark_blue'], linewidth=1.5)
axes[0].set_ylabel('Original', fontweight='bold', fontsize=12)
axes[0].set_title('Time Series Decomposition: Air Passengers',
                   fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=11)

# Trend component
axes[1].plot(df.index, result.trend, color=COLORS['dark_purple'], linewidth=2)
axes[1].set_ylabel('Trend', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=11)

# Seasonal component
axes[2].plot(df.index, result.seasonal, color=COLORS['orange'], linewidth=1.5)
axes[2].set_ylabel('Seasonal', fontweight='bold', fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(labelsize=11)

# Residual component
axes[3].plot(df.index, result.resid, color=COLORS['dark_green'], linewidth=1)
axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[3].set_ylabel('Residual', fontweight='bold', fontsize=12)
axes[3].set_xlabel('Year', fontweight='bold', fontsize=12)
axes[3].grid(True, alpha=0.3)
axes[3].tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('decomposition.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ decomposition.png saved")

# ============================================================================
# DIAGRAM 4: Stationarity Process
# ============================================================================
print("4. Generating stationarity_process.png...")

# Calculate differenced series
df['diff1'] = df['passengers'].diff()
df['diff_seasonal'] = df['passengers'].diff(12)
df['diff_both'] = df['passengers'].diff(12).diff()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(df.index, df['passengers'], color=COLORS['dark_blue'], linewidth=1.5)
axes[0, 0].set_title('Original Series (Non-Stationary)', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Passengers', fontweight='bold', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(labelsize=10)

axes[0, 1].plot(df.index, df['diff1'], color=COLORS['dark_purple'], linewidth=1.5)
axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_title('First-Order Differenced (Borderline)', fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Δ Passengers', fontweight='bold', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(labelsize=10)

axes[1, 0].plot(df.index, df['diff_seasonal'], color=COLORS['orange'], linewidth=1.5)
axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Seasonally Differenced (Still Non-Stationary)', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Δ₁₂ Passengers', fontweight='bold', fontsize=11)
axes[1, 0].set_xlabel('Year', fontweight='bold', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(labelsize=10)

axes[1, 1].plot(df.index, df['diff_both'], color=COLORS['dark_green'], linewidth=1.5)
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('First-Order + Seasonal Differenced (Stationary)', fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Δ₁Δ₁₂ Passengers', fontweight='bold', fontsize=11)
axes[1, 1].set_xlabel('Year', fontweight='bold', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('stationarity_process.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ stationarity_process.png saved")

# ============================================================================
# DIAGRAM 5: ACF and PACF
# ============================================================================
print("5. Generating acf_pacf.png...")

# Use the stationary series
stationary_series = df['diff_both'].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# ACF plot
plot_acf(stationary_series, lags=40, ax=axes[0], alpha=0.05)
axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Lag', fontsize=12)
axes[0].set_ylabel('Autocorrelation', fontsize=12)
axes[0].tick_params(labelsize=11)

# PACF plot
plot_pacf(stationary_series, lags=40, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Lag', fontsize=12)
axes[1].set_ylabel('Partial Autocorrelation', fontsize=12)
axes[1].tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('acf_pacf.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ acf_pacf.png saved")

# ============================================================================
# DIAGRAM 6: SARIMA Forecast
# ============================================================================
print("6. Generating sarima_forecast.png...")

# Split data
train_size = 120
train = df['passengers'][:train_size]
test = df['passengers'][train_size:]

# Fit best SARIMA model
best_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False, maxiter=200)

best_forecast = best_model.forecast(steps=len(test))
forecast_se = best_model.get_forecast(steps=len(test)).se_mean
lower_ci = best_forecast - 1.96 * forecast_se
upper_ci = best_forecast + 1.96 * forecast_se

plt.figure(figsize=(14, 5))

# Plot training data
plt.plot(train.index, train, label='Training Data', color=COLORS['dark_blue'], linewidth=1.5)

# Plot test data
plt.plot(test.index, test, label='Actual Test Data', color=COLORS['dark_purple'],
         linewidth=2, marker='o', markersize=4)

# Plot forecast
plt.plot(test.index, best_forecast, label='SARIMA(1,1,1)(1,1,1)₁₂ Forecast',
         color=COLORS['orange'], linewidth=2, linestyle='--', marker='s', markersize=4)

# Add confidence intervals
plt.fill_between(test.index, lower_ci, upper_ci,
                 color=COLORS['orange'], alpha=0.2, label='95% Confidence Interval')

plt.axvline(x=train.index[-1], color='black', linestyle=':', linewidth=2, alpha=0.5)
plt.text(train.index[-1], 600, 'Train/Test Split', rotation=90,
         verticalalignment='top', fontsize=11)

plt.title('SARIMA Forecast vs Actual: Air Passengers', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontweight='bold', fontsize=12)
plt.ylabel('Number of Passengers (thousands)', fontweight='bold', fontsize=12)
plt.legend(loc='upper left', framealpha=0.9, fontsize=11)
plt.grid(True, alpha=0.3)
plt.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('sarima_forecast.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ sarima_forecast.png saved")

# ============================================================================
# DIAGRAM 7: Exponential Smoothing Comparison
# ============================================================================
print("7. Generating exponential_smoothing.png...")

# Fit exponential smoothing models
ses_model = ExponentialSmoothing(train, trend=None, seasonal=None).fit()
holt_model = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
hw_add_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
hw_mul_model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12).fit()

ses_forecast = ses_model.forecast(steps=len(test))
holt_forecast = holt_model.forecast(steps=len(test))
hw_add_forecast = hw_add_model.forecast(steps=len(test))
hw_mul_forecast = hw_mul_model.forecast(steps=len(test))

ses_mae = mean_absolute_error(test, ses_forecast)
holt_mae = mean_absolute_error(test, holt_forecast)
hw_add_mae = mean_absolute_error(test, hw_add_forecast)
hw_mul_mae = mean_absolute_error(test, hw_mul_forecast)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models_es = [
    ('Simple Exponential Smoothing', ses_forecast, ses_mae, '#E63946'),
    ("Holt's Linear Method", holt_forecast, holt_mae, '#F4A261'),
    ('Holt-Winters Additive', hw_add_forecast, hw_add_mae, '#2A9D8F'),
    ('Holt-Winters Multiplicative', hw_mul_forecast, hw_mul_mae, '#264653')
]

for idx, (name, forecast, mae, color) in enumerate(models_es):
    ax = axes[idx // 2, idx % 2]

    # Plot training and test data
    ax.plot(train.index[-36:], train[-36:], label='Training',
            color=COLORS['dark_blue'], linewidth=1.5, alpha=0.7)
    ax.plot(test.index, test, label='Actual', color=COLORS['dark_purple'],
            linewidth=2, marker='o', markersize=4)

    # Plot forecast
    ax.plot(test.index, forecast, label='Forecast', color=color,
            linewidth=2, linestyle='--', marker='s', markersize=4)

    ax.axvline(x=train.index[-1], color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_title(f'{name}\nMAE: {mae:.2f}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Passengers', fontweight='bold', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

axes[1, 0].set_xlabel('Year', fontweight='bold', fontsize=11)
axes[1, 1].set_xlabel('Year', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('exponential_smoothing.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ exponential_smoothing.png saved")

# ============================================================================
# DIAGRAM 8: Final Comparison
# ============================================================================
print("8. Generating final_comparison.png...")

plt.figure(figsize=(14, 6))

# Plot training data (last 36 months for clarity)
plt.plot(train.index[-36:], train[-36:], label='Training Data',
         color=COLORS['dark_blue'], linewidth=1.5, alpha=0.7)

# Plot actual test data
plt.plot(test.index, test, label='Actual Test Data', color='#000000',
         linewidth=2.5, marker='o', markersize=6, zorder=10)

# Plot SARIMA forecast
plt.plot(test.index, best_forecast, label='SARIMA(1,1,1)(1,1,1)₁₂',
         color=COLORS['orange'], linewidth=2, linestyle='--', marker='s', markersize=5)

# Plot Holt-Winters forecast
plt.plot(test.index, hw_mul_forecast, label='Holt-Winters Multiplicative',
         color=COLORS['dark_green'], linewidth=2, linestyle='--', marker='^', markersize=5)

# Add vertical line at train/test split
plt.axvline(x=train.index[-1], color='red', linestyle=':', linewidth=2, alpha=0.7)
plt.text(train.index[-1], 650, 'Train/Test Split', rotation=90,
         verticalalignment='top', fontsize=12, fontweight='bold')

plt.title('Model Comparison: SARIMA vs Holt-Winters', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontweight='bold', fontsize=12)
plt.ylabel('Number of Passengers (thousands)', fontweight='bold', fontsize=12)
plt.legend(loc='upper left', framealpha=0.95, fontsize=11)
plt.grid(True, alpha=0.3)
plt.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ final_comparison.png saved")

print("\n" + "="*60)
print("ALL DIAGRAMS GENERATED SUCCESSFULLY!")
print("="*60)
print("\nGenerated diagrams:")
print("  1. components.png - Main decomposition visualization")
print("  2. raw_series.png - Original time series")
print("  3. decomposition.png - Detailed decomposition")
print("  4. stationarity_process.png - Differencing transformations")
print("  5. acf_pacf.png - Autocorrelation analysis")
print("  6. sarima_forecast.png - SARIMA model forecast")
print("  7. exponential_smoothing.png - Exponential smoothing comparison")
print("  8. final_comparison.png - Final model comparison")
print("="*60)

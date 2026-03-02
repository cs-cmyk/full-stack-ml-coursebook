"""
Generate all diagrams for Chapter 18: Linear Regression
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for consistency
plt.style.use('default')
plt.rcParams['font.size'] = 12

# Create diagrams directory if it doesn't exist
import os
os.makedirs('diagrams', exist_ok=True)

print("Generating diagrams for Linear Regression chapter...")
print("=" * 70)

# ============================================================================
# DIAGRAM 1: Regression Line with Residuals (Basic Concept)
# ============================================================================
print("\n1. Generating regression_line_residuals.png...")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data: x and y with a linear relationship plus noise
x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
y = np.array([3.2, 4.1, 5.8, 6.2, 7.9, 8.3, 9.1, 10.5, 11.2, 12.1, 13.5, 13.8, 15.1, 16.0])

# Fit linear regression
X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual data points
ax.scatter(x, y, color='#2196F3', s=80, alpha=0.7, label='Actual data', zorder=3)

# Plot regression line
ax.plot(x, y_pred, color='#F44336', linewidth=2.5,
        label=f'Fitted line: ŷ = {model.intercept_:.2f} + {model.coef_[0]:.2f}x', zorder=2)

# Draw residuals for a few points
highlight_indices = [2, 5, 9, 12]
for idx in highlight_indices:
    ax.plot([x[idx], x[idx]], [y[idx], y_pred[idx]],
            color='#FF9800', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

    # Annotate residual
    mid_y = (y[idx] + y_pred[idx]) / 2
    residual = y[idx] - y_pred[idx]
    ax.annotate(f'ε = {residual:.2f}',
                xy=(x[idx], mid_y),
                xytext=(x[idx] + 0.5, mid_y),
                fontsize=9, color='#FF9800',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#FF9800', alpha=0.8))

# Annotations for specific points
ax.annotate('Actual value', xy=(x[5], y[5]), xytext=(x[5] - 1.5, y[5] + 1),
            fontsize=10, color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5))

ax.annotate('Predicted value\n(on the line)', xy=(x[5], y_pred[5]),
            xytext=(x[5] + 1.5, y_pred[5] - 1.5),
            fontsize=10, color='#F44336',
            arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))

ax.set_xlabel('Feature (x)', fontsize=12, fontweight='bold')
ax.set_ylabel('Target (y)', fontsize=12, fontweight='bold')
ax.set_title('Linear Regression: Fitting a Line to Minimize Residuals',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(1, 16)

plt.tight_layout()
plt.savefig('diagrams/regression_line_residuals.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✓ Saved regression_line_residuals.png")

# ============================================================================
# DIAGRAM 2 & 3: California Housing Analysis
# ============================================================================
print("\n2. Loading California Housing dataset...")

# Load California Housing dataset
housing = fetch_california_housing()
X_full = pd.DataFrame(housing.data, columns=housing.feature_names)
y_full = pd.Series(housing.target, name='MedHouseValue')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# Simple Linear Regression (Single Feature)
print("\n3. Generating simple_regression_california.png...")
X_train_simple = X_train[['MedInc']]
X_test_simple = X_test[['MedInc']]

model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)
y_pred_simple = model_simple.predict(X_test_simple)

r2_simple = r2_score(y_test, y_pred_simple)

# Visualize simple regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test_simple['MedInc'], y_test, alpha=0.4, s=20, color='#2196F3',
           label='Actual')
ax.scatter(X_test_simple['MedInc'], y_pred_simple, alpha=0.5, s=20,
           color='#9C27B0', label='Predicted')

# Plot regression line
x_range = np.linspace(X_test_simple['MedInc'].min(), X_test_simple['MedInc'].max(), 100)
y_range = model_simple.predict(x_range.reshape(-1, 1))
ax.plot(x_range, y_range, color='#FF9800', linewidth=3, label='Regression line')

ax.set_xlabel('Median Income ($10k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Median House Value ($100k)', fontsize=12, fontweight='bold')
ax.set_title(f'Simple Linear Regression (R² = {r2_simple:.3f})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diagrams/simple_regression_california.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✓ Saved simple_regression_california.png")

# Multiple Linear Regression (All Features)
print("\n4. Generating coefficients_multiple.png...")
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)
y_pred_multiple = model_multiple.predict(X_test)

# Display coefficients
coef_df = pd.DataFrame({
    'Feature': X_full.columns,
    'Coefficient': model_multiple.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

# Visualize coefficients
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#4CAF50' if c > 0 else '#2196F3' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.8)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Feature Coefficients in Multiple Linear Regression',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('diagrams/coefficients_multiple.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✓ Saved coefficients_multiple.png")

# ============================================================================
# DIAGRAM 4: Residual Diagnostics
# ============================================================================
print("\n5. Generating residual_diagnostics.png...")

# Compute residuals
residuals = y_test - y_pred_multiple

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Residuals vs Predicted Values
axes[0, 0].scatter(y_pred_multiple, residuals, alpha=0.4, s=20, color='#2196F3')
axes[0, 0].axhline(y=0, color='#F44336', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Predicted\n(Check for patterns)',
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Histogram of Residuals
axes[0, 1].hist(residuals, bins=50, color='#9C27B0', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='#F44336', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribution of Residuals\n(Check for normality)',
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Q-Q Plot
(quantiles, values), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[1, 0].scatter(quantiles, values, alpha=0.6, s=30, color='#FF9800')
axes[1, 0].plot(quantiles, slope * quantiles + intercept, color='#F44336',
                linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Q-Q Plot\n(Points should fall on diagonal)',
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Predicted vs Actual
axes[1, 1].scatter(y_test, y_pred_multiple, alpha=0.4, s=20, color='#2196F3')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                color='#F44336', linestyle='--', linewidth=2, label='Perfect prediction')
axes[1, 1].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Predicted vs Actual\n(Should fall on diagonal)',
                     fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/residual_diagnostics.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✓ Saved residual_diagnostics.png")

print("\n" + "=" * 70)
print("✓ All diagrams generated successfully!")
print("=" * 70)
print("\nGenerated files:")
print("  1. diagrams/regression_line_residuals.png")
print("  2. diagrams/simple_regression_california.png")
print("  3. diagrams/coefficients_multiple.png")
print("  4. diagrams/residual_diagnostics.png")

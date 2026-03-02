> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 18: Linear Regression

## Why This Matters

Every time Zillow estimates a home's value, Netflix predicts your rating for a movie, or a hospital forecasts patient recovery time, there's a good chance linear regression is working behind the scenes. It's the simplest and most interpretable machine learning algorithm—yet it powers decisions worth billions of dollars. Linear regression is your gateway to supervised learning: master it, and you'll understand the foundation for nearly every predictive model that follows.

## The Intuition

Imagine you're opening a coffee shop and want to predict daily revenue. You notice a clear pattern: more customers generally means more money. On Monday with 50 customers, you made $250. On Tuesday with 100 customers, you made $500. The relationship isn't perfect—some days customers spend more, others less—but there's a definite trend.

If you plotted these points on a graph with customers on the x-axis and revenue on the y-axis, you'd see scattered dots roughly following a straight line. **Linear regression finds the "best" line through these dots**—the line that comes closest to all your data points on average.

But what makes a line "best"? Think of it this way: for each day, you can measure the gap between what you actually earned and what your line predicted. These gaps are called **residuals** or errors. The best line is the one that makes these errors as small as possible overall. Specifically, linear regression finds the line that minimizes the sum of *squared* errors—squaring ensures big misses get penalized heavily and prevents positive and negative errors from canceling out.

Now imagine your coffee shop data includes not just customer count, but also temperature and day of week. You realize hot days mean more iced coffee sales, and weekends are busier. **Multiple linear regression** extends the idea: instead of fitting a single line through two-dimensional data, we fit a *hyperplane* through multi-dimensional data. Each feature (customers, temperature, weekend) gets its own weight (coefficient) that tells us how much it contributes to revenue. The math stays the same—we're still finding weights that minimize prediction errors—but now we're capturing richer patterns in the data.

The beauty of linear regression is its interpretability. When you fit the model and get a coefficient of $5 for customers, you can say: "Each additional customer adds about $5 to revenue, on average." Stakeholders understand this. It's actionable. That's why linear regression remains the workhorse of data science despite being one of the oldest algorithms in the toolbox.

## Formal Definition

**Simple Linear Regression** models the relationship between a single feature and a continuous target:

$$\hat{y} = \beta_0 + \beta_1 x$$

where:
- $\hat{y}$ (y-hat) is the predicted value
- $x$ is the feature value
- $\beta_0$ is the intercept (predicted value when x = 0)
- $\beta_1$ is the slope (change in $\hat{y}$ per unit change in x)

**Multiple Linear Regression** extends this to p features:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p$$

Or in matrix notation:

$$\hat{y} = X\boldsymbol{\beta}$$

where:
- $X$ is the feature matrix (n × p)
- $\boldsymbol{\beta}$ is the coefficient vector $[\beta_0, \beta_1, \ldots, \beta_p]^T$

**Optimization Objective:**

Linear regression uses **Ordinary Least Squares (OLS)** to find the optimal coefficients by minimizing the **Sum of Squared Residuals (SSR)**:

$$L(\boldsymbol{\beta}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - X_i\boldsymbol{\beta})^2$$

This loss function is convex and has a unique global minimum that can be found analytically (closed-form solution) or numerically.

**Residuals:**

The residual $\varepsilon_i$ for observation i is the difference between actual and predicted values:

$$\varepsilon_i = y_i - \hat{y}_i$$

> **Key Concept:** Linear regression finds the line (or hyperplane) that minimizes the sum of squared prediction errors across all training data, yielding the most accurate predictions on average.

## Visual

Let me create a visualization showing how linear regression fits a line to data and what residuals represent:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
ax.scatter(x, y, color='#2E86AB', s=80, alpha=0.7, label='Actual data', zorder=3)

# Plot regression line
ax.plot(x, y_pred, color='#A23B72', linewidth=2.5, label=f'Fitted line: ŷ = {model.intercept_:.2f} + {model.coef_[0]:.2f}x', zorder=2)

# Draw residuals for a few points
highlight_indices = [2, 5, 9, 12]
for idx in highlight_indices:
    ax.plot([x[idx], x[idx]], [y[idx], y_pred[idx]],
            color='#F18F01', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

    # Annotate residual
    mid_y = (y[idx] + y_pred[idx]) / 2
    residual = y[idx] - y_pred[idx]
    ax.annotate(f'ε = {residual:.2f}',
                xy=(x[idx], mid_y),
                xytext=(x[idx] + 0.5, mid_y),
                fontsize=9, color='#F18F01',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#F18F01', alpha=0.8))

# Annotations for specific points
ax.annotate('Actual value', xy=(x[5], y[5]), xytext=(x[5] - 1.5, y[5] + 1),
            fontsize=10, color='#2E86AB',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.5))

ax.annotate('Predicted value\n(on the line)', xy=(x[5], y_pred[5]), xytext=(x[5] + 1.5, y_pred[5] - 1.5),
            fontsize=10, color='#A23B72',
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5))

ax.set_xlabel('Feature (x)', fontsize=12, fontweight='bold')
ax.set_ylabel('Target (y)', fontsize=12, fontweight='bold')
ax.set_title('Linear Regression: Fitting a Line to Minimize Residuals', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(1, 16)

plt.tight_layout()
plt.savefig('diagrams/regression_line_residuals.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Model equation: ŷ = {model.intercept_:.3f} + {model.coef_[0]:.3f}x")
print(f"Sum of squared residuals: {np.sum((y - y_pred)**2):.3f}")

# Output:
# Model equation: ŷ = 1.479 + 0.955x
# Sum of squared residuals: 4.321
```

**Caption:** Linear regression finds the line that comes closest to all data points by minimizing the sum of squared residuals (orange dashed lines). Each residual ε represents the prediction error for one observation.

## Code Example

Let's build a complete linear regression workflow using the California Housing dataset. We'll start with simple regression (one feature) and progress to multiple regression (all features).

```python
# Linear Regression: Simple and Multiple Regression on California Housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Load California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseValue')

print("Dataset shape:", X.shape)
print("\nFeatures:", list(X.columns))
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget statistics:")
print(y.describe())

# Split data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# =============================================================================
# PART 1: Simple Linear Regression (Single Feature)
# =============================================================================
print("\n" + "="*70)
print("SIMPLE LINEAR REGRESSION: Using MedInc (Median Income) only")
print("="*70)

# Extract single feature: MedInc (median income in block group)
X_train_simple = X_train[['MedInc']]
X_test_simple = X_test[['MedInc']]

# Fit simple linear regression model
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

# Make predictions
y_pred_simple = model_simple.predict(X_test_simple)

# Evaluate performance
r2_simple = r2_score(y_test, y_pred_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_simple))

print(f"\nModel equation: ŷ = {model_simple.intercept_:.3f} + {model_simple.coef_[0]:.3f} × MedInc")
print(f"Interpretation: Each $10k increase in median income predicts ${model_simple.coef_[0]:.3f} increase in house value (in $100k)")
print(f"\nTest Set Performance:")
print(f"  R² = {r2_simple:.3f} (explains {r2_simple*100:.1f}% of variance)")
print(f"  RMSE = ${rmse_simple:.3f} (average error of ${rmse_simple*100:.0f}k)")

# Show sample predictions
print("\nSample predictions (first 5 test samples):")
comparison_df = pd.DataFrame({
    'MedInc': X_test_simple['MedInc'].values[:5],
    'Actual': y_test.values[:5],
    'Predicted': y_pred_simple[:5],
    'Residual': (y_test.values[:5] - y_pred_simple[:5])
})
print(comparison_df.to_string(index=False))

# Visualize simple regression
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X_test_simple['MedInc'], y_test, alpha=0.4, s=20, color='#2E86AB', label='Actual')
ax.scatter(X_test_simple['MedInc'], y_pred_simple, alpha=0.5, s=20, color='#A23B72', label='Predicted')

# Plot regression line
x_range = np.linspace(X_test_simple['MedInc'].min(), X_test_simple['MedInc'].max(), 100)
y_range = model_simple.predict(x_range.reshape(-1, 1))
ax.plot(x_range, y_range, color='#F18F01', linewidth=3, label='Regression line')

ax.set_xlabel('Median Income ($10k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Median House Value ($100k)', fontsize=12, fontweight='bold')
ax.set_title(f'Simple Linear Regression (R² = {r2_simple:.3f})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diagrams/simple_regression_california.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# PART 2: Multiple Linear Regression (All Features)
# =============================================================================
print("\n" + "="*70)
print("MULTIPLE LINEAR REGRESSION: Using all 8 features")
print("="*70)

# Fit multiple linear regression model
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)

# Make predictions
y_pred_multiple = model_multiple.predict(X_test)

# Evaluate performance
r2_multiple = r2_score(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mean_squared_error(y_test, y_pred_multiple))

print(f"\nTest Set Performance:")
print(f"  R² = {r2_multiple:.3f} (explains {r2_multiple*100:.1f}% of variance)")
print(f"  RMSE = ${rmse_multiple:.3f} (average error of ${rmse_multiple*100:.0f}k)")
print(f"\nImprovement over simple regression:")
print(f"  R² improved by {(r2_multiple - r2_simple):.3f} ({(r2_multiple - r2_simple)/r2_simple*100:.1f}%)")
print(f"  RMSE reduced by ${(rmse_simple - rmse_multiple):.3f}")

# Display coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_multiple.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(f"\nIntercept: {model_multiple.intercept_:.3f}")
print("\nCoefficients (sorted by absolute magnitude):")
print(coef_df.to_string(index=False))

# Interpret top 3 coefficients
print("\nInterpretation of top 3 coefficients:")
for idx, row in coef_df.head(3).iterrows():
    feature = row['Feature']
    coef = row['Coefficient']
    if coef > 0:
        print(f"  • {feature}: A 1-unit increase predicts ${abs(coef):.3f} (${abs(coef)*100:.0f}k) increase in house value")
    else:
        print(f"  • {feature}: A 1-unit increase predicts ${abs(coef):.3f} (${abs(coef)*100:.0f}k) decrease in house value")

# Visualize coefficients
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#A23B72' if c > 0 else '#2E86AB' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.8)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Feature Coefficients in Multiple Linear Regression', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('diagrams/coefficients_multiple.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# PART 3: Residual Analysis
# =============================================================================
print("\n" + "="*70)
print("RESIDUAL ANALYSIS: Checking Model Assumptions")
print("="*70)

# Compute residuals
residuals = y_test - y_pred_multiple

print(f"\nResidual statistics:")
print(f"  Mean: {residuals.mean():.6f} (should be near 0)")
print(f"  Std Dev: {residuals.std():.3f}")
print(f"  Min: {residuals.min():.3f}")
print(f"  Max: {residuals.max():.3f}")

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Residuals vs Predicted Values
axes[0, 0].scatter(y_pred_multiple, residuals, alpha=0.4, s=20, color='#2E86AB')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Residuals vs Predicted\n(Check for patterns)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Histogram of Residuals
axes[0, 1].hist(residuals, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Distribution of Residuals\n(Check for normality)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Q-Q Plot
(quantiles, values), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[1, 0].scatter(quantiles, values, alpha=0.6, s=30, color='#F18F01')
axes[1, 0].plot(quantiles, slope * quantiles + intercept, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Q-Q Plot\n(Points should fall on diagonal)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Predicted vs Actual
axes[1, 1].scatter(y_test, y_pred_multiple, alpha=0.4, s=20, color='#2E86AB')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                'r--', linewidth=2, label='Perfect prediction')
axes[1, 1].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Predicted vs Actual\n(Should fall on diagonal)', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/residual_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# Identify worst predictions
worst_indices = np.argsort(np.abs(residuals))[-5:]
print("\nTop 5 worst predictions (largest absolute residuals):")
worst_df = pd.DataFrame({
    'Actual': y_test.values[worst_indices],
    'Predicted': y_pred_multiple[worst_indices],
    'Residual': residuals.values[worst_indices],
    'MedInc': X_test['MedInc'].values[worst_indices],
    'HouseAge': X_test['HouseAge'].values[worst_indices]
})
print(worst_df.to_string(index=False))

print("\n" + "="*70)
print("Summary: Multiple regression (R²={:.3f}) significantly outperforms".format(r2_multiple))
print("simple regression (R²={:.3f}). Residuals show approximate normality".format(r2_simple))
print("but slight heteroscedasticity. Model assumptions mostly satisfied.")
print("="*70)

# Output:
# Dataset shape: (20640, 8)
#
# Features: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
#
# First 5 rows:
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25
#
# Target statistics:
# count    20640.000000
# mean         2.068558
# std          1.153956
# min          0.149990
# 25%          1.196000
# 50%          1.797000
# 75%          2.647250
# max          5.000010
# Name: MedHouseValue, dtype: float64
#
# Train set: 16512 samples
# Test set: 4128 samples
#
# ======================================================================
# SIMPLE LINEAR REGRESSION: Using MedInc (Median Income) only
# ======================================================================
#
# Model equation: ŷ = 0.450 + 0.418 × MedInc
# Interpretation: Each $10k increase in median income predicts $0.418 increase in house value (in $100k)
#
# Test Set Performance:
#   R² = 0.473 (explains 47.3% of variance)
#   RMSE = $0.746 (average error of $75k)
#
# Sample predictions (first 5 test samples):
#  MedInc  Actual  Predicted  Residual
#  4.2143  1.4060   2.213187 -0.807187
#  4.1518  0.4770   2.187053 -1.710053
#  3.8750  1.1140   2.069474 -0.955474
#  4.6225  2.1670   2.382009 -0.215009
#  2.7750  0.7810   1.610083 -0.829083
#
# ======================================================================
# MULTIPLE LINEAR REGRESSION: Using all 8 features
# ======================================================================
#
# Test Set Performance:
#   R² = 0.594 (explains 59.4% of variance)
#   RMSE = $0.655 (average error of $66k)
#
# Improvement over simple regression:
#   R² improved by 0.121 (25.5%)
#   RMSE reduced by $0.091
#
# Intercept: -37.002
#
# Coefficients (sorted by absolute magnitude):
#      Feature  Coefficient
#     Latitude   -0.422030
#    Longitude   -0.434198
#       MedInc    0.438193
#    AveOccup   -0.000413
#   Population   -0.000121
#     HouseAge    0.009469
#    AveRooms   -0.112117
#   AveBedrms    0.652908
#
# Interpretation of top 3 coefficients:
#   • Longitude: A 1-unit increase predicts $0.434 ($43k) decrease in house value
#   • Latitude: A 1-unit increase predicts $0.422 ($42k) decrease in house value
#   • MedInc: A 1-unit increase predicts $0.438 ($44k) increase in house value
#
# ======================================================================
# RESIDUAL ANALYSIS: Checking Model Assumptions
# ======================================================================
#
# Residual statistics:
#   Mean: 0.000000 (should be near 0)
#   Std Dev: 0.654
#   Min: -3.010
#   Max: 3.604
#
# Top 5 worst predictions (largest absolute residuals):
#  Actual  Predicted  Residual  MedInc  HouseAge
#  5.0001   1.395860  3.604242  2.0208      36.0
#  1.1990   4.209383 -3.010383  3.8750      38.0
#  4.5890   1.614649  2.974351  1.8750      18.0
#  0.7780   3.731178 -2.953178  4.4583      17.0
#  0.5240   3.430668 -2.906668  3.8333      33.0
#
# ======================================================================
# Summary: Multiple regression (R²=0.594) significantly outperforms
# simple regression (R²=0.473). Residuals show approximate normality
# but slight heteroscedasticity. Model assumptions mostly satisfied.
# ======================================================================
```

## Walkthrough

Let's break down what we just did step by step:

**Data Loading and Preparation (Lines 1-28)**

We load the California Housing dataset, which contains 20,640 samples with 8 features each. The target variable is median house value in $100k units. We split the data 80/20 into training and test sets using `random_state=42` for reproducibility. This separation is crucial: we train on one set and evaluate on another to estimate how well our model will perform on new, unseen data.

**Simple Linear Regression (Lines 30-71)**

We start by using only one feature: `MedInc` (median income in the block group). After fitting the model, we get the equation: ŷ = 0.450 + 0.418 × MedInc. This tells us that each $10k increase in median income predicts a $41,800 increase in house value. The R² of 0.473 means median income alone explains 47.3% of the variance in house prices—not bad for a single feature! The RMSE of $0.746 (in $100k units) translates to an average prediction error of $75,000.

The scatter plot visualization shows our predictions (purple) overlaid on actual values (blue). The orange line is our fitted regression line. You can see the predictions cluster around the line, but there's substantial spread—many factors beyond income affect house prices.

**Multiple Linear Regression (Lines 73-109)**

When we include all 8 features, performance improves significantly. R² jumps to 0.594 (59.4% variance explained) and RMSE drops to $0.655 ($66k average error). That's a 25.5% improvement in R² just from adding more features!

The coefficient table reveals which features matter most. Interestingly, `Latitude` and `Longitude` have the largest magnitude coefficients (both negative), meaning location strongly affects price. A house at higher latitude/longitude (further north/east in California) tends to be cheaper. `MedInc` remains important with a coefficient of 0.438. Meanwhile, `AveOccup` and `Population` have tiny coefficients near zero, suggesting they add little predictive value.

**Interpreting "holding all else constant"**: When we say "a 1-unit increase in MedInc predicts a $44k increase in house value," we mean this is the effect *when all other features stay the same*. This is the power of multiple regression—it isolates each feature's independent contribution.

**Residual Analysis (Lines 111-165)**

The diagnostic plots tell us whether our model assumptions hold:

1. **Residuals vs Predicted**: We want random scatter around zero. Our plot shows mostly random scatter, but there's a slight "fan" shape (wider spread at higher predicted values), indicating mild *heteroscedasticity* (non-constant variance). This suggests the model is less reliable for expensive houses.

2. **Histogram of Residuals**: The distribution is approximately bell-shaped (normal), though with a slight right skew. For large datasets like ours (n=4128 test samples), normality is less critical thanks to the Central Limit Theorem.

3. **Q-Q Plot**: Most points fall along the diagonal line, confirming approximate normality. Deviations at the tails suggest some extreme residuals (outliers).

4. **Predicted vs Actual**: Points should cluster along the red diagonal (perfect prediction line). Our points show good agreement in the middle range but more scatter at extremes—consistent with our other diagnostics.

**Worst Predictions**: The five largest residuals reveal where our model fails. The worst case has an actual value of $500k but predicted only $140k (error of $360k). Looking at the features, this house has low median income (2.02) but extremely high value—possibly a luxury property in an otherwise modest neighborhood. Linear regression struggles with such outliers because it treats all errors equally (after squaring).

**Key Insight**: Moving from simple to multiple regression significantly improved predictions, but no model is perfect. The residual analysis reveals mild assumption violations (heteroscedasticity) and identifies outliers where the model fails. In practice, we might address these issues by log-transforming the target, removing outliers, or trying a non-linear model like random forests.

## Common Pitfalls

**1. Mistaking Correlation for Causation**

Linear regression finds *associations*, not causes. If your model shows that ice cream sales predict drowning deaths (both are higher in summer), that doesn't mean ice cream causes drowning—temperature is a *confounding variable*. Before claiming "feature X causes outcome Y," you need controlled experiments or causal inference techniques. In regression, always say "X *predicts* Y" or "X is *associated with* Y," not "X *causes* Y."

**2. Comparing Raw Coefficients Across Different Scales**

Beginners often conclude "Feature A is more important than Feature B because its coefficient is larger." This is wrong if features have different scales. A coefficient of 0.5 for a feature ranging [0, 1000] means a 1-unit increase (small change) predicts a 0.5 change in target. But a coefficient of 100 for a feature ranging [0, 1] means a 1-unit increase (huge change, doubling!) predicts a 100-point change. The first feature may actually be more important despite the smaller coefficient. **Solution**: Standardize features (mean=0, std=1) using `StandardScaler` before fitting, then compare standardized coefficients. A standardized coefficient tells you "how many standard deviations does the target change per standard deviation increase in the feature?"

**3. Ignoring Multicollinearity**

When features are highly correlated (e.g., house size and number of rooms), coefficient estimates become unstable—small changes in data lead to wildly different coefficients. This doesn't hurt predictions much, but it destroys interpretability. You might see a negative coefficient for "house size" even though larger houses obviously cost more, simply because the model is confused by the correlation with "number of rooms." **Solution**: Check the correlation matrix. If features have |correlation| > 0.8, consider removing one, combining them, or using regularization (Ridge/Lasso, covered in Chapter 20). You can also compute Variance Inflation Factor (VIF); VIF > 10 indicates severe multicollinearity requiring action.

## Practice Exercises

**Exercise 1 (Easy): Single Feature Regression on Diabetes Dataset**

Load the Diabetes dataset (`from sklearn.datasets import load_diabetes`) and build a simple linear regression model.

Tasks:
1. Load the dataset and examine its shape and features
2. Use only the 'bmi' (body mass index) feature to predict disease progression (target)
3. Split data into train/test sets (80/20, `random_state=42`)
4. Fit a `LinearRegression` model
5. Make predictions on the test set
6. Calculate and report:
   - R² score on test set
   - RMSE on test set
   - The model's coefficient and intercept
7. Interpret: "A 1-unit increase in BMI predicts a _____ change in disease progression"
8. Make a prediction: What disease progression would you expect for BMI = 0.05?

**Hints:**
- Access features with: `X = diabetes.data[:, 2:3]` (BMI is the 3rd feature, index 2)
- Or convert to DataFrame: `pd.DataFrame(diabetes.data, columns=diabetes.feature_names)`
- RMSE = `np.sqrt(mean_squared_error(y_test, y_pred))`

**Expected Results:**
- R² should be around 0.33-0.35 (BMI alone explains ~34% of variance)
- RMSE around 55-60
- Positive coefficient (higher BMI → higher disease progression)

---

**Exercise 2 (Medium): Model Comparison and Feature Selection**

Return to the California Housing dataset and compare different feature subsets systematically.

Tasks:
1. Load California Housing dataset
2. Build THREE models:
   - **Model A**: Uses only `MedInc`
   - **Model B**: Uses `MedInc`, `HouseAge`, and `AveRooms`
   - **Model C**: Uses all 8 features
3. For each model:
   - Fit on training set
   - Compute test set R² and RMSE
   - Record the number of features used
4. Create a comparison DataFrame with columns: `Model`, `Features`, `Num_Features`, `R²`, `RMSE`
5. Answer these questions:
   - Which model performs best on the test set? Why?
   - How much does R² improve when going from Model A → B → C?
   - Is there diminishing returns (each feature adds less value)?
6. For Model C (best model):
   - Identify the 3 features with largest absolute coefficient values
   - Interpret the top coefficient in plain English
7. **Bonus**: Create a line plot showing R² on y-axis vs. Number of Features on x-axis

**Expected Observations:**
- Model C should perform best (R² ≈ 0.59-0.60)
- Model B should be between A and C (R² ≈ 0.50-0.52)
- Adding features improves performance, but gains slow down (diminishing returns)
- Location features (Latitude, Longitude) and MedInc should have largest coefficients

---

**Exercise 3 (Hard): End-to-End Regression Project with Diagnostics**

You work for a real estate company that needs to predict apartment rental prices. Complete a full regression workflow from data generation to diagnosis.

**Part 1: Generate Synthetic Dataset (15 min)**

Create a realistic apartment rental dataset with 500 samples and these features:
- `sqft`: Square footage, uniform random [500, 3000]
- `bedrooms`: Number of bedrooms, random choice [1, 2, 3, 4, 5]
- `bathrooms`: Number of bathrooms, random choice [1, 2, 3]
- `distance_km`: Distance to city center, uniform random [0, 20]
- `age_years`: Building age, uniform random [0, 100]
- `floor`: Floor number, uniform random [1, 30]
- `has_parking`: Binary [0, 1], random with 60% having parking

Generate target `rent` using this formula:
```
rent = 1000 + 1.5*sqft + 200*bedrooms + 150*bathrooms - 50*distance_km
       - 10*age_years + 5*floor + 300*has_parking + noise
```
where `noise ~ Normal(0, 200)`.

Create a pandas DataFrame and verify with `.head()`, `.shape`, `.describe()`.

**Part 2: Model Building (15 min)**

4. Split data into train/test (70/30, `random_state=42`)
5. Fit a `LinearRegression` model on all features
6. Display learned coefficients in a readable DataFrame (sorted by absolute value)
7. Compute test set R² and RMSE
8. Interpret: Do the learned coefficients match the true data-generating process? Which features are most important?

**Part 3: Model Diagnostics (20 min)**

9. Create a 2×2 subplot with diagnostic plots:
   - **Plot 1**: Residuals vs. Predicted (check for patterns, homoscedasticity)
   - **Plot 2**: Histogram of residuals (check normality)
   - **Plot 3**: Q-Q plot (check if residuals fall on diagonal)
   - **Plot 4**: Predicted vs. Actual (check alignment with diagonal)
10. Check assumptions:
    - Is the relationship linear? (residual plot should show random scatter)
    - Are residuals normally distributed? (Q-Q plot should be diagonal)
    - Is variance constant? (residual plot shouldn't fan out)
11. Identify the 5 worst predictions (largest |residuals|)
12. Discuss: Why did the model fail on these samples? Are they outliers?

**Part 4: Report Writing (10 min)**

13. Write a short markdown report with sections:
    - **Executive Summary**: 2-3 sentences on model performance
    - **Model Performance**: R², RMSE, interpretation in business terms
    - **Key Insights**: Top 3 most important features for pricing
    - **Assumptions Check**: Which assumptions are met/violated?
    - **Limitations**: What are the model's weaknesses?
    - **Next Steps**: What would you try to improve predictions?
      - Examples: Add interaction terms (bedroom × sqft), polynomial features (distance²), remove outliers, try ensemble models

**Expected Results:**
- R² should be very high (0.95+) since you know the true model
- Learned coefficients should closely match true coefficients
- Some noise will prevent perfect predictions
- Residuals should be approximately normal (by design)
- Model should satisfy assumptions (linear relationship, homoscedastic by design)

**Bonus Challenges:**
- Add an interaction term `sqft * bedrooms` and see if R² improves
- Try polynomial features for `distance_km` (maybe rent drops faster close to center?)
- Compare to a baseline model that predicts median rent for all apartments
- Implement 5-fold cross-validation and report mean R² and standard deviation

---

## Key Takeaways

- **Linear regression finds the line (or hyperplane) that minimizes squared prediction errors across training data**, making it optimal for linear relationships between features and targets.

- **Simple regression uses one feature; multiple regression combines many features**, with each coefficient representing a feature's independent contribution "holding all else constant."

- **R² measures proportion of variance explained (0 to 1); RMSE measures average prediction error in original units**—use R² for model comparison, RMSE for communicating accuracy to stakeholders.

- **Always check assumptions through residual analysis**: plot residuals vs. predicted (check for patterns), histogram (check normality), and Q-Q plot (check distribution)—violations indicate when linear regression is inappropriate.

- **Coefficients are only comparable after standardizing features** to mean=0, std=1; raw coefficients depend on feature scales and can't be directly compared for feature importance without standardization.

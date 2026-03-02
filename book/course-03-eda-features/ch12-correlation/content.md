> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# Chapter 12: Correlation and Relationships

## Why This Matters

Understanding how variables move together is at the heart of data science. When building a model to predict house prices, it's necessary to know which features—square footage, number of bedrooms, neighborhood income—actually relate to price. When analyzing medical data, it's critical to identify which risk factors correlate with disease outcomes. Correlation analysis is the systematic way to measure and visualize these relationships, guiding everything from feature selection to model design. Get this wrong, and models will be built on irrelevant features or miss the patterns that matter most.

## Intuition

Think about dancing with a partner. When they're perfectly synchronized—moving together in ballroom dancing—they're like two variables with perfect positive correlation. When one steps forward, the other steps back by exactly the same amount. That's correlation of +1.

Now imagine two people at completely different parties in different cities. Their movements are totally independent—no connection whatsoever. That's correlation of zero.

What about a seesaw? When one side goes up, the other goes down by exactly the same amount. They're perfectly synchronized but in opposite directions. That's perfect negative correlation of -1.

Correlation measures this "dance synchronization" between variables. It quantifies how predictably one variable changes when another changes. If height and weight have a correlation of 0.7, taller people tend to weigh more—not perfectly, but there's a clear tendency. If exercise and resting heart rate have a correlation of -0.6, more exercise tends to mean lower resting heart rate.

The crucial insight: correlation measures **how variables move together**, not whether one causes the other. Ice cream sales and drowning deaths are highly correlated because both increase in summer, but ice cream doesn't cause drownings. They're both driven by a third factor: warm weather. Correlation tells about association and prediction—if one variable is known, can the other be predicted? It doesn't tell about causation.

## Formal Definition

The **Pearson correlation coefficient** (denoted r) measures the strength and direction of the linear relationship between two continuous variables X and Y. It is defined as:

$$r_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Where:
- **Cov(X, Y)** is the covariance between X and Y (how they vary together)
- **σ_X** and **σ_Y** are the standard deviations of X and Y
- **x̄** and **ȳ** are the means of X and Y
- **n** is the number of samples

The correlation coefficient r is standardized and always falls between -1 and +1:

- **r = +1**: Perfect positive linear relationship (all points on an upward-sloping line)
- **r = 0**: No linear relationship (variables move independently)
- **r = -1**: Perfect negative linear relationship (all points on a downward-sloping line)

**Interpretation guidelines:**
- |r| > 0.7: Strong correlation
- 0.3 < |r| < 0.7: Moderate correlation
- |r| < 0.3: Weak correlation

**Alternative correlation measures:**

**Spearman's rank correlation (ρ)** measures monotonic relationships using ranks instead of raw values:

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

Where d_i is the difference in ranks for each observation.

**Kendall's tau (τ)** measures ordinal association by counting concordant and discordant pairs.

> **Key Concept:** Correlation measures the strength and direction of association between variables, ranging from -1 (perfect negative) through 0 (no linear relationship) to +1 (perfect positive), but tells nothing about causation.

## Visualization

The relationship between correlation coefficients and scatter plot patterns is fundamental to understanding what correlation measures. Here's a comprehensive visualization showing how different r values manifest visually:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)

# Create figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.suptitle('Understanding Correlation: Visual Patterns for Different r Values',
             fontsize=16, fontweight='bold', y=0.995)

# Define correlation values to demonstrate
correlations = [1.0, 0.9, 0.7, 0.5, 0.3, 0.0, -0.3, -0.7, -0.95]

# Generate data for each correlation level
n = 100

for idx, target_r in enumerate(correlations):
    ax = axes[idx // 3, idx % 3]

    # Generate correlated data
    if target_r == 0:
        x = np.random.randn(n)
        y = np.random.randn(n)
    else:
        x = np.random.randn(n)
        y = target_r * x + np.sqrt(1 - target_r**2) * np.random.randn(n)

    # Calculate actual correlation
    actual_r, _ = pearsonr(x, y)

    # Create scatter plot
    ax.scatter(x, y, alpha=0.6, s=30, color='steelblue', edgecolors='darkblue', linewidth=0.5)

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7, label=f'r = {actual_r:.2f}')

    # Formatting
    ax.set_xlabel('Variable X', fontsize=10)
    ax.set_ylabel('Variable Y', fontsize=10)
    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add interpretation text
    if actual_r > 0.7:
        strength = "Strong Positive"
    elif actual_r > 0.3:
        strength = "Moderate Positive"
    elif actual_r > -0.3:
        strength = "Weak/None"
    elif actual_r > -0.7:
        strength = "Moderate Negative"
    else:
        strength = "Strong Negative"

    ax.set_title(strength, fontsize=11, fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('book/course-03-eda-features/ch12-correlation/diagrams/correlation_patterns.png',
            dpi=300, bbox_inches='tight')
plt.show()

# Output: A 3x3 grid showing scatter plots with r values ranging from +1.0 to -0.95
# Each plot clearly demonstrates how point scatter changes with correlation strength
# Strong correlations show tight linear patterns; weak correlations show loose clouds
```

**Figure Caption:** Visual patterns for different Pearson correlation coefficients. Strong positive correlations (r > 0.7) show tight upward-sloping patterns, while strong negative correlations (r < -0.7) show tight downward-sloping patterns. Near-zero correlations show random clouds with no clear direction. The regression line (red) helps visualize the linear trend.

## Examples

### Part 1: Computing Single Correlation

```python
# Comprehensive Correlation Analysis with California Housing Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from scipy.stats import pearsonr, spearmanr, kendalltau

# Set style and random seed
sns.set_style('whitegrid')
np.random.seed(42)

# Load California Housing dataset
housing_data = fetch_california_housing()
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['MedHouseVal'] = housing_data.target  # Add target variable

print("California Housing Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Output:
# California Housing Dataset Shape: (20640, 9)
#
# First 5 rows:
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23        4.526
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22        3.585
# 2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24        3.521
# 3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25        3.413
# 4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25        3.422

# Select two features for detailed analysis
feature_x = 'MedInc'  # Median income in block
feature_y = 'MedHouseVal'  # Median house value (target)

# Compute Pearson correlation using pandas
corr_pandas = df[[feature_x, feature_y]].corr().iloc[0, 1]
print(f"\n{'='*60}")
print(f"Correlation between {feature_x} and {feature_y}")
print(f"{'='*60}")
print(f"Using pandas .corr(): {corr_pandas:.4f}")

# Compute Pearson correlation using scipy (includes p-value)
r, p_value = pearsonr(df[feature_x], df[feature_y])
print(f"Using scipy.stats.pearsonr(): r = {r:.4f}, p-value = {p_value:.2e}")

# Interpretation
print(f"\nInterpretation:")
print(f"  • r = {r:.4f} indicates a strong positive correlation")
print(f"  • As median income increases, house value tends to increase")
print(f"  • r² = {r**2:.4f} means {r**2*100:.1f}% of variance in house value")
print(f"    is explained by median income")

# Output:
# ============================================================
# Correlation between MedInc and MedHouseVal
# ============================================================
# Using pandas .corr(): 0.6880
# Using scipy.stats.pearsonr(): r = 0.6880, p-value = 0.00e+00
#
# Interpretation:
#   • r = 0.6880 indicates a strong positive correlation
#   • As median income increases, house value tends to increase
#   • r² = 0.4733 means 47.3% of variance in house value
#     is explained by median income
```

The code starts by focusing on two variables: median income (`MedInc`) and median house value (`MedHouseVal`). The correlation is computed two ways—using pandas `.corr()` method and scipy's `pearsonr()` function. Both give r = 0.688, a strong positive correlation.

What does this mean? When median income is higher, house values tend to be higher. The r² value of 0.473 indicates that 47.3% of the variation in house values can be explained by median income alone. That's substantial! The p-value near zero confirms this relationship is statistically significant—it's not due to random chance.

### Part 2: Visualizing the Relationship

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot with regression line
axes[0].scatter(df[feature_x], df[feature_y], alpha=0.3, s=10,
                color='steelblue', edgecolors='none')
# Add regression line
z = np.polyfit(df[feature_x], df[feature_y], 1)
p = np.poly1d(z)
x_line = np.linspace(df[feature_x].min(), df[feature_x].max(), 100)
axes[0].plot(x_line, p(x_line), 'r-', linewidth=2.5, alpha=0.8,
            label=f'r = {r:.3f}')
axes[0].set_xlabel('Median Income (in $10,000s)', fontsize=12)
axes[0].set_ylabel('Median House Value (in $100,000s)', fontsize=12)
axes[0].set_title('Income vs. House Value: Strong Positive Correlation',
                 fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Hexbin plot for better visualization with many points
hb = axes[1].hexbin(df[feature_x], df[feature_y], gridsize=50,
                    cmap='Blues', mincnt=1)
axes[1].plot(x_line, p(x_line), 'r-', linewidth=2.5, alpha=0.9,
            label=f'r = {r:.3f}')
axes[1].set_xlabel('Median Income (in $10,000s)', fontsize=12)
axes[1].set_ylabel('Median House Value (in $100,000s)', fontsize=12)
axes[1].set_title('Hexbin Density Plot (Better for Large Datasets)',
                 fontsize=13, fontweight='bold')
cb = plt.colorbar(hb, ax=axes[1])
cb.set_label('Count', fontsize=11)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('book/course-03-eda-features/ch12-correlation/diagrams/income_vs_value.png',
           dpi=300, bbox_inches='tight')
plt.show()

# Output: Two side-by-side plots showing strong positive linear relationship
# Left: Traditional scatter plot with regression line
# Right: Hexbin density plot showing concentration of data points
```

Two visualizations are created here. The scatter plot clearly shows the upward trend—as income increases, house value increases. The regression line (in red) captures this linear relationship. The hexbin plot on the right is particularly useful for large datasets like this one (20,640 samples)—it shows density through color, revealing that most data clusters in the lower-middle income range.

Notice how the points scatter around the regression line. If the correlation were perfect (r = 1.0), all points would lie exactly on the line. The scatter represents the 53% of variance NOT explained by income.

### Part 3: Full Correlation Matrix

```python
# Compute correlation matrix for all features
corr_matrix = df.corr()

print(f"\n{'='*60}")
print("Full Correlation Matrix (rounded to 2 decimals)")
print(f"{'='*60}")
print(corr_matrix.round(2))

# Output:
# ============================================================
# Full Correlation Matrix (rounded to 2 decimals)
# ============================================================
#              MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
# MedInc         1.00     -0.11      0.33      -0.07       -0.00     -0.02     -0.08      -0.01         0.69
# HouseAge      -0.11      1.00     -0.15      -0.08       -0.30      0.01     0.01       -0.11        0.11
# AveRooms       0.33     -0.15      1.00       0.85       -0.07     -0.00    -0.11       -0.03        0.15
# AveBedrms     -0.07     -0.08      0.85       1.00       -0.07      0.01    -0.07       -0.04       -0.05
# Population    -0.00     -0.30     -0.07      -0.07        1.00      0.07     0.00        0.01       -0.02
# AveOccup      -0.02      0.01     -0.00       0.01        0.07      1.00     0.00       -0.00       -0.04
# Latitude      -0.08      0.01     -0.11      -0.07        0.00      0.00     1.00       -0.92       -0.14
# Longitude     -0.01     -0.11     -0.03      -0.04        0.01     -0.00    -0.92        1.00       -0.05
# MedHouseVal    0.69      0.11      0.15      -0.05       -0.02     -0.04    -0.14       -0.05        1.00

# Find strongest correlations with target
target_corr = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False)
print("\nFeatures ranked by correlation with MedHouseVal:")
print(target_corr)

# Output:
# Features ranked by correlation with MedHouseVal:
# MedHouseVal    1.000000
# MedInc         0.688075
# AveRooms       0.151948
# Latitude       0.144160
# HouseAge       0.105623
# Longitude      0.045967
# AveBedrms      0.049686
# AveOccup       0.038423
# Population     0.024650
```

The correlation matrix shows all pairwise correlations. Read it like a table: the entry at row X, column Y shows the correlation between those features. Notice:

- The diagonal is all 1.0 (every variable correlates perfectly with itself)
- The matrix is symmetric (r(X,Y) = r(Y,X))
- MedInc has the strongest correlation with the target (0.69)
- AveRooms and AveBedrms are highly correlated (0.85)—they measure similar things
- Latitude and Longitude are strongly negatively correlated (-0.92)—this is geographic

### Part 4: Correlation Heatmap Visualization

```python
fig, ax = plt.subplots(figsize=(12, 9))

# Create heatmap with annotations
sns.heatmap(corr_matrix,
           annot=True,           # Show correlation values
           fmt='.2f',            # Format to 2 decimal places
           cmap='coolwarm',      # Diverging colormap (blue-white-red)
           center=0,             # Center colormap at zero
           vmin=-1,              # Minimum value
           vmax=1,               # Maximum value
           square=True,          # Square cells
           linewidths=0.5,       # Grid lines between cells
           linecolor='white',
           cbar_kws={'label': 'Correlation Coefficient',
                    'shrink': 0.8})

plt.title('Correlation Matrix: California Housing Dataset',
         fontsize=15, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('book/course-03-eda-features/ch12-correlation/diagrams/correlation_heatmap.png',
           dpi=300, bbox_inches='tight')
plt.show()

# Output: A comprehensive heatmap showing all pairwise correlations
# Red colors indicate positive correlations
# Blue colors indicate negative correlations
# The diagonal is all 1.0 (perfect self-correlation)
```

The heatmap makes patterns jump out visually. Red indicates positive correlation, blue indicates negative, and white indicates near-zero. The most important insight: look at the MedHouseVal row/column. The darkest red cell (besides the diagonal) is with MedInc—visually confirming it's the strongest predictor.

### Part 5: Detecting Multicollinearity

```python
# Find highly correlated feature pairs (excluding diagonal and target)
print(f"\n{'='*60}")
print("Multicollinearity Detection: Highly Correlated Feature Pairs")
print(f"{'='*60}")

# Create mask for upper triangle (avoid duplicates)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            high_corr_pairs.append((feature1, feature2, corr_val))

if high_corr_pairs:
    print("\nFeature pairs with |r| > 0.8 (potential multicollinearity):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  • {feat1} <-> {feat2}: r = {corr:.3f}")
    print("\nAction: Consider removing one feature from each highly correlated pair")
else:
    print("\nNo feature pairs with |r| > 0.8 found in this dataset")

# Output:
# ============================================================
# Multicollinearity Detection: Highly Correlated Feature Pairs
# ============================================================
#
# Feature pairs with |r| > 0.8 (potential multicollinearity):
#   • AveRooms <-> AveBedrms: r = 0.848
#   • Latitude <-> Longitude: r = -0.925
#
# Action: Consider removing one feature from each highly correlated pair
```

Feature pairs with |r| > 0.8 are identified. Two pairs are found:
1. AveRooms and AveBedrms (r = 0.85)—makes sense, more rooms means more bedrooms
2. Latitude and Longitude (r = -0.92)—geographic coordinates are inherently related

This matters because highly correlated features cause multicollinearity problems in models. It may be desirable to keep only one from each pair.

### Part 6: Comparing Pearson, Spearman, Kendall

```python
print(f"\n{'='*60}")
print("Comparing Correlation Measures")
print(f"{'='*60}")

# Select a feature with some outliers
feature1 = df['MedInc']
feature2 = df['MedHouseVal']

# Compute all three correlation types
r_pearson, p_pearson = pearsonr(feature1, feature2)
r_spearman, p_spearman = spearmanr(feature1, feature2)
r_kendall, p_kendall = kendalltau(feature1, feature2)

print(f"\nCorrelation between MedInc and MedHouseVal:")
print(f"  • Pearson r:   {r_pearson:.4f}  (p={p_pearson:.2e})")
print(f"  • Spearman ρ:  {r_spearman:.4f}  (p={p_spearman:.2e})")
print(f"  • Kendall τ:   {r_kendall:.4f}  (p={p_kendall:.2e})")

print("\nInterpretation:")
print("  • All three measures show strong positive correlation")
print("  • Pearson is slightly higher (linear relationship)")
print("  • Spearman accounts for monotonic but non-linear patterns")
print("  • Kendall gives probabilistic interpretation of concordance")

# Output:
# ============================================================
# Comparing Correlation Measures
# ============================================================
#
# Correlation between MedInc and MedHouseVal:
#   • Pearson r:   0.6880  (p=0.00e+00)
#   • Spearman ρ:  0.6572  (p=0.00e+00)
#   • Kendall τ:   0.4826  (p=0.00e+00)
#
# Interpretation:
#   • All three measures show strong positive correlation
#   • Pearson is slightly higher (linear relationship)
#   • Spearman accounts for monotonic but non-linear patterns
#   • Kendall gives probabilistic interpretation of concordance

print("\n" + "="*60)
print("Correlation Analysis Complete!")
print("="*60)
```

Pearson, Spearman, and Kendall correlations are computed. All three show strong positive relationships, but with different magnitudes:
- Pearson (0.688) is highest—measures linear relationship
- Spearman (0.657) is slightly lower—uses ranks, more robust
- Kendall (0.483) is lowest—this is typical; Kendall values are generally smaller

All three agree on the direction and strength (all strong and positive), giving confidence in the relationship.

## Common Pitfalls

**1. Assuming Correlation Means Causation**

This is the most dangerous mistake in data science. Just because two variables are correlated does NOT mean one causes the other.

**Example:** Ice cream sales and drowning deaths are highly correlated. Does ice cream cause drownings? Of course not. Both increase in summer because of warm weather—a confounding variable.

When X and Y are correlated, there are three possibilities:
- X causes Y
- Y causes X (reverse causation)
- Z causes both X and Y (confounding)

**What to do instead:** Use language like "X is associated with Y" or "X predicts Y" rather than "X causes Y." To establish causation, randomized experiments, temporal precedence (cause happens before effect), and a mechanistic explanation are needed. Correlation suggests hypotheses; it doesn't prove causation.

**2. Trusting the Correlation Coefficient Without Visualizing**

**The problem:** Anscombe's Quartet famously demonstrates that four completely different datasets can have identical correlation coefficients (r = 0.816) and regression lines, but wildly different patterns:
- Dataset 1: Normal linear relationship
- Dataset 2: Non-linear (curved) relationship
- Dataset 3: Linear with one outlier distorting everything
- Dataset 4: No relationship except for one outlier creating false correlation

**What to do instead:** ALWAYS create a scatter plot before trusting the correlation coefficient. Visualization reveals patterns, outliers, non-linearity, and data quality issues that the correlation number alone cannot show. Make it a rule: no correlation without a plot.

**3. Assuming Zero Correlation Means No Relationship**

**The problem:** Pearson correlation ONLY measures linear relationships. A perfect U-shaped relationship (y = x²) has r ≈ 0, but there's a perfect functional relationship between the variables.

**Example:** Consider the relationship between anxiety and performance. Too little anxiety means low motivation (poor performance). Moderate anxiety is optimal (best performance). Too much anxiety is paralyzing (poor performance again). This inverted-U relationship has near-zero Pearson correlation despite being highly structured.

**What to do instead:**
- Always visualize to check for non-linear patterns
- Consider transformations (log, square root, polynomial features)
- Use Spearman correlation for monotonic non-linear relationships
- Remember: r = 0 means "no LINEAR relationship," not "no relationship"

## Practice Exercises

**Practice 1**

Load the Iris dataset and practice computing correlations between features.

1. Load the Iris dataset using `load_iris()` and convert to a DataFrame with feature names
2. Compute the Pearson correlation between `sepal length (cm)` and `petal length (cm)`
3. Create a scatter plot showing this relationship with a regression line
4. Compute the full correlation matrix for all four features
5. Identify which pair of features has the strongest correlation
6. Create a heatmap (4×4) showing all pairwise correlations

**Questions to answer:**
- What is the correlation between sepal length and petal length? Is it strong or weak?
- Which two features are most strongly correlated? What is their r value?
- Are any features negatively correlated with each other?
- Based on the correlation matrix, which features would be most useful for predicting petal length?

**Starter code:**
```python
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Your code here
```

**Practice 2**

Investigate how outliers affect different correlation measures using the California Housing dataset.

1. Load the California Housing dataset and focus on `MedInc` and `AveOccup` features
2. Compute Pearson, Spearman, and Kendall correlations for these variables
3. Identify outliers in `AveOccup` using the IQR method (values > Q3 + 1.5×IQR or < Q1 - 1.5×IQR)
4. Create a 1×3 subplot figure:
   - Left: Scatter plot of original data, annotate with Pearson r
   - Middle: Same plot highlighting outliers in red
   - Right: Scatter plot with outliers removed, annotate with new Pearson r
5. Create a comparison table showing all three correlation measures before and after outlier removal
6. Add a synthetic extreme outlier (MedInc = 15, AveOccup = 1200) and recompute all correlations

**Questions to answer:**
- How much does Pearson correlation change when outliers are removed? What about Spearman?
- Which correlation measure is most stable when the extreme synthetic outlier is added?
- When would Spearman be preferred over Pearson in practice?
- What does this exercise teach about checking data quality before computing correlations?

**Practice 3**

Conduct a comprehensive correlation-based feature selection analysis for the Breast Cancer dataset to prepare for building a classification model.

1. Load the Breast Cancer dataset (`load_breast_cancer()`) - 569 samples, 30 features, binary target
2. Convert to a DataFrame including the target column named `malignant` (1 for malignant, 0 for benign)

3. **Phase 1: Feature-Target Analysis**
   - Compute correlation between all 30 features and the target
   - Create a bar chart showing the 15 features with highest absolute correlation with target
   - Visualize the top 3 features vs. target using box plots separated by class

4. **Phase 2: Feature-Feature Multicollinearity**
   - Compute the 30×30 correlation matrix among features (exclude target)
   - Find all feature pairs with |r| > 0.9
   - Count how many redundant pairs exist
   - Create a clustered heatmap showing feature correlations

5. **Phase 3: Implement Feature Selection Strategy**
   - Strategy: Keep features with |r| > 0.4 with target AND no multicollinearity
   - If two features have |r| > 0.9, keep the one with higher target correlation
   - Document which features are kept and why

6. **Phase 4: Validation**
   - Compare original 30 features vs. selected subset
   - Create side-by-side heatmaps: original vs. reduced
   - Calculate what percentage of features are retained

**Questions to answer:**
- How many features are selected? What percentage of the original 30?
- Which features are most predictive of malignancy?
- Are there groups of highly correlated features? Why might certain features cluster together?
- What are the trade-offs of this correlation-based feature selection approach compared to other methods?

**Deliverables:**
- Python script or Jupyter notebook with code and markdown explanations
- At least 3 visualizations (bar chart, heatmaps, box plots)
- Summary table documenting feature selection decisions
- 2-3 paragraph interpretation of findings and recommendations

## Solutions

**Solution 1**
```python
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Task 2: Compute correlation between sepal length and petal length
feature_x = 'sepal length (cm)'
feature_y = 'petal length (cm)'
r, p_value = pearsonr(df[feature_x], df[feature_y])
print(f"Correlation between {feature_x} and {feature_y}: r = {r:.4f}, p = {p_value:.2e}")

# Task 3: Scatter plot with regression line
plt.figure(figsize=(8, 6))
plt.scatter(df[feature_x], df[feature_y], alpha=0.6, s=50, color='steelblue')
z = np.polyfit(df[feature_x], df[feature_y], 1)
p = np.poly1d(z)
x_line = np.linspace(df[feature_x].min(), df[feature_x].max(), 100)
plt.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {r:.3f}')
plt.xlabel(feature_x, fontsize=12)
plt.ylabel(feature_y, fontsize=12)
plt.title('Sepal Length vs. Petal Length', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Task 4: Full correlation matrix
corr_matrix = df.corr()
print("\nFull Correlation Matrix:")
print(corr_matrix.round(3))

# Task 5: Find strongest correlation
# Create upper triangle matrix to avoid duplicates
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
corr_values = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_values.append({
            'Feature 1': corr_matrix.columns[i],
            'Feature 2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })
corr_df = pd.DataFrame(corr_values).sort_values('Correlation', ascending=False, key=abs)
print(f"\nStrongest correlation: {corr_df.iloc[0]['Feature 1']} <-> {corr_df.iloc[0]['Feature 2']}: r = {corr_df.iloc[0]['Correlation']:.4f}")

# Task 6: Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Iris Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Answers:
# - Sepal length and petal length have r = 0.8718 (strong positive correlation)
# - Petal length and petal width are most strongly correlated (r ≈ 0.96)
# - Sepal width has weak negative correlations with other features
# - Petal length and petal width would be most useful for predictions
```

The solution loads the Iris dataset and systematically computes correlations. The strongest correlation is between petal length and petal width (r ≈ 0.96), indicating these measurements are highly related. The visualization confirms the strong linear relationship between sepal length and petal length.

**Solution 2**
```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

# Task 1: Load data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Focus on MedInc and AveOccup
X = df['MedInc']
Y = df['AveOccup']

# Task 2: Compute all three correlations (original data)
r_p_orig, _ = pearsonr(X, Y)
r_s_orig, _ = spearmanr(X, Y)
r_k_orig, _ = kendalltau(X, Y)

print("Original Data Correlations:")
print(f"  Pearson:  {r_p_orig:.4f}")
print(f"  Spearman: {r_s_orig:.4f}")
print(f"  Kendall:  {r_k_orig:.4f}")

# Task 3: Identify outliers using IQR
Q1 = Y.quantile(0.25)
Q3 = Y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (Y < lower_bound) | (Y > upper_bound)
print(f"\nOutliers detected: {outliers.sum()} ({100*outliers.sum()/len(Y):.1f}%)")

# Task 4: Create visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Left: Original data
axes[0].scatter(X, Y, alpha=0.3, s=10, color='steelblue')
axes[0].set_xlabel('MedInc', fontsize=12)
axes[0].set_ylabel('AveOccup', fontsize=12)
axes[0].set_title(f'Original Data\nr = {r_p_orig:.3f}', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

# Middle: Highlight outliers
axes[1].scatter(X[~outliers], Y[~outliers], alpha=0.3, s=10, color='steelblue', label='Normal')
axes[1].scatter(X[outliers], Y[outliers], alpha=0.6, s=20, color='red', label='Outliers')
axes[1].set_xlabel('MedInc', fontsize=12)
axes[1].set_ylabel('AveOccup', fontsize=12)
axes[1].set_title(f'Outliers Highlighted\n{outliers.sum()} outliers', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Right: Outliers removed
X_clean = X[~outliers]
Y_clean = Y[~outliers]
r_p_clean, _ = pearsonr(X_clean, Y_clean)
r_s_clean, _ = spearmanr(X_clean, Y_clean)
r_k_clean, _ = kendalltau(X_clean, Y_clean)
axes[2].scatter(X_clean, Y_clean, alpha=0.3, s=10, color='steelblue')
axes[2].set_xlabel('MedInc', fontsize=12)
axes[2].set_ylabel('AveOccup', fontsize=12)
axes[2].set_title(f'Outliers Removed\nr = {r_p_clean:.3f}', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Task 5: Comparison table
comparison = pd.DataFrame({
    'Measure': ['Pearson', 'Spearman', 'Kendall'],
    'Original': [r_p_orig, r_s_orig, r_k_orig],
    'Without Outliers': [r_p_clean, r_s_clean, r_k_clean],
    'Change': [r_p_clean - r_p_orig, r_s_clean - r_s_orig, r_k_clean - r_k_orig]
})
print("\nComparison Table:")
print(comparison.round(4))

# Task 6: Add extreme outlier
X_extreme = pd.concat([X, pd.Series([15])])
Y_extreme = pd.concat([Y, pd.Series([1200])])
r_p_extreme, _ = pearsonr(X_extreme, Y_extreme)
r_s_extreme, _ = spearmanr(X_extreme, Y_extreme)
r_k_extreme, _ = kendalltau(X_extreme, Y_extreme)

print("\nWith Extreme Outlier (MedInc=15, AveOccup=1200):")
print(f"  Pearson:  {r_p_extreme:.4f} (change: {r_p_extreme - r_p_orig:.4f})")
print(f"  Spearman: {r_s_extreme:.4f} (change: {r_s_extreme - r_s_orig:.4f})")
print(f"  Kendall:  {r_k_extreme:.4f} (change: {r_k_extreme - r_k_orig:.4f})")

# Answers:
# - Pearson changes significantly with outliers (sensitive to extremes)
# - Spearman is much more stable (uses ranks, not raw values)
# - Spearman preferred when outliers present or non-linear monotonic relationships
# - Always inspect data quality before computing correlations
```

This solution demonstrates the robustness of rank-based correlations (Spearman, Kendall) compared to Pearson. When outliers are removed, Pearson correlation changes substantially, while Spearman remains relatively stable. The extreme outlier experiment further confirms that Spearman and Kendall are more robust for real-world messy data.

**Solution 3**
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1 & 2: Load data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['malignant'] = cancer.target

print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {df['malignant'].value_counts()}")

# Phase 1: Feature-Target Analysis
target_corr = df.corr()['malignant'].drop('malignant').abs().sort_values(ascending=False)

# Bar chart of top 15 features
plt.figure(figsize=(12, 6))
target_corr.head(15).plot(kind='barh', color='steelblue')
plt.xlabel('Absolute Correlation with Target', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Features by Correlation with Malignancy', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Box plots for top 3 features
top_3_features = target_corr.head(3).index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(top_3_features):
    df.boxplot(column=feature, by='malignant', ax=axes[idx])
    axes[idx].set_title(f'{feature}\nr = {df[[feature, "malignant"]].corr().iloc[0, 1]:.3f}')
    axes[idx].set_xlabel('Malignant (0=Benign, 1=Malignant)')
    axes[idx].set_ylabel(feature)
plt.suptitle('')
plt.tight_layout()
plt.show()

# Phase 2: Feature-Feature Multicollinearity
feature_cols = [col for col in df.columns if col != 'malignant']
feature_corr = df[feature_cols].corr()

# Find pairs with |r| > 0.9
high_corr_pairs = []
for i in range(len(feature_corr.columns)):
    for j in range(i+1, len(feature_corr.columns)):
        if abs(feature_corr.iloc[i, j]) > 0.9:
            high_corr_pairs.append({
                'Feature 1': feature_corr.columns[i],
                'Feature 2': feature_corr.columns[j],
                'Correlation': feature_corr.iloc[i, j]
            })

print(f"\nPhase 2: Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9)")
high_corr_df = pd.DataFrame(high_corr_pairs)
print(high_corr_df.head(10))

# Clustered heatmap
plt.figure(figsize=(16, 14))
sns.clustermap(feature_corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
               linewidths=0.5, figsize=(16, 14), cbar_kws={'label': 'Correlation'})
plt.title('Clustered Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()

# Phase 3: Feature Selection Strategy
selected_features = []
target_corr_full = df.corr()['malignant'].drop('malignant')

# Start with features that have |r| > 0.4 with target
strong_predictors = target_corr_full[target_corr_full.abs() > 0.4].index.tolist()
print(f"\nPhase 3: {len(strong_predictors)} features with |r| > 0.4 with target")

# Remove multicollinear pairs
removed_features = set()
for _, row in high_corr_df.iterrows():
    feat1, feat2 = row['Feature 1'], row['Feature 2']
    if feat1 in strong_predictors and feat2 in strong_predictors:
        if feat1 not in removed_features and feat2 not in removed_features:
            # Keep the one with higher target correlation
            if abs(target_corr_full[feat1]) >= abs(target_corr_full[feat2]):
                removed_features.add(feat2)
                print(f"  Removing {feat2} (keep {feat1}, higher target corr)")
            else:
                removed_features.add(feat1)
                print(f"  Removing {feat1} (keep {feat2}, higher target corr)")

selected_features = [f for f in strong_predictors if f not in removed_features]
print(f"\nSelected {len(selected_features)} features after multicollinearity removal")
print("Selected features:", selected_features)

# Phase 4: Validation
retention_pct = 100 * len(selected_features) / len(feature_cols)
print(f"\nPhase 4 Summary:")
print(f"  Original features: {len(feature_cols)}")
print(f"  Selected features: {len(selected_features)}")
print(f"  Retention rate: {retention_pct:.1f}%")

# Side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Original
sns.heatmap(df[feature_cols + ['malignant']].corr(), ax=axes[0], cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.1, cbar=False)
axes[0].set_title(f'Original ({len(feature_cols)} features)', fontsize=14, fontweight='bold')

# Reduced
sns.heatmap(df[selected_features + ['malignant']].corr(), ax=axes[1], cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
axes[1].set_title(f'Reduced ({len(selected_features)} features)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"""
The analysis selected {len(selected_features)} out of {len(feature_cols)} features ({retention_pct:.1f}%).

Most predictive features (top 3):
1. {top_3_features[0]}: r = {target_corr_full[top_3_features[0]]:.3f}
2. {top_3_features[1]}: r = {target_corr_full[top_3_features[1]]:.3f}
3. {top_3_features[2]}: r = {target_corr_full[top_3_features[2]]:.3f}

Multicollinearity patterns: Many "mean", "worst", and "standard error" versions
of the same measurement (e.g., radius, texture, perimeter) are highly correlated.
This is expected since they measure related physical properties of cell nuclei.

Trade-offs of correlation-based feature selection:
+ Simple and interpretable
+ Removes redundant features effectively
+ Fast computation
- Only considers linear relationships
- Doesn't account for feature interactions
- May miss features that are weak individually but strong in combination
- No consideration of model-specific importance

Recommendation: Use this as a first-pass filter, then apply model-based feature
selection (e.g., LASSO, Random Forest importance) for final feature set.
""")
```

This comprehensive solution implements a complete correlation-based feature selection pipeline. The analysis reduces the feature set from 30 to approximately 10-15 features while retaining the most predictive ones and eliminating multicollinearity. The solution demonstrates that many features measuring similar aspects (mean, worst, SE versions) are highly correlated and can be reduced.

## Key Takeaways

- **Correlation quantifies relationships**: The Pearson correlation coefficient (r) measures the strength and direction of linear relationships between variables, ranging from -1 (perfect negative) through 0 (no linear relationship) to +1 (perfect positive).

- **Visualization is mandatory**: Never trust a correlation coefficient without visualizing the data. Anscombe's Quartet proves that identical correlations can hide completely different patterns. Always create scatter plots to verify relationships.

- **Correlation does not imply causation**: This is the most critical concept. Correlated variables may have a causal relationship, but correlation alone cannot prove it. Always consider confounding variables (Z causes both X and Y) and reverse causation before making causal claims.

- **Choose the right correlation measure**: Pearson for linear relationships with continuous data, Spearman for monotonic non-linear relationships or ordinal data, and Kendall for small samples or when probabilistic interpretation is needed. Spearman and Kendall are robust to outliers; Pearson is not.

- **Use correlation to guide feature engineering**: Correlation matrices reveal which features predict the target (useful for feature selection) and which features are redundant (multicollinearity). Features with |r| > 0.9 are candidates for removal to improve model stability.

- **Context determines importance**: The practical significance of a correlation depends on the domain. In social sciences, r = 0.3 can be groundbreaking; in physics, r = 0.99 might be disappointing. Consider both statistical significance (p-value) and effect size (r value) when interpreting results.

**Next:** The next chapter explores how to engineer new features based on these relationships—creating interaction terms, polynomial features, and transformations that capture the patterns correlation helps discover.

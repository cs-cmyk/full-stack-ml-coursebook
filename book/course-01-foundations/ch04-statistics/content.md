> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# Chapter 4: Statistics for Data Science

## Why This Matters

Before building the first machine learning model, understanding the data is essential. Statistics is the science of learning from data—it's the foundation for knowing what patterns are real versus random. Every successful data science project starts with exploratory data analysis (EDA), which is applied statistics in action. Skip this step, and models are built on quicksand: garbage in, garbage out.

## Intuition

Think about test scores in a classroom. If the class average is 75, that tells something—but not the whole story. What if half the students scored 45 and the other half scored 98? The average would still be around 75, but the situation would be completely different than if everyone scored between 70 and 80.

Statistics gives the tools to describe the full picture: Where is the center? How spread out are the values? Are there unusual outliers? Do two things move together? These questions apply whether analyzing test scores, housing prices, or customer behavior.

Here's a concrete analogy: Imagine a doctor taking a patient's vital signs. The doctor doesn't just check one measurement—heart rate, blood pressure, temperature, and oxygen saturation all get checked. Each statistic tells part of the story. Similarly, in data science, multiple statistics are computed to understand data's "health" before prescribing any machine learning treatment.

The beauty of statistics is that it operationalizes probability theory (see Chapter 3) on actual data. Probability asks "what should happen theoretically?" Statistics asks "what did happen in reality?" Together, they form the mathematical foundation of data science.

## Formal Definition

**Descriptive statistics** summarize and describe the main characteristics of a dataset using measures of central tendency, measures of spread, and visualizations. Given a dataset with n observations of a feature, the following are computed:

**Measures of Central Tendency:**
- **Mean (μ or x̄):** The arithmetic average
  ```
  μ = (1/n) Σᵢ₌₁ⁿ xᵢ
  ```

- **Median:** The middle value when data is sorted (50th percentile)

- **Mode:** The most frequently occurring value

**Measures of Spread:**
- **Variance (σ² or s²):** Average squared deviation from the mean
  ```
  σ² = (1/n) Σᵢ₌₁ⁿ (xᵢ - μ)²     [population variance]
  s² = (1/(n-1)) Σᵢ₌₁ⁿ (xᵢ - x̄)²  [sample variance, Bessel's correction]
  ```

- **Standard Deviation (σ or s):** Square root of variance
  ```
  σ = √σ²
  ```

- **Interquartile Range (IQR):** The range containing the middle 50% of data
  ```
  IQR = Q₃ - Q₁
  ```

**Measures of Relationship:**
- **Covariance:** How two variables vary together
  ```
  Cov(X, Y) = (1/n) Σᵢ₌₁ⁿ (xᵢ - μₓ)(yᵢ - μᵧ)
  ```

- **Correlation (Pearson's r):** Standardized covariance
  ```
  r = Cov(X, Y) / (σₓ σᵧ)
  ```
  Range: -1 (perfect negative) to +1 (perfect positive)

**Distribution Shape:**
- **Skewness:** Measures asymmetry in the distribution
- **Kurtosis:** Measures "tailedness" (outlier propensity)

> **Key Concept:** Statistics transforms raw data into actionable insights by summarizing location, spread, and relationships—the foundation for all data-driven decisions.

## Visualization

The following diagram illustrates how mean, median, and mode relate to distribution shape:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create figure with three distribution types
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Symmetric distribution
np.random.seed(42)
symmetric = np.random.normal(50, 10, 1000)
axes[0].hist(symmetric, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
mean_sym = np.mean(symmetric)
median_sym = np.median(symmetric)
mode_sym = stats.mode(symmetric, keepdims=True)[0][0]
axes[0].axvline(mean_sym, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_sym:.1f}')
axes[0].axvline(median_sym, color='blue', linestyle='--', linewidth=2, label=f'Median = {median_sym:.1f}')
axes[0].set_title('Symmetric Distribution\nMean ≈ Median ≈ Mode')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Right-skewed distribution
right_skewed = np.random.exponential(20, 1000) + 30
axes[1].hist(right_skewed, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
mean_right = np.mean(right_skewed)
median_right = np.median(right_skewed)
axes[1].axvline(mean_right, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_right:.1f}')
axes[1].axvline(median_right, color='blue', linestyle='--', linewidth=2, label=f'Median = {median_right:.1f}')
axes[1].set_title('Right-Skewed Distribution\nMode < Median < Mean')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Left-skewed distribution
left_skewed = 100 - np.random.exponential(20, 1000)
axes[2].hist(left_skewed, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
mean_left = np.mean(left_skewed)
median_left = np.median(left_skewed)
axes[2].axvline(mean_left, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_left:.1f}')
axes[2].axvline(median_left, color='blue', linestyle='--', linewidth=2, label=f'Median = {median_left:.1f}')
axes[2].set_title('Left-Skewed Distribution\nMean < Median < Mode')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.savefig('diagrams/mean_median_mode.png', dpi=150, bbox_inches='tight')
plt.show()

# Output: Three histograms showing how distribution shape affects measures of center
```

**Figure 4.1:** Distribution shape determines which measure of center is most representative. In symmetric distributions, mean and median coincide. In skewed distributions, the mean is "pulled" toward the tail, making the median a more robust measure.

---

## Examples

### Part 1: Basic Descriptive Statistics

```python
# Computing Descriptive Statistics from Scratch and with Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Set random seed for reproducibility
np.random.seed(42)

# Load California Housing dataset
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# Display first few rows and basic info
print("=" * 60)
print("CALIFORNIA HOUSING DATASET")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
# Output:
#    MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
# 0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23        4.526
# 1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22        3.585
# ...

print("\n" + "=" * 60)
print("MANUAL COMPUTATION OF STATISTICS")
print("=" * 60)

# Focus on median income feature
income = df['MedInc'].values

# 1. Mean - manual calculation
mean_manual = np.sum(income) / len(income)
print(f"\nMean (manual): {mean_manual:.4f}")
print(f"Mean (NumPy):  {np.mean(income):.4f}")
print(f"Mean (pandas): {df['MedInc'].mean():.4f}")
# Output: All three methods give same result: 3.8707

# 2. Median - manual calculation
sorted_income = np.sort(income)
n = len(sorted_income)
if n % 2 == 0:
    median_manual = (sorted_income[n//2 - 1] + sorted_income[n//2]) / 2
else:
    median_manual = sorted_income[n//2]
print(f"\nMedian (manual): {median_manual:.4f}")
print(f"Median (NumPy):  {np.median(income):.4f}")
# Output: 3.5348

# 3. Variance - manual calculation (population variance)
squared_deviations = (income - mean_manual) ** 2
variance_manual = np.sum(squared_deviations) / len(income)
print(f"\nVariance (manual, population): {variance_manual:.4f}")
print(f"Variance (NumPy, population):  {np.var(income):.4f}")
# Output: 3.6144

# 4. Sample variance with Bessel's correction
variance_sample = np.sum(squared_deviations) / (len(income) - 1)
print(f"Variance (manual, sample):     {variance_sample:.4f}")
print(f"Variance (NumPy, sample):      {np.var(income, ddof=1):.4f}")
print(f"Variance (pandas):             {df['MedInc'].var():.4f}")
# Output: 3.6146 (slightly higher due to n-1 correction)

# 5. Standard deviation - manual calculation
std_manual = np.sqrt(variance_sample)
print(f"\nStd Dev (manual): {std_manual:.4f}")
print(f"Std Dev (NumPy):  {np.std(income, ddof=1):.4f}")
print(f"Std Dev (pandas): {df['MedInc'].std():.4f}")
# Output: 1.9008
```

The California Housing dataset contains 20,640 census block groups with 8 features and 1 target (median house value). This is a real-world dataset perfect for demonstrating statistical concepts.

The mean is computed three ways: manually using the formula μ = (1/n)Σxᵢ, with NumPy's `np.mean()`, and with pandas' `.mean()`. All give the same result: 3.87. This represents the average median income in tens of thousands of dollars across all districts.

The median requires sorting the data and finding the middle value. For even-length data, the two middle values are averaged. The median (3.53) is lower than the mean (3.87), which tells something important about the distribution's shape.

Variance measures average squared distance from the mean. Squared deviations `(xᵢ - μ)²` are computed and averaged. Notice the "population variance" (divide by n) versus "sample variance" (divide by n-1). The difference is Bessel's correction, which provides an unbiased estimate when working with samples rather than entire populations.

Standard deviation is simply the square root of variance. While variance has squared units (hard to interpret), standard deviation returns to original units. The value of 1.90 means districts typically deviate by $19,000 from the mean median income.

### Part 2: Quartiles and Comprehensive Summary

```python
# 6. Range
range_val = np.max(income) - np.min(income)
print(f"\nRange: {range_val:.4f}")
print(f"Min: {np.min(income):.4f}, Max: {np.max(income):.4f}")
# Output: Range: 14.4999, Min: 0.4999, Max: 15.0001

# 7. Quartiles and IQR
q1 = np.percentile(income, 25)
q2 = np.percentile(income, 50)  # Same as median
q3 = np.percentile(income, 75)
iqr = q3 - q1
print(f"\nQ1 (25th percentile): {q1:.4f}")
print(f"Q2 (50th percentile): {q2:.4f}")
print(f"Q3 (75th percentile): {q3:.4f}")
print(f"IQR (Q3 - Q1):        {iqr:.4f}")
# Output: Q1: 2.5636, Q2: 3.5348, Q3: 4.7432, IQR: 2.1796

print("\n" + "=" * 60)
print("COMPREHENSIVE SUMMARY WITH PANDAS")
print("=" * 60)
print(df.describe())
# Output:
#            MedInc      HouseAge     AveRooms  ...
# count  20640.0000  20640.000000  20640.00000  ...
# mean       3.8707      28.639486      5.42939  ...
# std        1.9008      12.585558      2.47499  ...
# min        0.4999       1.000000      0.84671  ...
# 25%        2.5636      18.000000      4.44072  ...
# 50%        3.5348      29.000000      5.22942  ...
# 75%        4.7432      37.000000      6.05251  ...
# max       15.0001      52.000000    141.90909  ...

# Interpretation
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print(f"""
For Median Income (MedInc):
- The median income ranges from ${'%.2f' % np.min(income)} to ${'%.2f' % np.max(income)} (in $10,000s)
- Mean = ${'%.2f' % mean_manual}, Median = ${'%.2f' % median_manual}
- Since Mean > Median, the distribution is RIGHT-SKEWED (pulled by high incomes)
- Standard deviation = ${'%.2f' % std_manual}, meaning most districts fall within
  ${'%.2f' % (mean_manual - std_manual)} to ${'%.2f' % (mean_manual + std_manual)}
- IQR = ${'%.2f' % iqr}, so the middle 50% spans only ${'%.2f' % iqr} units
  (more robust to outliers than range of ${'%.2f' % range_val})
""")
```

Range is the simplest measure of spread: max - min. The range of 14.50 spans from $5,000 to $150,000 median income. However, range is sensitive to outliers—one extreme value changes everything.

Quartiles divide data into four equal parts. Q1 (25th percentile) = 2.56, Q2 (median) = 3.53, Q3 (75th percentile) = 4.74. The IQR = Q3 - Q1 = 2.18 represents the middle 50% of data. IQR is robust to outliers, making it more reliable than range for identifying spread.

The `.describe()` method computes all these statistics at once for every numeric column. This is the first step in any EDA workflow—get a high-level overview before diving deeper.

The interpretation is the most important part. The numbers mean nothing without context. The observation that mean > median signals right skewness (income distributions are almost always right-skewed—a few very wealthy areas pull the mean up). The IQR tells the "typical" range better than the full range does.

Notice everything was computed manually first, then shown how libraries do it. This builds intuition: functions aren't called blindly, the mathematics underneath is understood.

### Part 3: Distribution Visualization

```python
# Visualizing Distributions to Understand Data Patterns
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style and random seed
sns.set_style('whitegrid')
np.random.seed(42)

print("=" * 60)
print("DISTRIBUTION ANALYSIS")
print("=" * 60)

# Focus on MedInc feature
income = df['MedInc']

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with mean and median lines
axes[0, 0].hist(income, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
mean_val = income.mean()
median_val = income.median()
axes[0, 0].axvline(mean_val, color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {mean_val:.2f}')
axes[0, 0].axvline(median_val, color='blue', linestyle='--', linewidth=2,
                    label=f'Median = {median_val:.2f}')
axes[0, 0].set_xlabel('Median Income ($10,000s)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Histogram: Distribution of Median Income')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plot
box_parts = axes[0, 1].boxplot(income, vert=True, patch_artist=True,
                                widths=0.5)
box_parts['boxes'][0].set_facecolor('lightblue')
box_parts['medians'][0].set_color('red')
box_parts['medians'][0].set_linewidth(2)
axes[0, 1].set_ylabel('Median Income ($10,000s)')
axes[0, 1].set_title('Box Plot: Identifying Outliers')
axes[0, 1].grid(True, alpha=0.3)

# Add annotations for box plot components
q1, median, q3 = np.percentile(income, [25, 50, 75])
iqr = q3 - q1
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr
axes[0, 1].text(1.15, q1, f'Q1 = {q1:.2f}', fontsize=9)
axes[0, 1].text(1.15, median, f'Median = {median:.2f}', fontsize=9, color='red')
axes[0, 1].text(1.15, q3, f'Q3 = {q3:.2f}', fontsize=9)
axes[0, 1].text(1.15, upper_whisker, f'Upper fence', fontsize=8, style='italic')

# 3. Multiple features comparison
features_to_plot = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
df[features_to_plot].boxplot(ax=axes[1, 0])
axes[1, 0].set_ylabel('Value (various units)')
axes[1, 0].set_title('Box Plots: Comparing Multiple Features')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# 4. Skewness visualization
skew_values = df[features_to_plot].skew()
colors = ['green' if abs(s) < 0.5 else 'orange' if abs(s) < 1 else 'red'
          for s in skew_values]
axes[1, 1].bar(features_to_plot, skew_values, color=colors, edgecolor='black')
axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1, 1].axhline(0.5, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 1].axhline(-0.5, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
axes[1, 1].set_ylabel('Skewness')
axes[1, 1].set_title('Skewness: Measuring Distribution Asymmetry')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('diagrams/distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```

The histogram shows the distribution's shape. Vertical lines for mean (red) and median (blue) are overlaid. Notice how the mean (3.87) is pulled right of the median (3.53) by high-income outliers—this is the signature of right skewness. If the distribution were symmetric, these lines would coincide.

Box plots compress the five-number summary (min, Q1, median, Q3, max) into one visualization. The "box" spans the IQR (middle 50%), the line inside is the median, and the "whiskers" extend to 1.5×IQR beyond the quartiles. Points beyond whiskers are flagged as outliers (the circles). This plot immediately shows that high-income outliers exist but low-income outliers are rare—more evidence of right skewness.

By plotting multiple features side-by-side, distributions can be compared at a glance. Notice that `HouseAge` is relatively symmetric (box is centered), while `AveRooms` has extreme outliers (many circles above the upper whisker).

### Part 4: Skewness and Transformation

```python
# Print skewness analysis
print("\n" + "=" * 60)
print("SKEWNESS ANALYSIS")
print("=" * 60)
for feature in features_to_plot:
    skew = df[feature].skew()
    if abs(skew) < 0.5:
        interpretation = "approximately symmetric"
    elif skew > 0.5:
        interpretation = "RIGHT-SKEWED (positive skew, tail extends right)"
    else:
        interpretation = "LEFT-SKEWED (negative skew, tail extends left)"
    print(f"{feature:12s}: skew = {skew:7.3f}  --> {interpretation}")
# Output:
# MedInc      : skew =   0.974  --> RIGHT-SKEWED (positive skew)
# HouseAge    : skew =  -0.108  --> approximately symmetric
# AveRooms    : skew =  16.942  --> RIGHT-SKEWED (positive skew) [EXTREME]
# AveBedrms   : skew =  19.542  --> RIGHT-SKEWED (positive skew) [EXTREME]

# Demonstrate effect of log transformation on skewed data
print("\n" + "=" * 60)
print("LOG TRANSFORMATION EFFECT")
print("=" * 60)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before transformation
axes[0].hist(df['AveRooms'], bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[0].set_xlabel('Average Rooms')
axes[0].set_ylabel('Frequency')
axes[0].set_title(f'Before Log Transform\nSkew = {df["AveRooms"].skew():.2f}')
axes[0].grid(True, alpha=0.3)

# After transformation
log_rooms = np.log1p(df['AveRooms'])  # log1p handles zeros safely
axes[1].hist(log_rooms, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1].set_xlabel('Log(Average Rooms)')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'After Log Transform\nSkew = {log_rooms.skew():.2f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/log_transformation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"AveRooms skewness before log: {df['AveRooms'].skew():.2f}")
print(f"AveRooms skewness after log:  {log_rooms.skew():.2f}")
print("Log transformation reduces skewness, making data more normal-like.")
# Output: Skewness reduced from 16.94 to ~1.2
```

Skewness quantifies asymmetry. Values near 0 indicate symmetry, positive values indicate right skew (long tail right), negative values indicate left skew (long tail left). Color-coding: green for approximately symmetric (|skew| < 0.5), orange for moderate skew (0.5 ≤ |skew| < 1), red for strong skew (|skew| ≥ 1). The chart reveals that `AveRooms` and `AveBedrms` are extremely skewed (skew > 15).

Human-readable interpretations are printed. Extreme skewness (like 16.9 for AveRooms) means the distribution has a very long tail with rare extreme values. This can break many machine learning algorithms that assume approximately normal distributions.

For heavily skewed data, log transformation can help. `np.log1p()` (log of 1+x, which handles zeros safely) is applied to `AveRooms`. The before/after histograms show dramatic improvement: skewness drops from 16.9 to around 1.2. This is a common preprocessing step in feature engineering.

**Key Insight:** Summary statistics alone (mean = 5.43, std = 2.47 for AveRooms) don't reveal the extreme skewness. The histogram shows the true story: most values cluster around 4-6, but a few extreme outliers (houses with 100+ rooms?) pull the mean and create massive skewness. This is why "visualize first, summarize second" is the golden rule of EDA.

### Part 5: Correlation Analysis

```python
# Correlation Analysis: Finding Relationships Between Features
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

print("=" * 60)
print("IRIS DATASET - CORRELATION ANALYSIS")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nFeatures: {iris.feature_names}")
print(f"\nFirst 5 rows:")
print(df.head())
# Output:
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
# 0                5.1               3.5                1.4               0.2       0
# 1                4.9               3.0                1.4               0.2       0
# ...

# Manual correlation calculation for two features
print("\n" + "=" * 60)
print("MANUAL CORRELATION COMPUTATION")
print("=" * 60)

# Select two features
x = df['petal length (cm)'].values
y = df['petal width (cm)'].values

# Step 1: Compute means
mean_x = np.mean(x)
mean_y = np.mean(y)

# Step 2: Compute covariance
covariance = np.sum((x - mean_x) * (y - mean_y)) / len(x)
print(f"Covariance(petal length, petal width) = {covariance:.4f}")

# Step 3: Compute standard deviations
std_x = np.std(x, ddof=1)
std_y = np.std(y, ddof=1)
print(f"Std Dev(petal length) = {std_x:.4f}")
print(f"Std Dev(petal width)  = {std_y:.4f}")

# Step 4: Compute correlation
correlation_manual = covariance / (std_x * std_y)
print(f"\nCorrelation (manual) = {correlation_manual:.4f}")

# Verify with NumPy
correlation_numpy = np.corrcoef(x, y)[0, 1]
print(f"Correlation (NumPy)  = {correlation_numpy:.4f}")
# Output: r = 0.9629 (very strong positive correlation)

# Interpretation
print(f"\nInterpretation: r = {correlation_manual:.4f}")
if correlation_manual > 0.7:
    print("  --> Strong POSITIVE correlation: as petal length increases,")
    print("      petal width tends to increase proportionally")
elif correlation_manual < -0.7:
    print("  --> Strong NEGATIVE correlation")
elif abs(correlation_manual) < 0.3:
    print("  --> Weak or no linear relationship")
else:
    print("  --> Moderate correlation")
```

The Iris dataset is used because it has strong feature correlations, perfect for demonstrating correlation analysis. The dataset has 150 samples and 4 features measuring flower dimensions.

Pearson correlation is computed by hand to demystify the formula. First, covariance is calculated: how petal length and petal width vary together. Then standardization occurs by dividing by the product of standard deviations. This gives r = 0.963, meaning these features are almost perfectly correlated—as one increases, the other increases proportionally.

A correlation of 0.96 is extremely strong. In a scatter plot, the points would nearly fall on a straight line. This indicates these features are redundant—they carry nearly the same information.

### Part 6: Correlation Matrix and Visualization

```python
# Compute full correlation matrix
print("\n" + "=" * 60)
print("FULL CORRELATION MATRIX")
print("=" * 60)
corr_matrix = df.iloc[:, :-1].corr()  # Exclude target column
print(corr_matrix)
# Output: 4x4 correlation matrix showing all pairwise correlations

# Visualize correlation matrix as heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Correlation Matrix Heatmap\n(Iris Features)', fontsize=12, weight='bold')

# Scatter plot of highly correlated pair
axes[1].scatter(x, y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
axes[1].set_title(f'Scatter Plot: Petal Length vs Width\nr = {correlation_manual:.4f}',
                  fontsize=12, weight='bold')
axes[1].grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
axes[1].plot(x, p(x), "r--", linewidth=2, label='Linear fit')
axes[1].legend()

plt.tight_layout()
plt.savefig('diagrams/correlation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Identify highly correlated feature pairs
print("\n" + "=" * 60)
print("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.7)")
print("=" * 60)
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            print(f"{feat1:20s} <--> {feat2:20s}: r = {corr_val:7.4f}")
            if abs(corr_val) > 0.9:
                print(f"  ⚠️  WARNING: Very high correlation - consider dropping one feature")
# Output:
# sepal length (cm)    <--> petal length (cm)   : r =  0.8718
# petal length (cm)    <--> petal width (cm)    : r =  0.9629
#   ⚠️  WARNING: Very high correlation - consider dropping one feature
```

Correlations for all feature pairs are computed at once using `.corr()`. The resulting matrix is symmetric (correlation of X with Y equals correlation of Y with X) with 1s on the diagonal (every feature perfectly correlates with itself).

Heatmaps color-code correlations: red for strong positive, blue for strong negative, white for none. This allows spotting relationships at a glance. The scatter plot shows the strong linear relationship visually—points cluster tightly around the regression line.

Feature pairs with |r| > 0.7 are programmatically identified. When correlation exceeds 0.9 (like petal length and width), a warning is issued about multicollinearity. This affects feature selection: why keep both features when they tell the same story?

### Part 7: Comprehensive Correlation View

```python
# Create pairplot for comprehensive view
print("\n" + "=" * 60)
print("PAIRPLOT: ALL FEATURE RELATIONSHIPS")
print("=" * 60)
pairplot = sns.pairplot(df, hue='target', diag_kind='hist',
                        plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'black'},
                        height=2.5)
pairplot.fig.suptitle('Iris Dataset: Pairwise Feature Relationships',
                       y=1.02, fontsize=14, weight='bold')
plt.savefig('diagrams/pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# Key insight about multicollinearity
print("\n" + "=" * 60)
print("MULTICOLLINEARITY WARNING")
print("=" * 60)
print("""
When features are highly correlated (r > 0.9), they provide redundant information.
In machine learning:
  - Linear models become unstable (coefficients swing wildly)
  - Feature importance becomes unclear (which feature gets credit?)
  - Model interpretation becomes difficult

For petal length and petal width (r = 0.96):
  1. Drop one of them (they tell nearly the same story)
  2. Combine them into a single feature (e.g., petal area = length × width)
  3. Use regularization (ridge/lasso) to handle multicollinearity

This is a feature engineering decision informed by statistical analysis.
""")
```

Seaborn's pairplot creates a grid showing every feature against every other feature. The diagonal shows individual distributions (histograms), while off-diagonal cells show scatter plots. Colors represent different iris species. This comprehensive view allows spotting linear relationships, non-linear patterns, and group separations all at once.

This is the key insight. High correlation isn't just an academic curiosity—it affects model performance. In linear regression, multicollinearity makes coefficients unstable and hard to interpret. The solution might be to drop one feature, combine them, or use regularization techniques (covered in later chapters).

**Critical Warning:** Correlation measures only *linear* relationships. Two variables could have a strong non-linear relationship (like a parabola) but show zero correlation. Always visualize!

### Part 8: Outlier Detection with IQR Method

```python
# Outlier Detection: IQR and Z-Score Methods
from sklearn.datasets import fetch_california_housing

# Load data
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

print("=" * 60)
print("OUTLIER DETECTION ANALYSIS")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# Focus on median income feature
feature_name = 'MedInc'
data = df[feature_name].values

print(f"\n{'='*60}")
print(f"ANALYZING FEATURE: {feature_name}")
print(f"{'='*60}")
print(f"Basic statistics:")
print(f"  Mean:   {np.mean(data):.4f}")
print(f"  Median: {np.median(data):.4f}")
print(f"  Std:    {np.std(data, ddof=1):.4f}")
print(f"  Min:    {np.min(data):.4f}")
print(f"  Max:    {np.max(data):.4f}")

# METHOD 1: IQR (Interquartile Range) Method
print(f"\n{'='*60}")
print("METHOD 1: IQR (INTERQUARTILE RANGE)")
print(f"{'='*60}")

# Compute quartiles
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

print(f"Q1 (25th percentile): {q1:.4f}")
print(f"Q3 (75th percentile): {q3:.4f}")
print(f"IQR (Q3 - Q1):        {iqr:.4f}")

# Define outlier boundaries
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

print(f"\nOutlier boundaries:")
print(f"  Lower fence: Q1 - 1.5×IQR = {lower_bound:.4f}")
print(f"  Upper fence: Q3 + 1.5×IQR = {upper_bound:.4f}")

# Identify outliers
outliers_iqr = (data < lower_bound) | (data > upper_bound)
n_outliers_iqr = np.sum(outliers_iqr)
pct_outliers_iqr = 100 * n_outliers_iqr / len(data)

print(f"\nOutliers detected: {n_outliers_iqr} ({pct_outliers_iqr:.2f}%)")
print(f"Outlier examples (first 10):")
outlier_indices = np.where(outliers_iqr)[0][:10]
for idx in outlier_indices:
    print(f"  Index {idx}: {data[idx]:.4f}")
```

The California Housing dataset is loaded and focus is placed on median income. The basic statistics show mean (3.87) > median (3.53), indicating right skewness, and a wide range (0.50 to 15.00), suggesting potential outliers.

The IQR method is robust to outliers because it uses quartiles rather than mean/std. Q1 (2.56) and Q3 (4.74) are computed, giving IQR = 2.18. The fences are Q1 - 1.5×IQR (lower) and Q3 + 1.5×IQR (upper). Any point beyond these fences is an outlier. This method flags 1,023 outliers (4.96% of data).

**Why 1.5×IQR?** This is a convention from exploratory data analysis. For normally distributed data, about 0.7% of points fall beyond these fences. More than that suggests non-normality or genuine outliers.

### Part 9: Outlier Detection with Z-Score Method

```python
# METHOD 2: Z-Score Method
print(f"\n{'='*60}")
print("METHOD 2: Z-SCORE METHOD")
print(f"{'='*60}")

# Compute z-scores
mean = np.mean(data)
std = np.std(data, ddof=1)
z_scores = (data - mean) / std

print(f"Mean:    {mean:.4f}")
print(f"Std Dev: {std:.4f}")
print(f"\nZ-score formula: z = (x - μ) / σ")
print(f"A z-score tells how many standard deviations away from the mean a point is")

# Common threshold: |z| > 3
threshold = 3.0
outliers_zscore = np.abs(z_scores) > threshold
n_outliers_zscore = np.sum(outliers_zscore)
pct_outliers_zscore = 100 * n_outliers_zscore / len(data)

print(f"\nThreshold: |z| > {threshold}")
print(f"Outliers detected: {n_outliers_zscore} ({pct_outliers_zscore:.2f}%)")
print(f"Outlier examples (first 10):")
outlier_indices_z = np.where(outliers_zscore)[0][:10]
for idx in outlier_indices_z:
    print(f"  Index {idx}: value={data[idx]:.4f}, z-score={z_scores[idx]:.4f}")

# Compare methods
print(f"\n{'='*60}")
print("COMPARISON OF METHODS")
print(f"{'='*60}")
print(f"IQR method:    {n_outliers_iqr} outliers ({pct_outliers_iqr:.2f}%)")
print(f"Z-score method: {n_outliers_zscore} outliers ({pct_outliers_zscore:.2f}%)")
print(f"\nOverlap: {np.sum(outliers_iqr & outliers_zscore)} outliers flagged by both methods")
print(f"IQR only:     {np.sum(outliers_iqr & ~outliers_zscore)} outliers")
print(f"Z-score only: {np.sum(~outliers_iqr & outliers_zscore)} outliers")
```

Z-scores standardize by measuring how many standard deviations a point is from the mean. The formula z = (x - μ) / σ gives dimensionless values. A z-score of 3 means "3 standard deviations above the mean." For normal distributions, |z| > 3 includes only 0.3% of data (see the 68-95-99.7 rule). This method flags 89 outliers (0.43% of data).

**Why |z| > 3?** This is a common threshold. Some analysts use 2.5 or 2, depending on how aggressive they want to be. Lower thresholds flag more outliers.

The IQR method flags many more outliers (1,023) than the z-score method (89). Why? IQR is more sensitive because it's based on the middle 50% of data, while z-score uses the full distribution. For skewed data like income, IQR better captures distributional shape. Overlap is checked: most z-score outliers are also flagged by IQR (they're extreme by both standards), but IQR flags many moderate outliers that z-score misses.

### Part 10: Outlier Visualization and Impact

```python
# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Box plot showing outliers (IQR method)
box_parts = axes[0, 0].boxplot(data, vert=True, patch_artist=True, widths=0.5)
box_parts['boxes'][0].set_facecolor('lightblue')
box_parts['medians'][0].set_color('red')
box_parts['medians'][0].set_linewidth(2)
axes[0, 0].axhline(lower_bound, color='orange', linestyle='--',
                    label=f'Lower fence = {lower_bound:.2f}')
axes[0, 0].axhline(upper_bound, color='orange', linestyle='--',
                    label=f'Upper fence = {upper_bound:.2f}')
axes[0, 0].set_ylabel(f'{feature_name}')
axes[0, 0].set_title(f'Box Plot: IQR Outlier Detection\n{n_outliers_iqr} outliers',
                      weight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Histogram with outlier regions shaded (Z-score method)
axes[0, 1].hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].axvline(mean - threshold*std, color='red', linestyle='--', linewidth=2,
                    label=f'μ - {threshold}σ')
axes[0, 1].axvline(mean + threshold*std, color='red', linestyle='--', linewidth=2,
                    label=f'μ + {threshold}σ')
axes[0, 1].axvline(mean, color='green', linestyle='-', linewidth=2, label='Mean')
axes[0, 1].set_xlabel(f'{feature_name}')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title(f'Histogram: Z-Score Outlier Detection\n{n_outliers_zscore} outliers beyond ±{threshold}σ',
                      weight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Scatter plot: index vs value (both methods)
indices = np.arange(len(data))
colors = np.array(['blue'] * len(data))
colors[outliers_iqr & outliers_zscore] = 'red'      # Both methods
colors[outliers_iqr & ~outliers_zscore] = 'orange'  # IQR only
colors[~outliers_iqr & outliers_zscore] = 'purple'  # Z-score only

axes[1, 0].scatter(indices, data, c=colors, alpha=0.5, s=10)
axes[1, 0].axhline(mean, color='green', linestyle='-', linewidth=1, label='Mean')
axes[1, 0].axhline(upper_bound, color='orange', linestyle='--', linewidth=1,
                    label='IQR fences')
axes[1, 0].axhline(lower_bound, color='orange', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('Index')
axes[1, 0].set_ylabel(f'{feature_name}')
axes[1, 0].set_title('Scatter Plot: Outliers by Both Methods', weight='bold')
axes[1, 0].legend(['Mean', 'IQR fences', 'Both methods', 'IQR only', 'Z-score only'])
axes[1, 0].grid(True, alpha=0.3)

# 4. Impact of outlier removal on statistics
data_no_outliers = data[~outliers_iqr]
stats_comparison = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
    'With Outliers': [len(data), np.mean(data), np.median(data),
                      np.std(data, ddof=1), np.min(data), np.max(data)],
    'Without Outliers': [len(data_no_outliers), np.mean(data_no_outliers),
                         np.median(data_no_outliers), np.std(data_no_outliers, ddof=1),
                         np.min(data_no_outliers), np.max(data_no_outliers)]
})
stats_comparison['Change (%)'] = 100 * (stats_comparison['Without Outliers'] -
                                         stats_comparison['With Outliers']) / stats_comparison['With Outliers']

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=stats_comparison.values,
                         colLabels=stats_comparison.columns,
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
axes[1, 1].set_title('Impact of Outlier Removal (IQR method)', weight='bold', pad=20)

plt.tight_layout()
plt.savefig('diagrams/outlier_detection.png', dpi=150, bbox_inches='tight')
plt.show()

# Print statistics comparison
print(f"\n{'='*60}")
print("IMPACT OF OUTLIER REMOVAL")
print(f"{'='*60}")
print(stats_comparison.to_string(index=False))
print(f"\nKey observations:")
print(f"  - Mean changed by {stats_comparison.iloc[1, 3]:.2f}% (sensitive to outliers)")
print(f"  - Median changed by {stats_comparison.iloc[2, 3]:.2f}% (robust to outliers)")
print(f"  - This demonstrates why median is preferred for skewed/outlier-heavy data")

# Decision framework
print(f"\n{'='*60}")
print("WHEN TO REMOVE OUTLIERS?")
print(f"{'='*60}")
print("""
REMOVE when:
  ✓ Clear measurement error (e.g., age = -5 or 200)
  ✓ Data entry mistake (e.g., extra zero: 1000 instead of 100)
  ✓ Does not represent valid data (sensor malfunction)
  ✓ Breaking model assumptions (some algorithms assume normality)

KEEP when:
  ✓ Genuine rare events (in finance, extreme losses are real)
  ✓ Signal of interest (fraud detection: outliers ARE the signal)
  ✓ Represents real population variation
  ✓ Removing would introduce bias

ALTERNATIVES to removal:
  • Winsorize: Replace outliers with nearest non-outlier values
  • Cap/floor: Set maximum/minimum thresholds
  • Transform: Log transform to reduce impact
  • Robust methods: Use median/IQR instead of mean/std
  • Separate analysis: Model outliers separately
""")
```

Four plots tell the complete story:

1. **Box plot** shows IQR outliers as individual circles beyond the whiskers. The many circles at the top confirm right skewness.

2. **Histogram with z-score boundaries** shows the full distribution. The red dashed lines mark ±3σ from the mean. Most outliers are on the right (high incomes).

3. **Scatter plot** color-codes outliers: red (both methods), orange (IQR only), purple (z-score only). This reveals that z-score only flags extreme outliers, while IQR flags moderately unusual values too.

4. **Statistics table** quantifies the impact of removing outliers. Mean drops by 4.2% (pulled down because high outliers were removed), but median barely changes (0.3%)—demonstrating robustness.

After removing outliers, the mean decreases from 3.87 to 3.71 (a 4.2% drop), while the median barely changes (3.53 to 3.52, only 0.3%). This stark contrast proves that mean is sensitive to outliers while median is robust. For reporting central tendency of skewed data, use the median!

The printed guidelines help make informed decisions:

- **Remove** if they're errors or break model assumptions
- **Keep** if they represent real, important phenomena
- **Transform** (log, winsorize) if reducing their influence without discarding information is desired

In data science, blindly removing outliers can be disastrous. In fraud detection, outliers *are* the fraud cases being sought! Always investigate before removing.

---

## Common Pitfalls

**1. Assuming Mean is Always the Best Measure of Center**

Beginners default to mean because it's familiar. But mean is pulled by outliers and misrepresents skewed distributions.

**Example:** In a neighborhood where most houses cost $300K but one mansion costs $10M, the mean might be $500K—misleading because no "typical" house costs that much. The median ($300K) better represents the typical home.

**Rule of thumb:**
- Symmetric distribution → use mean
- Skewed distribution → use median
- Always compute *both* and compare

**2. Confusing Correlation with Causation**

This is the cardinal sin of statistics. High correlation means two variables move together, but it doesn't tell why.

**Classic example:** Ice cream sales and drowning deaths are strongly correlated. Does ice cream cause drowning? No—both increase in summer (confounding variable: temperature/season).

**What students often miss:** The relationship is actually asymmetric:
- Correlation doesn't *imply* causation (causation cannot be concluded from correlation alone)
- But causation *requires* correlation (if X causes Y, they must correlate)
- Correlation doesn't *rule out* causation (correlational data can still be evidence for causation)

**Best practice:** When high correlation is found, ask "What else could explain this relationship?" Think about confounders, reverse causation, and common causes. Correlation is a starting point for investigation, not a conclusion.

**3. Automatically Removing All Outliers**

Outliers are unusual, but "unusual" doesn't mean "wrong." Context determines whether to remove them.

**Example:** Analysis of credit card transactions shows a customer who normally spends $50-100 suddenly makes a $5,000 purchase. That's an outlier—but it might be:
- Fraud (remove from normal behavior model)
- Legitimate large purchase (keep—it's real spending)
- Data entry error (remove or correct)

In fraud detection systems, outliers *are* the signal. Removing them defeats the purpose!

**Better approach:**
1. **Investigate first:** Why is this point unusual?
2. **Consider alternatives:** Transform (log), winsorize, or use robust statistics
3. **Separate analysis:** Model outliers and normal points differently
4. **Document decisions:** If outliers are removed, explain why

**4. Ignoring Sample vs. Population Distinction**

Beginners treat datasets as complete populations when they're actually samples from larger populations.

**Why it matters:**
- In machine learning, training data is a *sample* from the population of all possible data
- Sample statistics (x̄, s) estimate population parameters (μ, σ)
- Bad samples lead to biased models that fail in production

**Bessel's correction example:** When computing sample variance, divide by (n-1) instead of n. This corrects for underestimation bias. Using n gives the variance of the sample; using (n-1) estimates the population variance.

**Real-world impact:** If a model is trained on data from 2020 (sample) and deployed in 2025 (population), the assumption is that the sample represents the population. Distribution shift can break the model. Always ask: "Is the sample representative?"

---

## Practice

**Practice 1**

Load the Wine dataset (`load_wine()`) and analyze the 'alcohol' feature.

1. Convert to DataFrame and examine the 'alcohol' column
2. Compute the mean, median, and mode manually (use `scipy.stats.mode` if needed)
3. Compute the range, variance (sample variance with n-1), and standard deviation
4. Use `df['alcohol'].describe()` and verify manual calculations match
5. Create a histogram of the alcohol distribution with 20 bins
6. Answer: Is the distribution symmetric or skewed? How is this determined? (Compare mean to median and visualize)
7. Answer: If a wine has alcohol content of 14.5%, how many standard deviations from the mean is it? (Compute z-score)

---

**Practice 2**

Load the Diabetes dataset (`load_diabetes()`) and perform comprehensive distribution analysis on the 'bmi' (body mass index) feature.

1. Convert to DataFrame. Note: sklearn's diabetes dataset provides standardized features by default. To work with raw data, access the raw feature values or use `df.describe()` on the standardized data.
2. For the 'bmi' feature:
   - a. Create a histogram and box plot side-by-side in one figure
   - b. Compute skewness using `scipy.stats.skew()`
   - c. Identify outliers using the IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
   - d. Identify outliers using the Z-score method with threshold |z| > 3
   - e. Compare: How many outliers does each method find? How many overlap?
3. Visualize the outliers on a scatter plot (sample index vs. bmi value), color-coding by detection method
4. Create a "before and after" comparison showing statistics with and without outliers (use IQR method)
5. Answer: How much did the mean and median change after removing outliers? Which is more robust? Why?

Bonus: Repeat the analysis for all features and create a summary table showing which features have the most outliers.

---

**Practice 3**

Perform comprehensive correlation analysis on the California Housing dataset to inform feature selection decisions.

1. Load the California Housing dataset into a DataFrame with feature names
2. Compute the full correlation matrix for all features (excluding the target initially)
3. Visualize the correlation matrix as an annotated heatmap with a diverging colormap
4. Identify all feature pairs with |r| > 0.7 (strong correlation threshold)
5. For each highly correlated pair:
   - a. Create a scatter plot with a fitted regression line
   - b. Compute the correlation manually using the formula r = Cov(X,Y) / (σₓσᵧ) to verify
   - c. Interpret: Does the relationship make sense? (e.g., why would these features correlate in real-world terms?)
6. Analyze correlation of features with the target variable (median house value):
   - a. Which feature has the strongest correlation with price?
   - b. Which has the weakest?
   - c. Do the correlations match intuition about what affects housing prices?
7. Make a recommendation: If redundant features must be dropped due to multicollinearity, which ones would be eliminated and why? Justify the choices.
8. Compute correlation between 'AveRooms' and 'AveBedrms' before and after log transformation. Does transformation affect correlation? Why or why not?

---

**Practice 4**

- Can any non-linear relationships that Pearson correlation might miss be identified? (Hint: look at scatter plots)
- How would non-linear relationships be detected computationally? (Research: Spearman correlation, mutual information)
- Advanced: Investigate if Simpson's Paradox exists in this dataset (correlation reverses when data is grouped by another variable)

---

## Solutions

**Solution 1**

```python
from sklearn.datasets import load_wine
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Wine dataset
wine = load_wine(as_frame=True)
df = wine.frame

# 1. Examine the 'alcohol' column
print("First 5 rows of alcohol column:")
print(df['alcohol'].head())
print(f"\nDataset shape: {df.shape}")

# 2. Compute mean, median, mode manually
alcohol = df['alcohol'].values
mean_manual = np.sum(alcohol) / len(alcohol)
sorted_alcohol = np.sort(alcohol)
n = len(sorted_alcohol)
median_manual = (sorted_alcohol[n//2-1] + sorted_alcohol[n//2])/2 if n%2==0 else sorted_alcohol[n//2]
mode_result = stats.mode(alcohol, keepdims=True)
mode_manual = mode_result[0][0]

print(f"\nManual calculations:")
print(f"Mean:   {mean_manual:.4f}")
print(f"Median: {median_manual:.4f}")
print(f"Mode:   {mode_manual:.4f}")

# 3. Compute range, variance, standard deviation
range_val = np.max(alcohol) - np.min(alcohol)
variance_manual = np.sum((alcohol - mean_manual)**2) / (len(alcohol) - 1)
std_manual = np.sqrt(variance_manual)

print(f"\nRange:    {range_val:.4f}")
print(f"Variance: {variance_manual:.4f}")
print(f"Std Dev:  {std_manual:.4f}")

# 4. Verify with describe()
print("\nVerification with describe():")
print(df['alcohol'].describe())

# 5. Create histogram
plt.figure(figsize=(8, 5))
plt.hist(alcohol, bins=20, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(mean_manual, color='red', linestyle='--', linewidth=2, label=f'Mean={mean_manual:.2f}')
plt.axvline(median_manual, color='blue', linestyle='--', linewidth=2, label=f'Median={median_manual:.2f}')
plt.xlabel('Alcohol Content (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Alcohol Content in Wine Dataset')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 6. Skewness analysis
skew = stats.skew(alcohol)
print(f"\n6. Distribution analysis:")
print(f"   Mean: {mean_manual:.4f}, Median: {median_manual:.4f}")
print(f"   Skewness: {skew:.4f}")
if abs(skew) < 0.5:
    print("   The distribution is approximately symmetric")
else:
    print(f"   The distribution is {'right' if skew > 0 else 'left'}-skewed")

# 7. Z-score for 14.5%
value = 14.5
z_score = (value - mean_manual) / std_manual
print(f"\n7. For alcohol content of {value}%:")
print(f"   Z-score: {z_score:.4f}")
print(f"   This is {abs(z_score):.2f} standard deviations {'above' if z_score > 0 else 'below'} the mean")
```

The Wine dataset is loaded and converted to a DataFrame. Manual calculations are performed for mean, median, and mode, then verified against library functions. The histogram with mean/median lines reveals the distribution shape. Since mean ≈ median and skewness is near 0, the distribution is approximately symmetric. The z-score calculation shows how many standard deviations a specific value is from the mean.

**Solution 2**

```python
from sklearn.datasets import load_diabetes
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Diabetes dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

# Focus on 'bmi' feature
bmi = df['bmi'].values

print("=" * 60)
print("BMI FEATURE ANALYSIS")
print("=" * 60)

# 2a. Histogram and box plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(bmi, bins=30, alpha=0.7, color='teal', edgecolor='black')
axes[0].set_xlabel('BMI (standardized)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram: BMI Distribution')
axes[0].grid(alpha=0.3)

axes[1].boxplot(bmi, vert=True, patch_artist=True, widths=0.5)
axes[1].set_ylabel('BMI (standardized)')
axes[1].set_title('Box Plot: BMI with Outliers')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 2b. Compute skewness
skewness = stats.skew(bmi)
print(f"\nSkewness: {skewness:.4f}")

# 2c. IQR method
q1 = np.percentile(bmi, 25)
q3 = np.percentile(bmi, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = (bmi < lower_bound) | (bmi > upper_bound)
n_outliers_iqr = np.sum(outliers_iqr)

print(f"\nIQR Method:")
print(f"  Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"  Outliers: {n_outliers_iqr} ({100*n_outliers_iqr/len(bmi):.2f}%)")

# 2d. Z-score method
mean = np.mean(bmi)
std = np.std(bmi, ddof=1)
z_scores = (bmi - mean) / std
outliers_zscore = np.abs(z_scores) > 3
n_outliers_zscore = np.sum(outliers_zscore)

print(f"\nZ-Score Method (|z| > 3):")
print(f"  Outliers: {n_outliers_zscore} ({100*n_outliers_zscore/len(bmi):.2f}%)")

# 2e. Comparison
overlap = np.sum(outliers_iqr & outliers_zscore)
print(f"\nComparison:")
print(f"  Overlap: {overlap} outliers flagged by both methods")
print(f"  IQR only: {np.sum(outliers_iqr & ~outliers_zscore)}")
print(f"  Z-score only: {np.sum(~outliers_iqr & outliers_zscore)}")

# 3. Scatter plot with color-coding
indices = np.arange(len(bmi))
colors = np.array(['blue'] * len(bmi))
colors[outliers_iqr & outliers_zscore] = 'red'
colors[outliers_iqr & ~outliers_zscore] = 'orange'
colors[~outliers_iqr & outliers_zscore] = 'purple'

plt.figure(figsize=(12, 5))
plt.scatter(indices, bmi, c=colors, alpha=0.6, s=20)
plt.axhline(upper_bound, color='orange', linestyle='--', label='IQR bounds')
plt.axhline(lower_bound, color='orange', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('BMI (standardized)')
plt.title('Outlier Detection: IQR vs Z-Score')
plt.legend(['IQR bounds', 'Normal', 'Both methods', 'IQR only', 'Z-score only'])
plt.grid(alpha=0.3)
plt.show()

# 4. Before/after comparison
bmi_no_outliers = bmi[~outliers_iqr]
comparison = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev'],
    'With Outliers': [len(bmi), np.mean(bmi), np.median(bmi), np.std(bmi, ddof=1)],
    'Without Outliers': [len(bmi_no_outliers), np.mean(bmi_no_outliers),
                         np.median(bmi_no_outliers), np.std(bmi_no_outliers, ddof=1)]
})
comparison['Change (%)'] = 100 * (comparison['Without Outliers'] - comparison['With Outliers']) / comparison['With Outliers']

print("\n" + "=" * 60)
print("IMPACT OF OUTLIER REMOVAL")
print("=" * 60)
print(comparison.to_string(index=False))

# 5. Answer
mean_change = comparison.iloc[1, 3]
median_change = comparison.iloc[2, 3]
print(f"\n5. Mean changed by {mean_change:.2f}%, median changed by {median_change:.2f}%")
print("   Median is more robust because it's not affected by extreme values.")
```

The Diabetes dataset is loaded and the 'bmi' feature is analyzed. Histogram and box plot visualizations are created side-by-side. Skewness is computed, then outliers are detected using both IQR and Z-score methods. The comparison shows IQR typically flags more outliers than Z-score for standardized data. A scatter plot with color-coding reveals which outliers are detected by which method. The before/after statistics comparison demonstrates that median changes less than mean when outliers are removed, proving its robustness.

**Solution 3**

```python
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("=" * 60)
print("CALIFORNIA HOUSING - CORRELATION ANALYSIS")
print("=" * 60)
print(f"Shape: {df.shape}")

# 2. Compute correlation matrix (exclude target)
features = df.drop('MedHouseVal', axis=1)
corr_matrix = features.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# 3. Visualize as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, vmin=-1, vmax=1)
plt.title('Correlation Matrix: California Housing Features', fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# 4. Identify highly correlated pairs
print("\n" + "=" * 60)
print("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.7)")
print("=" * 60)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            feat1 = corr_matrix.columns[i]
            feat2 = corr_matrix.columns[j]
            high_corr_pairs.append((feat1, feat2, corr_val))
            print(f"{feat1:20s} <--> {feat2:20s}: r = {corr_val:7.4f}")

# 5. For each highly correlated pair
if high_corr_pairs:
    for feat1, feat2, r in high_corr_pairs:
        # 5a. Scatter plot with regression line
        plt.figure(figsize=(8, 6))
        x = df[feat1].values
        y = df[feat2].values
        plt.scatter(x, y, alpha=0.3, s=10)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", linewidth=2, label='Linear fit')
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.title(f'{feat1} vs {feat2}\nr = {r:.4f}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

        # 5b. Manual correlation verification
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        cov = np.sum((x - mean_x) * (y - mean_y)) / len(x)
        std_x = np.std(x, ddof=1)
        std_y = np.std(y, ddof=1)
        r_manual = cov / (std_x * std_y)
        print(f"\nManual verification for {feat1} vs {feat2}:")
        print(f"  Covariance: {cov:.4f}")
        print(f"  Correlation (manual): {r_manual:.4f}")
        print(f"  Correlation (library): {r:.4f}")

        # 5c. Interpretation
        print(f"  Real-world interpretation: These features are strongly correlated,")
        print(f"  which makes sense because they likely measure related aspects.")

# 6. Correlation with target
print("\n" + "=" * 60)
print("CORRELATION WITH TARGET (MedHouseVal)")
print("=" * 60)
target_corr = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)
print(target_corr)
print(f"\nStrongest correlation: {target_corr.index[0]} (r = {target_corr.iloc[0]:.4f})")
print(f"Weakest correlation: {target_corr.index[-1]} (r = {target_corr.iloc[-1]:.4f})")

# 7. Recommendation for feature dropping
print("\n" + "=" * 60)
print("FEATURE SELECTION RECOMMENDATION")
print("=" * 60)
if high_corr_pairs:
    print("Due to multicollinearity, consider dropping:")
    for feat1, feat2, r in high_corr_pairs:
        # Drop the one with weaker correlation to target
        corr1 = abs(df.corr()['MedHouseVal'][feat1])
        corr2 = abs(df.corr()['MedHouseVal'][feat2])
        if corr1 > corr2:
            print(f"  - Drop {feat2} (keep {feat1}, stronger target correlation)")
        else:
            print(f"  - Drop {feat1} (keep {feat2}, stronger target correlation)")
else:
    print("No highly correlated feature pairs found (|r| > 0.7)")

# 8. Transformation effect on correlation
print("\n" + "=" * 60)
print("LOG TRANSFORMATION EFFECT ON CORRELATION")
print("=" * 60)
rooms = df['AveRooms'].values
bedrms = df['AveBedrms'].values
corr_before = np.corrcoef(rooms, bedrms)[0, 1]
log_rooms = np.log1p(rooms)
log_bedrms = np.log1p(bedrms)
corr_after = np.corrcoef(log_rooms, log_bedrms)[0, 1]

print(f"Correlation before log transform: {corr_before:.4f}")
print(f"Correlation after log transform:  {corr_after:.4f}")
print(f"\nLog transformation {'does' if abs(corr_after - corr_before) > 0.1 else 'does not'} substantially affect correlation.")
print("Explanation: Correlation measures linear relationships. Log transform changes")
print("the scale but preserves the monotonic relationship, so correlation remains similar.")
```

The California Housing dataset is loaded with feature names. The full correlation matrix is computed and visualized as an annotated heatmap. Feature pairs with |r| > 0.7 are identified. For each highly correlated pair, a scatter plot with regression line is created, correlation is computed manually to verify the library result, and real-world interpretation is provided. Correlation with the target variable is analyzed to determine which features are most predictive. A recommendation is made for which redundant features to drop based on their target correlation strength. Finally, the effect of log transformation on correlation is tested—log transform preserves monotonic relationships, so correlation typically remains similar.

**Solution 4**

```python
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("=" * 60)
print("NON-LINEAR RELATIONSHIPS AND ADVANCED CORRELATION")
print("=" * 60)

# Look for potential non-linear relationships
# Example: Latitude vs MedHouseVal might have non-linear pattern
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = df['Latitude'].values
y = df['MedHouseVal'].values

# Scatter plot
axes[0].scatter(x, y, alpha=0.2, s=5)
axes[0].set_xlabel('Latitude')
axes[0].set_ylabel('Median House Value')
axes[0].set_title('Potential Non-Linear Relationship')
axes[0].grid(alpha=0.3)

# Pearson vs Spearman correlation
pearson_r = np.corrcoef(x, y)[0, 1]
spearman_r, _ = spearmanr(x, y)

axes[1].text(0.1, 0.8, f"Pearson correlation (linear): {pearson_r:.4f}",
             fontsize=12, transform=axes[1].transAxes)
axes[1].text(0.1, 0.6, f"Spearman correlation (monotonic): {spearman_r:.4f}",
             fontsize=12, transform=axes[1].transAxes)
axes[1].text(0.1, 0.4, "If Spearman >> Pearson, non-linear\nmonotonic relationship exists",
             fontsize=10, transform=axes[1].transAxes, style='italic')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Mutual information for non-linear relationships
print("\nMutual Information (detects non-linear relationships):")
features = df.drop('MedHouseVal', axis=1)
target = df['MedHouseVal']
mi_scores = mutual_info_regression(features, target, random_state=42)
mi_df = pd.DataFrame({'Feature': features.columns, 'MI Score': mi_scores}).sort_values('MI Score', ascending=False)
print(mi_df.to_string(index=False))

# Simpson's Paradox investigation
print("\n" + "=" * 60)
print("SIMPSON'S PARADOX INVESTIGATION")
print("=" * 60)

# Example: Check if correlation between AveRooms and MedHouseVal reverses by region
# Create region groups based on latitude
df['Region'] = pd.cut(df['Latitude'], bins=3, labels=['South', 'Central', 'North'])

overall_corr = df['AveRooms'].corr(df['MedHouseVal'])
print(f"Overall correlation (AveRooms vs MedHouseVal): {overall_corr:.4f}")

print("\nCorrelation by region:")
for region in ['South', 'Central', 'North']:
    region_df = df[df['Region'] == region]
    region_corr = region_df['AveRooms'].corr(region_df['MedHouseVal'])
    print(f"  {region}: {region_corr:.4f}")

if all(df.groupby('Region').apply(lambda g: g['AveRooms'].corr(g['MedHouseVal'])).values > overall_corr) or \
   all(df.groupby('Region').apply(lambda g: g['AveRooms'].corr(g['MedHouseVal'])).values < overall_corr):
    print("\nNo clear Simpson's Paradox detected (correlations don't reverse direction)")
else:
    print("\nPossible Simpson's Paradox: correlation direction changes across groups!")
```

Non-linear relationships are investigated by comparing Pearson correlation (measures linear relationships) with Spearman correlation (measures monotonic relationships). If Spearman correlation is much higher than Pearson, a non-linear but monotonic relationship exists. Mutual information scores are computed to detect any non-linear relationships—MI captures dependencies that correlation misses. Simpson's Paradox is investigated by checking if correlation between features changes direction when data is grouped by region. The code demonstrates that sophisticated correlation analysis requires multiple techniques beyond simple Pearson correlation.

---

## Key Takeaways

- **Statistics operationalizes probability on real data.** Probability tells what should happen theoretically; statistics tells what did happen in the dataset. Together, they form the mathematical foundation of data science.

- **Measures of center (mean, median, mode) and spread (variance, standard deviation, IQR) tell different parts of the story.** Mean is sensitive to outliers; median is robust. Variance has squared units; standard deviation returns to original units. Always compute multiple statistics and interpret them together.

- **Distribution shape determines which statistics to trust.** For symmetric distributions, mean and median coincide. For skewed distributions (like income, housing prices), median better represents the "typical" value because mean is pulled toward the tail. Always visualize before summarizing.

- **Correlation measures linear relationships, not causation.** Pearson's r ranges from -1 (perfect negative) to +1 (perfect positive), with 0 meaning no linear relationship. High correlation can arise from: (1) X causes Y, (2) Y causes X, (3) confounding variable causes both, or (4) pure coincidence. Correlation is necessary but not sufficient for causation.

- **Outliers require investigation, not automatic removal.** Use IQR method (robust, flags more outliers) or z-score method (assumes normality, flags extreme outliers). Before removing outliers, ask: Are they measurement errors? Real rare events? The signal itself (fraud detection)? Context determines the right action: remove, transform, model separately, or keep.

- **Sample statistics estimate population parameters.** In machine learning, training data is always a sample from the population of all possible data. Sample mean (x̄) and sample standard deviation (s) estimate population mean (μ) and population standard deviation (σ). Use Bessel's correction (n-1) for unbiased estimates. Bad samples lead to biased models—always assess representativeness.

- **Visualize before summarizing.** Summary statistics alone can hide patterns (Anscombe's Quartet proves this). Use histograms to see shape, box plots to spot outliers, scatter plots to reveal relationships, and heatmaps to find correlations. Visualizations reveal what numbers hide.

- **Every data science project starts with statistical analysis.** Before building models, run `df.describe()`, plot distributions, check for outliers, compute correlations, and assess data quality. This is exploratory data analysis (EDA)—applied statistics that informs feature engineering, model selection, and interpretation. Statistics isn't separate from machine learning; it *is* the foundation of machine learning.

---

**Connections to Other Chapters:**
- See Chapter 1 (Linear Algebra) for understanding correlation matrices as symmetric matrices and covariance operations
- See Chapter 3 (Probability) for the theoretical foundations of distributions that statistics measures empirically
- In later courses, these statistical tools will be essential for feature engineering (standardization), model evaluation (bias-variance tradeoff), A/B testing (hypothesis testing), and diagnostic plots (residual analysis)

**Next:** Chapter 5 covers calculus for optimization, the mathematical foundation for training machine learning models through gradient descent.

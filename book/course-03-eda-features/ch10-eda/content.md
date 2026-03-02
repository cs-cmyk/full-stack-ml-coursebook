> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# Chapter 10: Exploratory Data Analysis

## Why This Matters

Imagine building a machine learning model on a dataset without ever looking at it—trusting the numbers blindly, feeding them into an algorithm, and hoping for the best. You'd likely end up with a model that fails spectacularly, makes nonsensical predictions, or worse, appears to work but produces dangerously misleading results. Exploratory Data Analysis (EDA) is the detective work of data science: the systematic investigation that helps you understand your data's structure, quality, patterns, and quirks before you invest time building models. Professional data scientists spend 60–80% of their time on data understanding and preparation, because models are only as good as the data they learn from. Skip EDA, and you're building on quicksand.

## Intuition

### The Home Inspection Analogy

Think of buying a house. The photos look beautiful online, the price seems reasonable, and the seller assures you everything is in perfect condition. Would you sign the papers without a thorough inspection?

**Without inspection (no EDA):**
- The foundation has cracks you didn't see
- The plumbing leaks behind the walls
- The electrical system violates safety codes
- The roof will need replacement next year
- You discover all this after you've committed your money

**With inspection (doing EDA):**
- Walk through every room (check every feature)
- Test the systems (look for data quality issues)
- Examine the foundation (understand the structure)
- Review the history (understand data provenance)
- Make an informed decision with eyes open

In data science, the dataset is the house being "bought" for a project. EDA is the thorough inspection that reveals what is really being worked with. Without it, an elaborate model might be built on a dataset with critical flaws—missing values that bias results, outliers that skew patterns, or data entry errors that mislead algorithms. With EDA, the data's limitations and strengths are understood, allowing appropriate models to be built that produce reliable results.

### The Detective Framework

EDA transforms the analyst into a data detective. Arriving at a "crime scene" (the dataset), questions are immediately asked:
- What happened here? (What does this data represent?)
- What evidence is available? (What features exist?)
- What's missing? (Are there gaps in the data?)
- Do these clues fit together? (Are the values consistent and logical?)
- What patterns emerge? (What relationships exist?)
- What story does the evidence tell? (What insights can guide modeling?)

Great detectives don't jump to conclusions—they examine evidence methodically, document findings carefully, and let the data reveal its secrets. That's exactly what happens in EDA.

## Formal Definition

**Exploratory Data Analysis (EDA)** is the systematic process of investigating a dataset to summarize its main characteristics, discover patterns and relationships, detect anomalies and outliers, test assumptions, and assess data quality through summary statistics and data visualization techniques. EDA is typically performed before formal modeling and serves to inform preprocessing decisions, feature engineering strategies, and model selection.

For a dataset with feature matrix **X** ∈ ℝ^(n×p) (where n = number of samples, p = number of features) and target vector **y** ∈ ℝ^n, EDA involves:

1. **Univariate Analysis**: Examining the distribution of individual features x_j (j = 1, ..., p) and target y
2. **Bivariate Analysis**: Investigating relationships between each feature x_j and target y
3. **Multivariate Analysis**: Exploring interactions among multiple features and their collective relationship with y
4. **Data Quality Assessment**: Identifying missing values, outliers, duplicates, and inconsistencies

The goal is to build comprehensive understanding before modeling, where understanding informs decisions about data preprocessing, feature transformation, and algorithm selection.

> **Key Concept:** EDA is not optional—it is the essential foundation upon which all reliable data science is built. Good modeling decisions cannot be made without first understanding the data.

## Visualization

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   1. FIRST LOOK │────▶│ 2. INDIVIDUAL       │────▶│ 3. TARGET           │
│                 │     │    FEATURES         │     │    VARIABLE         │
│ • Shape         │     │                     │     │                     │
│ • Data types    │     │ • Distributions     │     │ • Class balance     │
│ • Memory usage  │     │ • Missing values    │     │   (classification)  │
│ • First rows    │     │ • Unique counts     │     │ • Distribution      │
│                 │     │ • Summary stats     │     │   (regression)      │
└─────────────────┘     └─────────────────────┘     └─────────────────────┘
         │                                                      │
         │                                                      │
         │              ┌─────────────────────┐                │
         └──────────────▶  ITERATE & DOCUMENT ◀────────────────┘
                        └─────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────┐            ┌─────────────────────┐
        │ 4. RELATIONSHIPS    │            │ 5. DATA QUALITY     │
        │                     │            │                     │
        │ • Feature-target    │            │ • Outliers          │
        │   correlations      │            │ • Duplicates        │
        │ • Feature-feature   │            │ • Inconsistencies   │
        │   interactions      │            │ • Logical errors    │
        │ • Patterns          │            │                     │
        └─────────────────────┘            └─────────────────────┘
```

Each step reveals information that guides the next steps, and discoveries often loop back to re-examine earlier aspects with new questions. The following sections explore each step with complete, runnable examples.

## Examples

### Part 1: Loading and Initial Structure

```python
# Complete EDA Workflow - Classification Problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# Set random seed for reproducibility
np.random.seed(42)

# Configure visualization defaults
plt.style.use('default')
sns.set_palette("husl")

# ============================================================
# STEP 1: FIRST LOOK - Load and Understand Structure
# ============================================================

# Load dataset and convert to DataFrame
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("=" * 60)
print("STEP 1: FIRST LOOK")
print("=" * 60)

# Basic structure
print(f"\nDataset shape: {df.shape}")
print(f"  - {df.shape[0]} samples (patients)")
print(f"  - {df.shape[1]-1} features + 1 target")

# Data types and memory usage
print(f"\nData types:\n{df.dtypes.value_counts()}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# First few rows
print("\nFirst 3 rows:")
print(df.head(3))

# Basic info
print("\nDataset info:")
df.info()
# Output: All features are float64, target is int64, no missing values
```

**Part 1 Walkthrough:**

This initial exploration immediately reveals this is a **manageable dataset**—569 samples with 30 features. All features are numeric (float64), making analysis straightforward without categorical encoding. The memory footprint is small (around 135 KB), so working in-memory is feasible without performance concerns. The first rows give a sense of the scale: features have different ranges (radius in tens, area in hundreds), suggesting scaling will be needed for some algorithms.

### Part 2: Individual Feature Analysis

```python
# ============================================================
# STEP 2: INDIVIDUAL FEATURES - Explore Distributions
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: INDIVIDUAL FEATURES")
print("=" * 60)

# Summary statistics for numeric features
print("\nSummary statistics (first 5 features):")
print(df.iloc[:, :5].describe())
# Output: Shows mean, std, min, quartiles, max for each feature

# Check for missing values
missing_counts = df.isnull().sum()
print(f"\nMissing values: {missing_counts.sum()} total")
print(f"  - No missing values detected in any column")

# Visualize distributions of selected features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Distribution of Key Features', fontsize=16, fontweight='bold')

feature_subset = ['mean radius', 'mean texture', 'mean area',
                  'mean smoothness', 'mean compactness', 'mean concavity']

for idx, feature in enumerate(feature_subset):
    row = idx // 3
    col = idx % 3
    axes[row, col].hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].set_title(f'{feature.title()}')

plt.tight_layout()
plt.savefig('eda_feature_distributions.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_feature_distributions.png")
# Output: Shows histograms revealing distribution shapes (some skewed, some normal)
```

**Part 2 Walkthrough:**

The `.describe()` output reveals that features have vastly different scales—`mean area` ranges from 143 to 2501, while `mean smoothness` ranges from 0.05 to 0.16. This immediately indicates that **feature scaling will be essential** for algorithms sensitive to scale (logistic regression, SVM, k-NN). The histograms show several features with right-skewed distributions, which is common in medical measurements. Some features might benefit from log transformation, though it's not critical for tree-based models.

### Part 3: Target Variable Analysis

```python
# ============================================================
# STEP 3: TARGET VARIABLE - Examine Class Balance
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: TARGET VARIABLE")
print("=" * 60)

# Target distribution
target_counts = df['target'].value_counts()
print("\nTarget variable distribution:")
print(f"  - Benign (1): {target_counts[1]} samples ({target_counts[1]/len(df)*100:.1f}%)")
print(f"  - Malignant (0): {target_counts[0]} samples ({target_counts[0]/len(df)*100:.1f}%)")
# Output: 357 benign (62.7%), 212 malignant (37.3%) - reasonably balanced

# Visualize target balance
fig, ax = plt.subplots(figsize=(8, 5))
target_counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
ax.set_xticklabels(['Malignant', 'Benign'], rotation=0)
ax.set_ylabel('Count')
ax.set_xlabel('Diagnosis')
ax.set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
ax.axhline(y=len(df)/2, color='gray', linestyle='--', label='Perfect balance')
ax.legend()

for i, count in enumerate(target_counts.values):
    ax.text(i, count + 10, str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_target_distribution.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_target_distribution.png")
```

**Part 3 Walkthrough:**

The class distribution (63% benign, 37% malignant) is reasonably balanced. This is good news—special handling for class imbalance like SMOTE or class weights is not needed. Standard metrics (accuracy, precision, recall, F1-score) will be meaningful. If the imbalance were more extreme (say 90/10), more care would be needed with evaluation metrics and potentially using resampling techniques.

### Part 4: Correlation Analysis

```python
# ============================================================
# STEP 4: RELATIONSHIPS - Feature-Target Correlations
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: RELATIONSHIPS")
print("=" * 60)

# Calculate correlations with target
feature_cols = [col for col in df.columns if col != 'target']
correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)

print("\nTop 10 features most correlated with target:")
print(correlations.head(10))
# Output: Shows which features have strongest positive correlation with benign diagnosis

print("\nBottom 5 features (most negatively correlated):")
print(correlations.tail(5))
# Output: Negative correlations indicate features higher in malignant tumors

# Visualize top correlations
fig, ax = plt.subplots(figsize=(10, 8))
top_features = pd.concat([correlations.head(10), correlations.tail(5)])
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features.values]
top_features.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Correlation with Target (Benign=1)')
ax.set_title('Top Features by Correlation with Diagnosis', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.savefig('eda_feature_correlations.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_feature_correlations.png")
```

**Part 4 Walkthrough:**

The correlation analysis reveals several features showing **strong correlations with the target** (above 0.7 in absolute value). Features like `worst concave points` and `worst perimeter` are highly predictive. This suggests that even simple models like logistic regression should perform well. However, **high multicollinearity** is also discovered—for example, `mean radius`, `mean perimeter`, and `mean area` are all highly correlated with each other (correlation > 0.9), which makes sense geometrically. This redundancy means:
- Some features could potentially be dropped without losing information
- Regularization (Ridge/Lasso) would help by penalizing redundant features
- Principal Component Analysis (PCA) could reduce dimensionality while preserving variance

### Part 5: Multicollinearity and Scatter Analysis

```python
# Create correlation heatmap for feature-feature relationships
print("\nGenerating correlation heatmap (may take a moment)...")
# Select subset of features for readability
mean_features = [col for col in df.columns if 'mean' in col or col == 'target']
corr_matrix = df[mean_features].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title('Correlation Heatmap - Mean Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: eda_correlation_heatmap.png")
# Output: Reveals multicollinearity (some features highly correlated with each other)

# Scatter plots of top 2 features vs target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_2_features = correlations.head(2).index

for idx, feature in enumerate(top_2_features):
    # Separate by target class
    benign = df[df['target'] == 1][feature]
    malignant = df[df['target'] == 0][feature]

    axes[idx].scatter(malignant.index, malignant.values, alpha=0.6,
                      label='Malignant', color='#e74c3c', s=30)
    axes[idx].scatter(benign.index, benign.values, alpha=0.6,
                      label='Benign', color='#2ecc71', s=30)
    axes[idx].set_xlabel('Sample Index')
    axes[idx].set_ylabel(feature)
    axes[idx].set_title(f'{feature.title()} by Diagnosis')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_top_features_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: eda_top_features_scatter.png")
```

**Part 5 Walkthrough:**

The scatter plots show clear visual separation between malignant and benign tumors for top features—an encouraging sign for classification performance. The heatmap reveals extensive multicollinearity that will need to be addressed through feature selection, regularization, or dimensionality reduction.

### Part 6: Data Quality Assessment

```python
# ============================================================
# STEP 5: DATA QUALITY - Check for Outliers & Issues
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: DATA QUALITY")
print("=" * 60)

# Check for duplicates
n_duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {n_duplicates}")
# Output: 0 duplicates in this clean dataset

# Detect outliers using IQR method for key features
outlier_summary = []
for feature in feature_subset:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(df)) * 100

    outlier_summary.append({
        'feature': feature,
        'n_outliers': n_outliers,
        'pct_outliers': pct_outliers,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })

outlier_df = pd.DataFrame(outlier_summary)
print("\nOutlier detection (IQR method):")
print(outlier_df.to_string(index=False))
# Output: Shows number and percentage of outliers per feature

# Visualize outliers with box plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Box Plots - Outlier Detection', fontsize=16, fontweight='bold')

for idx, feature in enumerate(feature_subset):
    row = idx // 3
    col = idx % 3
    axes[row, col].boxplot([df[df['target']==0][feature],
                            df[df['target']==1][feature]],
                           labels=['Malignant', 'Benign'])
    axes[row, col].set_ylabel(feature)
    axes[row, col].set_title(f'{feature.title()}')
    axes[row, col].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_outlier_boxplots.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_outlier_boxplots.png")
```

**Part 6 Walkthrough:**

This dataset is remarkably clean—no missing values, no duplicates, no obvious errors. In real-world projects, this step often reveals significant issues that require attention. The box plots show outliers, but in medical data, these often represent real extreme cases rather than errors. Domain expertise (e.g., consulting with an oncologist) would be needed to decide whether to keep or remove them. The fact that outliers appear in both classes suggests they're legitimate measurements of unusual tumors.

### Part 7: Summary and Modeling Implications

```python
# ============================================================
# SUMMARY: KEY FINDINGS & MODELING IMPLICATIONS
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY: KEY FINDINGS")
print("=" * 60)

print("""
1. DATA STRUCTURE:
   - 569 samples, 30 features, binary target
   - All numeric features (float64), no missing values
   - Clean dataset requiring minimal preprocessing

2. TARGET VARIABLE:
   - Reasonably balanced: 62.7% benign, 37.3% malignant
   - No class imbalance correction needed
   - Standard classification metrics (accuracy, F1) appropriate

3. FEATURE DISTRIBUTIONS:
   - Most features show some right skew
   - Some features have clear separation between classes
   - Several features show outliers (but may be legitimate extreme values)

4. RELATIONSHIPS:
   - Strong correlations exist between features and target
   - Top predictive features: worst concave points, worst perimeter, worst radius
   - High multicollinearity among related features (e.g., radius, area, perimeter)
     → Suggests regularization (Ridge/Lasso) or feature selection may help

5. DATA QUALITY:
   - No missing values or duplicates
   - Outliers detected but appear to be real extreme measurements
   - No obvious data entry errors or inconsistencies

6. MODELING RECOMMENDATIONS:
   - Linear models likely effective (clear linear relationships)
   - Consider dimensionality reduction (PCA) due to multicollinearity
   - Tree-based models (Random Forest, XGBoost) good alternatives
   - Feature scaling important for distance-based algorithms
   - Cross-validation essential due to moderate sample size
""")

print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)
```

**Part 7 Walkthrough:**

Armed with these insights, informed decisions can be made:
- **Algorithm choice**: Both linear models (with regularization) and tree-based models should work well
- **Preprocessing**: Scale features, consider PCA, keep outliers (domain suggests they're real)
- **Evaluation**: Standard classification metrics are appropriate given class balance
- **Feature engineering**: Probably not needed—features are already well-engineered medical measurements
- **Expected performance**: Strong feature-target correlations suggest high accuracy should be achievable (likely >90%)

This is the power of EDA—no model has been trained yet, but there is already a clear roadmap.

## Examples: Regression Data

### Part 1: Loading Regression Dataset

```python
# EDA for Regression Problem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

np.random.seed(42)

# Load California Housing data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedianHouseValue'] = housing.target

print("=" * 60)
print("EDA: CALIFORNIA HOUSING (REGRESSION)")
print("=" * 60)

# ============================================================
# First Look
# ============================================================
print("\nDataset structure:")
print(f"  Shape: {df.shape} ({df.shape[0]:,} samples, {df.shape[1]-1} features)")
print(f"  Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print(f"\nFirst 3 rows:\n{df.head(3)}")
```

**Regression Part 1 Walkthrough:**

The California Housing dataset has over 20,000 samples, making it a medium-sized dataset suitable for regression analysis. The memory usage is manageable, and all features appear to be numeric.

### Part 2: Target Analysis for Regression

```python
# ============================================================
# Target Analysis (Regression-specific)
# ============================================================
print("\n" + "=" * 60)
print("TARGET VARIABLE ANALYSIS")
print("=" * 60)

target = df['MedianHouseValue']
print(f"\nTarget: MedianHouseValue")
print(f"  Mean:   ${target.mean():.2f} (hundreds of thousands)")
print(f"  Median: ${target.median():.2f}")
print(f"  Std:    ${target.std():.2f}")
print(f"  Min:    ${target.min():.2f}")
print(f"  Max:    ${target.max():.2f}")
# Output: Mean $2.07, Max $5.00 - note that max seems suspiciously round!

# Check for suspicious patterns
print(f"\n⚠️  ANOMALY DETECTED:")
print(f"  - Samples at max value (5.0): {(target == 5.0).sum()} ({(target == 5.0).sum()/len(df)*100:.1f}%)")
print(f"  - This suggests data was CAPPED at $500,000")
print(f"  - True values above 500k were truncated")
print(f"  - Implication: Model predictions may be biased for expensive homes")
# Output: ~800 samples exactly at 5.0 - a data collection artifact!

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(target, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].axvline(target.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {target.mean():.2f}')
axes[0].axvline(target.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {target.median():.2f}')
axes[0].set_xlabel('Median House Value ($100k)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Target Distribution')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Box plot
axes[1].boxplot(target, vert=True)
axes[1].set_ylabel('Median House Value ($100k)')
axes[1].set_title('Target Box Plot (Outlier Detection)')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_regression_target.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_regression_target.png")
```

**Regression Part 2 Walkthrough:**

A critical **data collection artifact** is discovered: the target is capped at $500k (5.0). This affects approximately 4% of samples and will bias model predictions for expensive homes. This is exactly the type of insight that only EDA can reveal—no algorithm would detect this, but it fundamentally affects modeling strategy.

### Part 3: Feature Distribution Analysis

```python
# ============================================================
# Feature Distributions
# ============================================================
print("\n" + "=" * 60)
print("FEATURE DISTRIBUTIONS")
print("=" * 60)

feature_cols = [col for col in df.columns if col != 'MedianHouseValue']

# Summary statistics
print("\nSummary statistics:")
print(df[feature_cols].describe())

# Check for skewness
from scipy import stats
print("\nSkewness (>1 suggests right skew, <-1 suggests left skew):")
for col in feature_cols:
    skew = stats.skew(df[col])
    print(f"  {col:20s}: {skew:6.2f}", end='')
    if abs(skew) > 1:
        print("  ← Consider log transform")
    else:
        print()
# Output: AveOccup is highly skewed (25.5!) - log transform candidate

# Visualize all feature distributions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, col in enumerate(feature_cols):
    axes[idx].hist(df[col], bins=40, edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(col)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_regression_features.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_regression_features.png")
```

**Regression Part 3 Walkthrough:**

The skewness analysis reveals that `AveOccup` is extremely skewed (skewness = 25.5), making it a strong candidate for log transformation. Most other features are reasonably distributed, though some show mild skewness.

### Part 4: Feature-Target Relationships

```python
# ============================================================
# Feature-Target Relationships
# ============================================================
print("\n" + "=" * 60)
print("FEATURE-TARGET RELATIONSHIPS")
print("=" * 60)

# Calculate correlations
correlations = df[feature_cols].corrwith(target).sort_values(ascending=False)
print("\nCorrelations with MedianHouseValue:")
for feat, corr in correlations.items():
    print(f"  {feat:20s}: {corr:6.3f}")
# Output: MedInc (median income) has strongest correlation (0.688)

# Visualize top 4 feature relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()
top_4_features = correlations.abs().nlargest(4).index

for idx, feature in enumerate(top_4_features):
    # Scatter plot with trend line
    axes[idx].scatter(df[feature], target, alpha=0.3, s=10)

    # Add trend line
    z = np.polyfit(df[feature], target, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df[feature].min(), df[feature].max(), 100)
    axes[idx].plot(x_trend, p(x_trend), "r--", linewidth=2, label='Trend')

    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('MedianHouseValue')
    axes[idx].set_title(f'{feature} vs Target (r={correlations[feature]:.3f})')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_regression_relationships.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_regression_relationships.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('eda_regression_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: eda_regression_heatmap.png")
```

**Regression Part 4 Walkthrough:**

`MedInc` (median income) emerges as the strongest predictor with correlation 0.688. Geographic features (latitude, longitude) show weaker linear correlation but may have non-linear spatial patterns that tree-based models could capture. The scatter plots reveal mostly linear relationships, though some noise is present.

### Part 5: Regression Summary

```python
# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("KEY FINDINGS - REGRESSION EDA")
print("=" * 60)

print("""
1. TARGET VARIABLE:
   - Right-skewed distribution (mean > median)
   - CRITICAL ISSUE: Target is capped at $500k (5.0)
     → ~4% of samples are at this ceiling
     → Model predictions for expensive homes will be biased
     → Consider removing these samples or treating separately

2. FEATURE DISTRIBUTIONS:
   - Most features reasonably distributed
   - AveOccup is extremely skewed (skewness = 25.5)
     → Log transform recommended before modeling

3. STRONGEST PREDICTORS:
   - MedInc (median income): r = 0.688 ← Most important!
   - Latitude: r = -0.145
   - Longitude: r = -0.047
   - Geographic features show weaker linear correlation but may have
     non-linear spatial patterns

4. FEATURE ENGINEERING OPPORTUNITIES:
   - Create location-based features (distance to city center, coastal proximity)
   - Combine AveRooms and AveBedrms into room ratios
   - Bin geographic coordinates into regions
   - Consider polynomial features for MedInc

5. MODELING RECOMMENDATIONS:
   - Linear regression baseline (interpret coefficients)
   - Tree-based models for non-linear relationships (Random Forest, XGBoost)
   - Feature scaling essential for regularized linear models
   - Consider removing or flagging capped target values
   - Use MAE or RMSE for evaluation (not R² alone)
   - Geographic features suggest spatial models or feature engineering
""")

print("=" * 60)
```

**Regression Part 5 Walkthrough:**

This regression example demonstrates key differences from classification EDA: focus on the **target distribution** (is it skewed? are transformations needed?), examination of **scatter plots** to understand linear vs. non-linear relationships, and discovery of a critical **data collection artifact** (the capped values at $500k) that will influence modeling approach.

## Examples: Messy Real-World Data

### Part 1: Initial Exploration of Messy Data

```python
# EDA on Messy Data with Missing Values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Load Titanic dataset (available through seaborn)
df_raw = sns.load_dataset('titanic')

print("=" * 60)
print("EDA: TITANIC SURVIVAL (MESSY DATA)")
print("=" * 60)

# ============================================================
# First Look - Understand the Mess
# ============================================================
print("\nDataset structure:")
print(f"  Shape: {df_raw.shape}")
print(f"\nColumn names and types:")
print(df_raw.dtypes)

print(f"\nFirst 3 rows:")
print(df_raw.head(3))
```

**Messy Data Part 1 Walkthrough:**

The Titanic dataset has mixed data types (numeric and categorical) and will likely have data quality issues typical of real-world historical data.

### Part 2: Missing Value Analysis

```python
# ============================================================
# Missing Value Analysis - CRITICAL for messy data
# ============================================================
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

# Count missing values per column
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
}).sort_values('Percentage', ascending=False)

print("\nMissing values by column:")
print(missing_df[missing_df['Missing_Count'] > 0])
# Output:
#   - deck: 77% missing ← Too much missing, likely drop this column
#   - age: 20% missing ← Need imputation strategy
#   - embark_town: 0.2% missing ← Very few, can drop rows or fill with mode

print(f"\n⚠️  KEY INSIGHTS:")
print(f"  - 'deck' is 77% missing → Too sparse to use reliably")
print(f"  - 'age' is 20% missing → Must impute (median by sex/class?)")
print(f"  - 'embark_town' barely missing → Fill with mode or drop rows")

# Visualize missing patterns
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
ax.set_title('Missing Value Patterns (Yellow = Missing)', fontsize=14, fontweight='bold')
ax.set_xlabel('Columns')
plt.tight_layout()
plt.savefig('eda_missing_patterns.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_missing_patterns.png")
```

**Messy Data Part 2 Walkthrough:**

Missing value analysis reveals varying degrees of missingness. The `deck` column is 77% missing and likely unusable. The `age` column is 20% missing and will require thoughtful imputation. The `embark_town` column has minimal missingness that can be easily addressed.

### Part 3: Target Variable and Categorical Features

```python
# ============================================================
# Target Variable - Survival Rate
# ============================================================
print("\n" + "=" * 60)
print("TARGET VARIABLE: SURVIVAL")
print("=" * 60)

survival_counts = df_raw['survived'].value_counts()
survival_rate = df_raw['survived'].mean()

print(f"\nOverall survival:")
print(f"  - Died (0): {survival_counts[0]} passengers ({survival_counts[0]/len(df_raw)*100:.1f}%)")
print(f"  - Survived (1): {survival_counts[1]} passengers ({survival_counts[1]/len(df_raw)*100:.1f}%)")
print(f"  - Survival rate: {survival_rate:.1%}")
# Output: 38% survived - imbalanced but manageable

# Visualize
fig, ax = plt.subplots(figsize=(8, 5))
survival_counts.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
ax.set_xticklabels(['Died', 'Survived'], rotation=0)
ax.set_ylabel('Count')
ax.set_xlabel('Outcome')
ax.set_title('Titanic Survival Distribution', fontsize=14, fontweight='bold')
for i, count in enumerate(survival_counts.values):
    ax.text(i, count + 10, str(count), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('eda_titanic_survival.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_titanic_survival.png")

# ============================================================
# Categorical Feature Analysis
# ============================================================
print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

categorical_cols = ['sex', 'pclass', 'embarked', 'who', 'embark_town']

for col in categorical_cols:
    print(f"\n{col.upper()}:")
    value_counts = df_raw[col].value_counts()
    print(value_counts)

    # Check for inconsistencies (in real data, look for typos, mixed case, etc.)
    n_unique = df_raw[col].nunique()
    print(f"  → {n_unique} unique values")
```

**Messy Data Part 3 Walkthrough:**

The survival rate of 38% represents moderate class imbalance that is manageable without extreme measures. Categorical feature analysis reveals the structure of variables like `sex`, `pclass`, and `embarked`.

### Part 4: Survival Analysis by Groups

```python
# Survival by categorical features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Survival by Sex
survival_by_sex = df_raw.groupby('sex')['survived'].agg(['sum', 'count', 'mean'])
ax = axes[0, 0]
survival_by_sex['mean'].plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
ax.set_title('Survival Rate by Sex', fontweight='bold')
ax.set_ylabel('Survival Rate')
ax.set_xlabel('Sex')
ax.set_xticklabels(['Female', 'Male'], rotation=0)
ax.axhline(y=survival_rate, color='gray', linestyle='--', label='Overall rate')
ax.legend()
for i, v in enumerate(survival_by_sex['mean']):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

# Survival by Class
survival_by_class = df_raw.groupby('pclass')['survived'].agg(['sum', 'count', 'mean'])
ax = axes[0, 1]
survival_by_class['mean'].plot(kind='bar', ax=ax, color=['#2ecc71', '#f39c12', '#e74c3c'])
ax.set_title('Survival Rate by Passenger Class', fontweight='bold')
ax.set_ylabel('Survival Rate')
ax.set_xlabel('Passenger Class')
ax.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
ax.axhline(y=survival_rate, color='gray', linestyle='--', label='Overall rate')
ax.legend()
for i, v in enumerate(survival_by_class['mean']):
    ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

# Survival by Embarkation Port
survival_by_port = df_raw.groupby('embarked')['survived'].agg(['sum', 'count', 'mean']).sort_values('mean', ascending=False)
ax = axes[1, 0]
survival_by_port['mean'].plot(kind='bar', ax=ax)
ax.set_title('Survival Rate by Embarkation Port', fontweight='bold')
ax.set_ylabel('Survival Rate')
ax.set_xlabel('Port')
ax.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'], rotation=45)
ax.axhline(y=survival_rate, color='gray', linestyle='--', label='Overall rate')
ax.legend()

# Age distribution by survival
ax = axes[1, 1]
df_raw[df_raw['survived']==0]['age'].hist(bins=30, alpha=0.6, label='Died', ax=ax, color='#e74c3c')
df_raw[df_raw['survived']==1]['age'].hist(bins=30, alpha=0.6, label='Survived', ax=ax, color='#2ecc71')
ax.set_title('Age Distribution by Survival', fontweight='bold')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Frequency')
ax.legend()

plt.tight_layout()
plt.savefig('eda_titanic_relationships.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_titanic_relationships.png")
```

**Messy Data Part 4 Walkthrough:**

Clear patterns emerge: sex and passenger class are strong predictors of survival. Women had much higher survival rates than men, and first-class passengers fared better than third-class passengers.

### Part 5: Numeric Features and Interactions

```python
# ============================================================
# Numeric Features
# ============================================================
print("\n" + "=" * 60)
print("NUMERIC FEATURES")
print("=" * 60)

numeric_cols = ['age', 'fare', 'sibsp', 'parch']

print("\nSummary statistics:")
print(df_raw[numeric_cols].describe())

# Fare is highly skewed - check it
print(f"\n⚠️  FARE ANALYSIS:")
print(f"  Mean fare: ${df_raw['fare'].mean():.2f}")
print(f"  Median fare: ${df_raw['fare'].median():.2f}")
print(f"  Max fare: ${df_raw['fare'].max():.2f}")
print(f"  → Mean >> Median suggests strong right skew")
print(f"  → Log transform recommended for modeling")

# ============================================================
# Feature Interactions - Sex AND Class
# ============================================================
print("\n" + "=" * 60)
print("FEATURE INTERACTIONS")
print("=" * 60)

# Survival by Sex and Class (interaction)
interaction = df_raw.groupby(['sex', 'pclass'])['survived'].mean().reset_index()
interaction_pivot = interaction.pivot(index='pclass', columns='sex', values='survived')

print("\nSurvival rate by Sex AND Class:")
print(interaction_pivot)
print("\n💡 INSIGHTS:")
print("  - Female + 1st class: Highest survival (~97%)")
print("  - Male + 3rd class: Lowest survival (~14%)")
print("  - Clear interaction effect: both sex and class matter")

fig, ax = plt.subplots(figsize=(10, 6))
interaction_pivot.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
ax.set_title('Survival Rate by Sex and Class (Interaction Effect)', fontsize=14, fontweight='bold')
ax.set_ylabel('Survival Rate')
ax.set_xlabel('Passenger Class')
ax.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
ax.legend(title='Sex', labels=['Female', 'Male'])
ax.axhline(y=survival_rate, color='gray', linestyle='--', linewidth=1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('eda_titanic_interaction.png', dpi=300, bbox_inches='tight')
print("\nSaved: eda_titanic_interaction.png")
```

**Messy Data Part 5 Walkthrough:**

The `fare` variable is highly skewed and will benefit from log transformation. The interaction analysis reveals that both sex and class matter: first-class women had ~97% survival while third-class men had only ~14% survival. This suggests interaction terms may improve model performance.

### Part 6: Data Quality Summary and Recommendations

```python
# ============================================================
# Data Quality Issues Summary
# ============================================================
print("\n" + "=" * 60)
print("DATA QUALITY SUMMARY")
print("=" * 60)

n_duplicates = df_raw.duplicated().sum()
print(f"\nDuplicate rows: {n_duplicates}")

print("\n✅ WHAT'S GOOD:")
print("  - No duplicates detected")
print("  - Categorical variables are clean (no typos or inconsistencies)")
print("  - Clear, interpretable relationships with target")
print("  - Moderate size (891 rows) allows thorough analysis")

print("\n⚠️  ISSUES TO ADDRESS:")
print("  1. Missing 'deck' (77%) → DROP this column")
print("  2. Missing 'age' (20%) → IMPUTE with median by sex/class")
print("  3. Missing 'embark_town' (0.2%) → DROP rows or fill with mode")
print("  4. 'fare' is highly skewed → LOG TRANSFORM before modeling")
print("  5. Possible data leakage: 'embarked' might correlate with survival")
print("     due to ticket price/class, not embarkation point itself")

# ============================================================
# Modeling Recommendations
# ============================================================
print("\n" + "=" * 60)
print("MODELING RECOMMENDATIONS")
print("=" * 60)

print("""
1. PREPROCESSING STEPS:
   a. Drop 'deck' column (too much missing data)
   b. Impute 'age': median by sex and class
   c. Handle 'embark_town': fill 2 missing values with mode ('Southampton')
   d. Log transform 'fare' to handle skewness
   e. Create 'family_size' feature (sibsp + parch + 1)
   f. Create 'is_alone' binary feature (family_size == 1)

2. FEATURE ENCODING:
   - One-hot encode: 'sex', 'pclass', 'embarked'
   - Consider: bin 'age' into categories (child/adult/senior)
   - Consider: bin 'fare' into quartiles after log transform

3. KEY FEATURES (based on EDA):
   - sex (strongest predictor!)
   - pclass (strong predictor)
   - age (moderate, after imputation)
   - fare (moderate, after log transform)
   - family_size (engineered feature)

4. MODEL SUGGESTIONS:
   - Logistic Regression (baseline, interpretable)
   - Random Forest (handles missing patterns, interactions)
   - XGBoost (likely best performance)
   - Neural network likely overkill for this size/complexity

5. EVALUATION:
   - Class imbalance (62% died, 38% survived) is moderate
   - Use stratified k-fold cross-validation
   - Report: accuracy, precision, recall, F1, ROC-AUC
   - Consider confusion matrix (false positive vs false negative costs)

6. EXPECTED PERFORMANCE:
   - Based on strong sex/class relationships: ~80-85% accuracy achievable
   - Most misclassifications likely in middle class (2nd/3rd class males)
""")

print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)
```

**Messy Data Part 6 Walkthrough:**

This messy data example shows the **real value of EDA**: critical data quality issues were discovered (massive missing values in `deck`, skewed `fare` distribution), strong predictors were identified (`sex` and `pclass`), and feature interactions were uncovered (women in first class had 97% survival rate while men in third class had only 14%). These insights directly inform preprocessing pipeline and feature engineering strategy.

## Common Pitfalls

**1. Skipping EDA or Rushing Through It**

The mistake: Beginners often want to jump straight to modeling, treating EDA as a tedious formality. Running `df.describe()`, glancing at the output, and immediately starting to train models.

Why this happens: Modeling feels like "real" data science—it's exciting to train algorithms and see accuracy scores. EDA can feel like busywork by comparison.

The consequence: Critical issues that sabotage models are missed. Perhaps 30% of the target variable is mislabeled, or two features are identical (data entry duplication), or the "time series" data has massive gaps. The model trains and even produces good metrics on the test set, but fails catastrophically in production because the data's limitations were not understood.

What to do instead: Embrace EDA as detective work. Set aside dedicated time (often 60-80% of project timeline) for thorough exploration. Use the 5-step framework systematically. Document findings in a well-organized notebook with markdown cells explaining what was discovered and why it matters. Remember: time spent in EDA is time saved debugging broken models later.

**2. Treating Correlation as Causation**

The mistake: After finding a strong correlation (say, r = 0.85 between ice cream sales and drowning deaths), concluding that one causes the other and making modeling decisions based on this assumption.

Why this happens: Humans are pattern-seeking creatures. When two variables move together, brains instinctively assume a causal relationship. The mantra "correlation does not equal causation" is drilled into statistics classes, but students often flip to the opposite error: assuming correlation *never* means causation.

The consequence: A genuinely predictive feature might be removed because the correlation isn't believed to be "real," or a spurious feature might be included that merely reflects a confounding variable. Worse, misleading insights might be presented to stakeholders ("Our analysis proves that feature X drives outcome Y") when no such causal relationship exists.

What to do instead: Understand the nuanced truth: correlation doesn't *always* mean causation, but it doesn't *never* mean causation either. When strong correlations are found, ask deeper questions: Is there a plausible mechanism? Could a third variable (confounder) explain both? Does the relationship hold across different subgroups? Does domain knowledge support a causal link? In the ice cream example, the confounding variable is temperature—hot weather increases both ice cream sales and swimming (hence drownings). For predictive modeling, correlation is often sufficient even without causation, but for decision-making and interpretation, the distinction is critical.

**3. Ignoring Data Collection Context**

The mistake: Analyzing data without understanding how, when, why, or by whom it was collected. Treating the dataset as an abstract mathematical object divorced from its real-world origin.

Why this happens: Datasets often arrive as clean CSV files with no documentation. It's tempting to accept them at face value and start exploring. Additionally, the data collection process seems irrelevant to statistical analysis.

The consequence: Crucial insights that explain anomalies are missed. For example, why does the California Housing dataset cap values at $500k? (Data collection artifact from 1990 census). Why is the `cabin` feature 77% missing in Titanic data? (Only first-class passengers reliably had cabin assignments recorded). Why do certain days show zero values in a retail dataset? (Store was closed for holidays). Without this context, artifacts might be treated as real patterns or real patterns as errors.

What to do instead: Before touching the data, ask: Where did this data come from? Who collected it and why? When was it collected? What decisions were made during collection (sampling strategy, measurement instruments, data entry procedures)? Are there known limitations or biases? Read documentation, talk to subject matter experts, and research the data's provenance. Add this context to the EDA notebook's introduction. When anomalies are discovered, investigate their origin rather than mechanically "fixing" them.

## Practice

**Practice 1**

Load the Wine dataset from sklearn (`load_wine()`) and perform a basic but thorough EDA.

1. Load the dataset and convert it to a Pandas DataFrame with feature names
2. Display the shape, data types, and first 5 rows
3. Generate summary statistics using `.describe()`
4. Check for missing values (report total and per-column counts)
5. Create a histogram for the `alcohol` feature
6. Display the target class distribution and visualize it with a bar plot
7. Calculate and display the correlation between each feature and the target

Answer these interpretation questions:
- How many samples and features are in the dataset?
- What is the mean and standard deviation of `alcohol`?
- Are the three wine classes balanced (equal representation)?
- Which feature shows the strongest correlation with the target?
- Based on the alcohol histogram, is the distribution approximately normal or skewed?

**Practice 2**

Analyze the Diabetes dataset (`load_diabetes()` from sklearn) to predict disease progression one year later.

1. **Data Loading & Target Analysis:**
   - Load the dataset into a DataFrame with feature names
   - Create a histogram of the target variable
   - Calculate mean, median, standard deviation, and check for skewness
   - Determine if the distribution is approximately normal or skewed

2. **Feature-Target Relationships:**
   - Calculate the Pearson correlation between each feature and the target
   - Sort correlations by absolute value and identify the top 3 most correlated features
   - Create a 1×3 subplot showing scatter plots for these top 3 features vs. target
   - Add trend lines to the scatter plots (hint: use `np.polyfit`)

3. **Multicollinearity Investigation:**
   - Create a correlation heatmap for all features (use seaborn)
   - Identify pairs of features with correlation > 0.7 or < -0.7
   - Explain what high feature-feature correlation means for modeling

4. **Interpretation:**
   - Write a 3-5 sentence summary of findings
   - Which features seem most predictive? Are relationships linear or non-linear?
   - What preprocessing or modeling decisions do the findings suggest?

Bonus challenge:
- Create a pair plot for the top 4 features and target using `sns.pairplot()`
- Investigate whether any features would benefit from transformation (log, square root, etc.)

**Practice 3**

**Scenario:** As a data scientist at a telecom company, the team has collected customer data to predict churn (customers canceling their subscriptions). Before building a predictive model, the data quality must be thoroughly understood and documented.

**Dataset:** Use the Telco Customer Churn dataset from Kaggle or create synthetic data with intentional issues using this code:

```python
# Generate synthetic telecom churn data with realistic messiness
import numpy as np
import pandas as pd
np.random.seed(42)

n = 1000
df = pd.DataFrame({
    'customer_id': range(1, n+1),
    'age': np.random.randint(18, 80, n),
    'tenure_months': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(20, 120, n),
    'total_charges': np.random.uniform(100, 8000, n),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No', 'none'], n, p=[0.3, 0.4, 0.2, 0.1]),
    'churn': np.random.choice([0, 1], n, p=[0.73, 0.27])
})

# Introduce realistic data quality issues
# 1. Missing values (realistic patterns)
missing_idx = np.random.choice(df.index, size=80, replace=False)
df.loc[missing_idx, 'age'] = np.nan

# 2. Inconsistencies in categorical data
df.loc[df['internet_service'] == 'No', 'internet_service'] = np.random.choice(['No', 'no', 'NO', 'None'],
                                                                                sum(df['internet_service'] == 'No'))

# 3. Outliers (possible data entry errors)
df.loc[np.random.choice(df.index, 5), 'age'] = -1  # Invalid age
df.loc[np.random.choice(df.index, 3), 'monthly_charges'] = 0  # Free service?

# 4. Duplicates
duplicate_rows = df.sample(10)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 5. Logical inconsistencies
# Some customers have total_charges < monthly_charges * tenure
problem_idx = df.sample(20).index
df.loc[problem_idx, 'total_charges'] = df.loc[problem_idx, 'monthly_charges'] * 0.5
```

**Part 1: Initial Exploration**
1. Load and display basic structure (shape, columns, types)
2. Identify which features are numeric vs. categorical
3. Calculate memory usage

**Part 2: Data Quality Investigation**
4. Find all missing values and calculate percentage missing per feature
5. Visualize missing patterns with a heatmap
6. Detect outliers in numeric features using both box plots and the IQR method
7. Identify duplicate records (exact matches)
8. Find logical inconsistencies:
   - Negative ages or ages > 120
   - Monthly charges = 0 (free service or data error?)
   - Total charges < monthly charges (impossible)
   - Tenure = 0 months but total charges > 0

**Part 3: Categorical Data Analysis**
9. For each categorical feature:
   - Count unique values
   - Display value counts
   - Check for inconsistencies (mixed case, typos, synonyms like "No"/"no"/"NO")
   - Visualize distributions with count plots

**Part 4: Target Variable Analysis**
10. Calculate churn rate (percentage of customers who left)
11. Assess class imbalance—is it severe enough to require special handling?
12. Visualize target distribution

**Part 5: Relationship Exploration**
13. Create visualizations showing how features relate to churn:
    - Box plots: numeric features by churn status
    - Grouped bar charts: categorical features by churn status
14. Calculate correlations between numeric features and churn
15. Identify which features seem most predictive (highest separation between classes)

**Part 6: Data Quality Report**
Write a professional report summarizing:
- **Top 5 data quality issues** found (be specific—not just "missing values" but "age is missing for 8% of customers, appears random")
- **Recommended actions** for each issue (impute with median? drop rows? investigate further?)
- **3-5 key insights** about churn patterns (e.g., "customers on month-to-month contracts churn at 3× the rate of two-year contract customers")
- **Suggested next steps** for preprocessing and feature engineering

**Part 7: Bonus Challenges**
- Create an automated EDA function that accepts any DataFrame and generates a report
- Use `pandas-profiling` or `sweetviz` to generate an automated report, then compare: what was caught that the automated tool missed?
- Investigate feature interactions: does the relationship between monthly charges and churn differ by contract type?

## Solutions

**Solution 1**

```python
from sklearn.datasets import load_wine
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Basic structure
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Summary statistics
print(f"\nSummary statistics:\n{df.describe()}")

# Missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Total missing: {df.isnull().sum().sum()}")

# Histogram of alcohol
plt.figure(figsize=(8, 5))
plt.hist(df['alcohol'], bins=20, edgecolor='black')
plt.xlabel('Alcohol')
plt.ylabel('Frequency')
plt.title('Distribution of Alcohol Content')
plt.savefig('wine_alcohol_hist.png')

# Target distribution
target_counts = df['target'].value_counts()
print(f"\nTarget distribution:\n{target_counts}")

plt.figure(figsize=(8, 5))
target_counts.plot(kind='bar')
plt.xlabel('Wine Class')
plt.ylabel('Count')
plt.title('Target Class Distribution')
plt.savefig('wine_target_dist.png')

# Correlations with target
correlations = df.drop('target', axis=1).corrwith(df['target']).sort_values(ascending=False)
print(f"\nFeature correlations with target:\n{correlations}")

# Answers:
# - 178 samples, 13 features
# - Mean alcohol: ~13.0, Std: ~0.8
# - Classes are reasonably balanced (59, 71, 48)
# - Strongest correlation: flavanoids (~0.85)
# - Alcohol distribution appears approximately normal with slight right skew
```

The Wine dataset is clean (no missing values), well-balanced, and has strong predictive features. The `flavanoids` feature shows the strongest correlation with wine class, suggesting it will be highly predictive in classification models.

**Solution 2**

```python
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Target analysis
target = df['target']
print(f"Target statistics:")
print(f"  Mean: {target.mean():.2f}")
print(f"  Median: {target.median():.2f}")
print(f"  Std: {target.std():.2f}")
print(f"  Skewness: {stats.skew(target):.2f}")

plt.figure(figsize=(8, 5))
plt.hist(target, bins=30, edgecolor='black')
plt.xlabel('Disease Progression')
plt.ylabel('Frequency')
plt.title('Target Distribution')
plt.savefig('diabetes_target_hist.png')

# Feature-target correlations
feature_cols = [col for col in df.columns if col != 'target']
correlations = df[feature_cols].corrwith(target).sort_values(key=abs, ascending=False)
print(f"\nTop 3 correlated features:\n{correlations.head(3)}")

# Scatter plots with trend lines
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
top_3 = correlations.head(3).index

for idx, feature in enumerate(top_3):
    axes[idx].scatter(df[feature], target, alpha=0.5)

    # Trend line
    z = np.polyfit(df[feature], target, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df[feature].min(), df[feature].max(), 100)
    axes[idx].plot(x_trend, p(x_trend), 'r--', linewidth=2)

    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Target')
    axes[idx].set_title(f'{feature} (r={correlations[feature]:.3f})')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diabetes_top_features.png')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('diabetes_heatmap.png')

# Identify multicollinearity
corr_matrix = df[feature_cols].corr()
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print(f"\nHigh correlations (>0.7):")
for feat1, feat2, corr in high_corr:
    print(f"  {feat1} - {feat2}: {corr:.3f}")

print("\nSummary:")
print("The target is approximately normally distributed with slight right skew.")
print(f"Top predictors are {top_3[0]}, {top_3[1]}, and {top_3[2]}.")
print("Relationships appear mostly linear based on scatter plots.")
print("High multicollinearity exists between some features, suggesting regularization or PCA.")
print("Preprocessing recommendations: standardize features, consider Ridge/Lasso regression.")
```

The Diabetes dataset shows moderate correlations with the target (strongest ~0.5), suggesting prediction will be challenging but feasible. The presence of multicollinearity indicates that regularized models or dimensionality reduction would be beneficial.

**Solution 3**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data (code provided in exercise)
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'customer_id': range(1, n+1),
    'age': np.random.randint(18, 80, n),
    'tenure_months': np.random.randint(1, 72, n),
    'monthly_charges': np.random.uniform(20, 120, n),
    'total_charges': np.random.uniform(100, 8000, n),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No', 'none'], n, p=[0.3, 0.4, 0.2, 0.1]),
    'churn': np.random.choice([0, 1], n, p=[0.73, 0.27])
})

# Introduce data quality issues
missing_idx = np.random.choice(df.index, size=80, replace=False)
df.loc[missing_idx, 'age'] = np.nan
df.loc[df['internet_service'] == 'No', 'internet_service'] = np.random.choice(['No', 'no', 'NO', 'None'],
                                                                                sum(df['internet_service'] == 'No'))
df.loc[np.random.choice(df.index, 5, replace=False), 'age'] = -1
df.loc[np.random.choice(df.index, 3, replace=False), 'monthly_charges'] = 0
duplicate_rows = df.sample(10)
df = pd.concat([df, duplicate_rows], ignore_index=True)
problem_idx = df.sample(20).index
df.loc[problem_idx, 'total_charges'] = df.loc[problem_idx, 'monthly_charges'] * 0.5

# Part 1: Initial exploration
print("=" * 60)
print("INITIAL EXPLORATION")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumeric features: {numeric_cols}")
print(f"Categorical features: {categorical_cols}")

# Part 2: Data quality
print("\n" + "=" * 60)
print("DATA QUALITY INVESTIGATION")
print("=" * 60)

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
print(f"\nMissing values:")
for col, count, pct in zip(missing.index, missing.values, missing_pct.values):
    if count > 0:
        print(f"  {col}: {count} ({pct:.1f}%)")

# Duplicates
n_duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {n_duplicates}")

# Outliers and logical issues
print(f"\nLogical inconsistencies:")
invalid_age = ((df['age'] < 0) | (df['age'] > 120)).sum()
print(f"  Invalid ages: {invalid_age}")

zero_charges = (df['monthly_charges'] == 0).sum()
print(f"  Zero monthly charges: {zero_charges}")

# Categorical inconsistencies
print(f"\nInternet service values:")
print(df['internet_service'].value_counts())
print("  → Inconsistent capitalization detected!")

# Part 3: Target analysis
print("\n" + "=" * 60)
print("TARGET ANALYSIS")
print("=" * 60)
churn_rate = df['churn'].mean()
print(f"Churn rate: {churn_rate:.1%}")
print(f"Class balance: {df['churn'].value_counts()}")

# Part 4: Relationships
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age by churn
df.boxplot(column='age', by='churn', ax=axes[0, 0])
axes[0, 0].set_title('Age by Churn Status')

# Monthly charges by churn
df.boxplot(column='monthly_charges', by='churn', ax=axes[0, 1])
axes[0, 1].set_title('Monthly Charges by Churn Status')

# Contract type by churn
churn_by_contract = df.groupby('contract_type')['churn'].mean()
churn_by_contract.plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_title('Churn Rate by Contract Type')
axes[1, 0].set_ylabel('Churn Rate')

# Tenure by churn
df.boxplot(column='tenure_months', by='churn', ax=axes[1, 1])
axes[1, 1].set_title('Tenure by Churn Status')

plt.tight_layout()
plt.savefig('telecom_churn_eda.png')

print("\n" + "=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)
print("""
TOP 5 DATA QUALITY ISSUES:
1. Missing age values: 8% of customers (80 records)
   → Recommendation: Impute with median age

2. Duplicate records: 10 exact duplicates found
   → Recommendation: Remove duplicates

3. Invalid age values: 5 customers with age = -1
   → Recommendation: Treat as missing and impute

4. Zero monthly charges: 3 customers with $0 charges
   → Recommendation: Investigate if legitimate (promotional?) or error

5. Inconsistent categorical values: 'No'/'no'/'NO'/'None' for internet service
   → Recommendation: Standardize to single value (e.g., 'No')

KEY INSIGHTS:
- Churn rate is 27% (moderate imbalance)
- Month-to-month contracts likely show higher churn than longer contracts
- Customers with longer tenure appear less likely to churn
- Monthly charges may differ between churned and retained customers

NEXT STEPS:
1. Clean categorical values (standardize capitalization)
2. Remove duplicates
3. Impute missing age values
4. Investigate zero-charge customers
5. Create engineered features (e.g., charges_per_month, is_long_term_contract)
6. Consider encoding categorical variables
7. Build baseline logistic regression model
""")
```

This comprehensive EDA reveals multiple data quality issues typical of real-world datasets: missing values, duplicates, logical inconsistencies, and categorical data problems. The systematic investigation provides a clear roadmap for data cleaning and preprocessing before modeling begins.

## Key Takeaways

- **EDA is the essential foundation of data science.** Good modeling decisions cannot be made without first understanding data structure, quality, and patterns. Professional data scientists spend 60-80% of project time on data exploration and preparation.

- **Follow a systematic framework.** The 5-step process (First Look → Individual Features → Target Variable → Relationships → Data Quality) ensures critical aspects of data are not missed. Each step reveals information that informs the next.

- **EDA is question-driven exploration, not mechanical box-checking.** Approach datasets like a detective investigating a crime scene. Ask: What does this represent? What patterns emerge? What's unusual? What's missing? Let discoveries guide to new questions.

- **Visualization is essential for understanding.** Summary statistics alone miss crucial patterns. Use histograms to see distributions, box plots to spot outliers, scatter plots to reveal relationships, and heatmaps to uncover correlations. Choose visualizations that answer specific questions.

- **Data quality issues are the rule, not the exception.** Real-world data has missing values, outliers, duplicates, inconsistencies, and errors. EDA surfaces these issues early so they can be addressed appropriately. Detection is easy; deciding what to do requires domain knowledge and critical thinking.

- **Document everything during exploration.** EDA findings are worthless if not recorded. Use Jupyter notebooks with markdown cells explaining what was discovered, why it matters, and what decisions it suggests. Future self (and team members) will benefit from this documentation.

- **Context matters profoundly.** Understanding how, when, and why data was collected explains anomalies and informs modeling choices. A capped value might be a data collection artifact. Missing values might follow systematic patterns. Always investigate the data's origin story.

- **EDA directly informs modeling decisions.** Every finding suggests actions: missing values → imputation strategy; skewed distributions → transformations; high correlations → feature selection; class imbalance → resampling or adjusted metrics; strong feature-target relationships → algorithm selection. The goal isn't just to understand data but to prepare for successful modeling.

**Next:** Chapter 11 covers handling missing data and outliers, building on the data quality issues identified during EDA.

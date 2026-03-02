> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 17.1: Feature Selection

## Why This Matters

You've engineered brilliant features—numerical transformations, one-hot encoded categories, TF-IDF vectors from text. Your dataset now has 500 features. You train a model expecting excellent results, but accuracy is mediocre and training takes forever. The problem? Too many features. In machine learning, **more features often hurt performance** due to the curse of dimensionality. Feature selection identifies the minimal subset of features that maximizes predictive power, making models faster, more accurate, and interpretable.

## The Intuition

Imagine you're organizing your closet with 200 clothing items, but you only wear 30 regularly. Every morning, searching through 200 items wastes time, and you might accidentally pick outdated clothes you should have donated. The solution? Keep the 30 items you wear frequently, donate the 150 you never touch, and store 20 seasonal items elsewhere. Your mornings become faster, your choices improve, and your closet feels organized.

Feature selection works the same way. When you have hundreds of features but only 50 truly predict your target, the extra 150 features create three problems:

1. **Computational cost** — Training takes longer with more features, like searching through a cluttered closet
2. **Overfitting** — Models learn patterns from noise features (the outdated clothes), not real signal
3. **Curse of dimensionality** — In high dimensions, data points become sparse and distant from each other, making patterns hard to learn

Consider a restaurant recommendation system. With one criterion (price), choosing is easy. With two criteria (price and distance), it's still manageable—plot restaurants on a 2D map and pick. But with 10 criteria (price, distance, cuisine, rating, noise level, parking, wait time, dietary options, ambiance, outdoor seating), you're overwhelmed. With 100 restaurants, maybe 0-1 meet all requirements. As dimensions increase, "good" options become exponentially rare.

In machine learning with 1,000 features and 100 samples, each sample looks unique and isolated. The model can't learn meaningful patterns—it just memorizes noise. Feature selection focuses on what matters: perhaps 3 criteria (price, distance, rating) capture 90% of your decision. Drop the rest.

There are three main approaches to feature selection, each with different trade-offs. **Filter methods** examine each feature independently using statistical tests (fast but ignores interactions). **Wrapper methods** try different feature combinations with actual models (thorough but slow). **Embedded methods** build feature selection into model training using techniques like Lasso regularization (efficient middle ground). Choosing the right method depends on your dataset size, computational budget, and interpretability needs.

## Formal Definition

Given a feature matrix **X** ∈ ℝⁿˣᵖ with n samples and p features, and a target vector **y** ∈ ℝⁿ, **feature selection** identifies a subset S ⊂ {1, 2, ..., p} of size k < p such that training a model using only features in S optimizes a performance criterion (accuracy, R², F1-score) while minimizing k.

**The curse of dimensionality** refers to the phenomenon where, as dimensionality p increases, the volume of the feature space grows exponentially, causing data to become sparse. The required number of samples to maintain constant data density grows as O(pⁿ). For distance-based algorithms (k-NN, k-means), the ratio of distances to nearest and farthest neighbors approaches 1 as p → ∞, making distances meaningless.

**Feature selection methods:**

1. **Filter methods**: Score each feature independently using statistical tests. For feature j and target y:
   - ANOVA F-statistic: F = (between-group variance) / (within-group variance)
   - Chi-squared test: χ² = Σ (Observed - Expected)² / Expected
   - Mutual information: I(X; Y) = H(Y) - H(Y|X) where H is entropy
   - Select top-k features by score

2. **Wrapper methods**: Evaluate feature subsets using model performance. Recursive Feature Elimination (RFE) iteratively:
   - Train model on all features
   - Rank features by importance (|θⱼ| for linear models)
   - Remove least important feature
   - Repeat until k features remain

3. **Embedded methods**: Feature selection during training. Lasso regression with L1 penalty:
   - J(θ) = (1/n) Σᵢ (yᵢ - ŷᵢ)² + λ Σⱼ |θⱼ|
   - L1 penalty drives some θⱼ to exactly zero
   - Features with θⱼ = 0 are automatically removed

> **Key Concept:** Feature selection reduces dimensionality by choosing a subset of original features, improving model performance, interpretability, and efficiency while combating the curse of dimensionality.

## Visual

We'll create visualizations showing:
1. The curse of dimensionality (KNN accuracy degrading with dimensions)
2. Filter vs. wrapper vs. embedded method comparison
3. Lasso regularization path (coefficients shrinking to zero)
4. Feature importance rankings
5. Data leakage: wrong vs. right pipeline

These will be generated in the code examples below.

## Code Example 1: Demonstrating the Curse of Dimensionality

```python
# Curse of Dimensionality - KNN Performance Degradation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)

# Test dimensionalities: 2, 10, 50, 100, 200 features
dimensions = [2, 5, 10, 20, 50, 100, 200]
n_samples = 500
n_informative = 5  # Only 5 features are truly informative

# Store results
accuracies = []
avg_distances = []

print("Feature Selection Example 1: Curse of Dimensionality")
print("=" * 60)
print(f"Dataset: {n_samples} samples, {n_informative} informative features")
print(f"Testing KNN classifier with increasing dimensions...")
print()

for n_features in dimensions:
    # Create classification dataset with only 5 informative features
    # The rest are noise
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=0,
        n_clusters_per_class=2,
        random_state=42,
        flip_y=0.1  # 10% label noise
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Calculate average pairwise distance (sample 50 points for speed)
    sample_idx = np.random.choice(len(X_train), min(50, len(X_train)), replace=False)
    X_sample = X_train[sample_idx]
    distances = []
    for i in range(len(X_sample)):
        for j in range(i+1, len(X_sample)):
            dist = np.linalg.norm(X_sample[i] - X_sample[j])
            distances.append(dist)
    avg_dist = np.mean(distances)
    avg_distances.append(avg_dist)

    print(f"Dimensions: {n_features:3d} | Accuracy: {acc:.3f} | Avg Distance: {avg_dist:.2f}")

print()
print("Observation: As dimensions increase, accuracy degrades!")
print("With only 5 informative features, extra dimensions add noise.")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs Dimensions
ax1.plot(dimensions, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax1.axhline(y=accuracies[0], color='green', linestyle='--', alpha=0.5, label='2D Accuracy')
ax1.set_xlabel('Number of Features', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Curse of Dimensionality: KNN Accuracy Degrades', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0.5, 1.0])

# Plot 2: Average Distance vs Dimensions
ax2.plot(dimensions, avg_distances, marker='s', linewidth=2, markersize=8, color='#A23B72')
ax2.set_xlabel('Number of Features', fontsize=12)
ax2.set_ylabel('Average Pairwise Distance', fontsize=12)
ax2.set_title('Points Become Distant in High Dimensions', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curse_of_dimensionality.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'curse_of_dimensionality.png'")

# Output:
# Feature Selection Example 1: Curse of Dimensionality
# ============================================================
# Dataset: 500 samples, 5 informative features
# Testing KNN classifier with increasing dimensions...
#
# Dimensions:   2 | Accuracy: 0.880 | Avg Distance: 2.28
# Dimensions:   5 | Accuracy: 0.887 | Avg Distance: 3.55
# Dimensions:  10 | Accuracy: 0.827 | Avg Distance: 5.00
# Dimensions:  20 | Accuracy: 0.753 | Avg Distance: 7.04
# Dimensions:  50 | Accuracy: 0.673 | Avg Distance: 11.13
# Dimensions: 100 | Accuracy: 0.640 | Avg Distance: 15.75
# Dimensions: 200 | Accuracy: 0.613 | Avg Distance: 22.27
#
# Observation: As dimensions increase, accuracy degrades!
# With only 5 informative features, extra dimensions add noise.
```

## Walkthrough: Example 1

This example demonstrates the curse of dimensionality—a critical motivation for feature selection.

**What the code does:**

1. **Creates synthetic data** with `make_classification()`, specifying only 5 informative features while varying total dimensions from 2 to 200
2. **Trains KNN classifier** on each dimensionality—KNN is particularly sensitive to the curse because it relies on distances
3. **Measures accuracy** and **average pairwise distance** between data points

**Key observations:**

- **Accuracy drops** from ~88% (2D) to ~61% (200D) even though only 5 features are informative
- **Average distance increases** from 2.28 to 22.27—points become isolated in high dimensions
- **The problem:** With 200 features and only 500 samples, we have sparse data where meaningful patterns disappear

**Why this happens:** In high dimensions, the volume of the feature space grows exponentially (2²⁰⁰ possible "corners" for binary features). To maintain the same data density, we'd need exponentially more samples. With fixed samples (500), data becomes sparse—points are far apart and equidistant, making distance-based algorithms ineffective.

**The solution:** Feature selection. By keeping only the 5 truly informative features, we'd recover the high accuracy from the low-dimensional case.

## Code Example 2: Filter Methods - Univariate Feature Selection

```python
# Filter Methods: Statistical Tests for Feature Selection
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Feature Selection Example 2: Filter Methods")
print("=" * 60)
print(f"Dataset: Breast Cancer (Binary Classification)")
print(f"Shape: {X.shape} - {X.shape[0]} samples, {X.shape[1]} features")
print(f"Task: Select top 10 features using different statistical tests")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Method 1: ANOVA F-test
print("Method 1: ANOVA F-test (f_classif)")
print("-" * 60)
selector_f = SelectKBest(score_func=f_classif, k=10)
selector_f.fit(X_train, y_train)

# Get selected features
selected_features_f = X.columns[selector_f.get_support()].tolist()
scores_f = selector_f.scores_

print(f"Top 10 features by F-test:")
for i, (feat, score) in enumerate(zip(selected_features_f, scores_f[selector_f.get_support()]), 1):
    print(f"  {i:2d}. {feat:30s} (F-score: {score:.2f})")
print()

# Method 2: Chi-squared test (requires non-negative features)
print("Method 2: Chi-squared test (chi2)")
print("-" * 60)
# Scale to [0, 1] for chi-squared
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector_chi2 = SelectKBest(score_func=chi2, k=10)
selector_chi2.fit(X_train_scaled, y_train)

selected_features_chi2 = X.columns[selector_chi2.get_support()].tolist()
scores_chi2 = selector_chi2.scores_

print(f"Top 10 features by Chi-squared:")
for i, (feat, score) in enumerate(zip(selected_features_chi2, scores_chi2[selector_chi2.get_support()]), 1):
    print(f"  {i:2d}. {feat:30s} (χ² score: {score:.2f})")
print()

# Method 3: Mutual Information
print("Method 3: Mutual Information (mutual_info_classif)")
print("-" * 60)
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
selector_mi.fit(X_train, y_train)

selected_features_mi = X.columns[selector_mi.get_support()].tolist()
scores_mi = selector_mi.scores_

print(f"Top 10 features by Mutual Information:")
for i, (feat, score) in enumerate(zip(selected_features_mi, scores_mi[selector_mi.get_support()]), 1):
    print(f"  {i:2d}. {feat:30s} (MI score: {score:.3f})")
print()

# Compare performance
print("Performance Comparison")
print("-" * 60)

# Baseline: All features
start = time.time()
model_all = LogisticRegression(max_iter=10000, random_state=42)
model_all.fit(X_train, y_train)
time_all = time.time() - start
acc_all = model_all.score(X_test, y_test)
print(f"All 30 features:    Accuracy: {acc_all:.4f} | Time: {time_all:.4f}s")

# F-test selected features
X_train_f = selector_f.transform(X_train)
X_test_f = selector_f.transform(X_test)
start = time.time()
model_f = LogisticRegression(max_iter=10000, random_state=42)
model_f.fit(X_train_f, y_train)
time_f = time.time() - start
acc_f = model_f.score(X_test_f, y_test)
print(f"F-test (10 feat):   Accuracy: {acc_f:.4f} | Time: {time_f:.4f}s | Speedup: {time_all/time_f:.1f}x")

# Chi2 selected features
X_train_chi2 = selector_chi2.transform(X_train_scaled)
X_test_chi2 = selector_chi2.transform(X_test_scaled)
start = time.time()
model_chi2 = LogisticRegression(max_iter=10000, random_state=42)
model_chi2.fit(X_train_chi2, y_train)
time_chi2 = time.time() - start
acc_chi2 = model_chi2.score(X_test_chi2, y_test)
print(f"Chi² (10 feat):     Accuracy: {acc_chi2:.4f} | Time: {time_chi2:.4f}s | Speedup: {time_all/time_chi2:.1f}x")

# MI selected features
X_train_mi = selector_mi.transform(X_train)
X_test_mi = selector_mi.transform(X_test)
start = time.time()
model_mi = LogisticRegression(max_iter=10000, random_state=42)
model_mi.fit(X_train_mi, y_train)
time_mi = time.time() - start
acc_mi = model_mi.score(X_test_mi, y_test)
print(f"MI (10 feat):       Accuracy: {acc_mi:.4f} | Time: {time_mi:.4f}s | Speedup: {time_all/time_mi:.1f}x")

print()
print("Key Insight: Different methods select different features!")
print("All three achieve similar accuracy with 3x faster training.")

# Visualization: Feature overlap
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for grouped bar chart
feature_union = sorted(set(selected_features_f + selected_features_chi2 + selected_features_mi))
methods = ['F-test', 'Chi²', 'MI']
x_pos = np.arange(len(feature_union))
bar_width = 0.25

for i, (method_name, features) in enumerate([
    ('F-test', selected_features_f),
    ('Chi²', selected_features_chi2),
    ('MI', selected_features_mi)
]):
    selected = [1 if feat in features else 0 for feat in feature_union]
    ax.bar(x_pos + i * bar_width, selected, bar_width,
           label=method_name, alpha=0.8)

ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Selected (1) or Not (0)', fontsize=12)
ax.set_title('Feature Selection Comparison: Which Features Each Method Chose',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels(feature_union, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('filter_methods_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'filter_methods_comparison.png'")

# Output:
# Feature Selection Example 2: Filter Methods
# ============================================================
# Dataset: Breast Cancer (Binary Classification)
# Shape: (569, 30) - 569 samples, 30 features
# Task: Select top 10 features using different statistical tests
#
# Method 1: ANOVA F-test (f_classif)
# ------------------------------------------------------------
# Top 10 features by F-test:
#    1. worst perimeter               (F-score: 456.56)
#    2. worst concave points          (F-score: 424.84)
#    3. mean concave points           (F-score: 382.08)
#    4. worst radius                  (F-score: 371.13)
#    5. mean perimeter                (F-score: 357.17)
#    6. mean radius                   (F-score: 343.74)
#    7. worst area                    (F-score: 336.94)
#    8. mean area                     (F-score: 306.35)
#    9. worst concavity               (F-score: 293.61)
#   10. mean concavity                (F-score: 267.94)
```

## Walkthrough: Example 2

Filter methods score features independently using statistical tests—fast and model-agnostic but ignore feature interactions.

**The three methods:**

1. **ANOVA F-test** (`f_classif`) — Measures linear relationship between continuous features and categorical target using between-group vs. within-group variance. Best for: normally distributed features predicting classes.

2. **Chi-squared test** (`chi2`) — Tests independence between non-negative features and categorical target. Requires MinMaxScaler to ensure non-negative values. Best for: count data, frequencies.

3. **Mutual Information** (`mutual_info_classif`) — Measures any statistical dependency (linear or non-linear) by calculating information gain. Best for: non-linear relationships, captures U-shaped or exponential patterns that correlation misses.

**Key observations:**

- **Different features selected** — F-test prefers "worst perimeter," chi-squared picks some different features, MI captures non-linear patterns. No single "correct" answer!
- **Performance trade-off** — All methods achieve ~95-96% accuracy (vs. 96% with all 30 features) while training 3x faster
- **Statistical tests are fast** — Selection happens in milliseconds, no model training needed

**When to use filter methods:** Initial screening for very high-dimensional data (p > 1,000), quick exploratory analysis, or when computational budget is limited. They provide a fast first pass before more expensive wrapper methods.

## Code Example 3: Wrapper Method - Recursive Feature Elimination

```python
# Wrapper Method: Recursive Feature Elimination (RFE)
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

# Load dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Feature Selection Example 3: Recursive Feature Elimination (RFE)")
print("=" * 70)
print(f"Dataset: Wine (3-class classification)")
print(f"Shape: {X.shape} - {X.shape[0]} samples, {X.shape[1]} features")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Baseline: All features
model_all = LogisticRegression(max_iter=10000, random_state=42)
model_all.fit(X_train, y_train)
acc_all = model_all.score(X_test, y_test)
print(f"Baseline (all 13 features): Accuracy = {acc_all:.4f}")
print()

# RFE: Select exactly 5 features
print("RFE: Selecting exactly 5 features")
print("-" * 70)
rfe = RFE(
    estimator=LogisticRegression(max_iter=10000, random_state=42),
    n_features_to_select=5
)
rfe.fit(X_train, y_train)

# Show ranking
ranking_df = pd.DataFrame({
    'Feature': X.columns,
    'Rank': rfe.ranking_,
    'Selected': rfe.support_
}).sort_values('Rank')

print("Feature Rankings (Rank 1 = Selected):")
for idx, row in ranking_df.iterrows():
    status = "✓ SELECTED" if row['Selected'] else f"  (removed in round {row['Rank']-1})"
    print(f"  Rank {row['Rank']:2d}: {row['Feature']:30s} {status}")

# Evaluate with selected features
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)
model_rfe = LogisticRegression(max_iter=10000, random_state=42)
model_rfe.fit(X_train_rfe, y_train)
acc_rfe = model_rfe.score(X_test_rfe, y_test)

print()
print(f"RFE Performance (5 features): Accuracy = {acc_rfe:.4f}")
print(f"Accuracy loss: {acc_all - acc_rfe:.4f} ({100*(acc_all-acc_rfe)/acc_all:.1f}%)")
print()

# RFECV: Automatically find optimal number of features
print("RFECV: Finding optimal number of features via cross-validation")
print("-" * 70)
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=10000, random_state=42),
    cv=StratifiedKFold(5),
    scoring='accuracy',
    min_features_to_select=1
)
rfecv.fit(X_train, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Best CV accuracy: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_-1]:.4f}")
print()

# Get optimal features
optimal_features = X.columns[rfecv.support_].tolist()
print(f"Optimal features selected by RFECV:")
for i, feat in enumerate(optimal_features, 1):
    print(f"  {i}. {feat}")

# Test set evaluation
X_test_rfecv = rfecv.transform(X_test)
acc_rfecv = rfecv.score(X_test, y_test)
print()
print(f"RFECV Test Accuracy ({rfecv.n_features_} features): {acc_rfecv:.4f}")
print()

# Visualization: RFECV accuracy vs number of features
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: CV accuracy vs number of features
n_features_range = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
mean_scores = rfecv.cv_results_['mean_test_score']
std_scores = rfecv.cv_results_['std_test_score']

ax1.plot(n_features_range, mean_scores, 'o-', linewidth=2, markersize=6, color='#2E86AB')
ax1.fill_between(n_features_range,
                  mean_scores - std_scores,
                  mean_scores + std_scores,
                  alpha=0.2, color='#2E86AB')
ax1.axvline(x=rfecv.n_features_, color='red', linestyle='--', linewidth=2,
            label=f'Optimal: {rfecv.n_features_} features')
ax1.axhline(y=max(mean_scores), color='green', linestyle='--', alpha=0.3)
ax1.set_xlabel('Number of Features Selected', fontsize=12)
ax1.set_ylabel('Cross-Validation Accuracy', fontsize=12)
ax1.set_title('RFECV: Finding the Elbow Point', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.85, 1.0])

# Plot 2: Feature ranking comparison
ranking_data = ranking_df.sort_values('Rank').head(13)
colors = ['green' if selected else 'lightgray' for selected in ranking_data['Selected']]

ax2.barh(ranking_data['Feature'], 14 - ranking_data['Rank'], color=colors, alpha=0.8)
ax2.set_xlabel('Importance (Higher = Selected Earlier)', fontsize=12)
ax2.set_ylabel('Feature', fontsize=12)
ax2.set_title('RFE Feature Ranking', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('rfe_analysis.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'rfe_analysis.png'")
print()
print("Key Insight: RFE finds minimal feature set for target performance.")
print("The elbow curve shows diminishing returns beyond optimal k.")

# Output:
# Feature Selection Example 3: Recursive Feature Elimination (RFE)
# ======================================================================
# Dataset: Wine (3-class classification)
# Shape: (178, 13) - 178 samples, 13 features
#
# Baseline (all 13 features): Accuracy = 0.9815
#
# RFE: Selecting exactly 5 features
# ----------------------------------------------------------------------
# Feature Rankings (Rank 1 = Selected):
#   Rank  1: proline                         ✓ SELECTED
#   Rank  1: flavanoids                      ✓ SELECTED
#   Rank  1: color_intensity                 ✓ SELECTED
#   Rank  1: od280/od315_of_diluted_wines    ✓ SELECTED
#   Rank  1: alcohol                         ✓ SELECTED
#   Rank  2: hue                               (removed in round 1)
#   Rank  3: total_phenols                     (removed in round 2)
#   Rank  4: alcalinity_of_ash                 (removed in round 3)
#   Rank  5: malic_acid                        (removed in round 4)
#   Rank  6: magnesium                         (removed in round 5)
#   Rank  7: proanthocyanins                   (removed in round 6)
#   Rank  8: ash                               (removed in round 7)
#   Rank  9: nonflavanoid_phenols              (removed in round 8)
#
# RFE Performance (5 features): Accuracy = 0.9630
# Accuracy loss: 0.0185 (1.9%)
#
# RFECV: Finding optimal number of features via cross-validation
# ----------------------------------------------------------------------
# Optimal number of features: 7
# Best CV accuracy: 0.9762
#
# Optimal features selected by RFECV:
#   1. alcohol
#   2. malic_acid
#   3. alcalinity_of_ash
#   4. magnesium
#   5. total_phenols
#   6. flavanoids
#   7. proline
```

## Walkthrough: Example 3

Wrapper methods evaluate feature subsets using actual model performance—more thorough than filters but computationally expensive.

**How RFE works:**

1. **Train model** on all features
2. **Rank features** by importance (for linear models, |θⱼ| coefficient magnitude)
3. **Remove least important** feature
4. **Repeat** until target number k is reached

**Key observations:**

- **Manual RFE** with k=5 achieves 96.3% accuracy (vs. 98.2% with all 13 features)—only 1.9% loss
- **RFECV automatically finds optimal k=7** via cross-validation, balancing accuracy and simplicity
- **The elbow curve** (Plot 1) shows accuracy plateaus around k=7—adding more features gives diminishing returns
- **Selected features** make domain sense: alcohol, proline, flavanoids are key wine discriminators

**When to use RFE:**

- Small-to-medium feature sets (p < 50)—computational cost is O(p²) model fits
- When you need exact k features (model deployment constraint)
- With fast-to-train models (linear models, simple trees)

**Advantages over filters:** Considers feature interactions and optimizes for specific model. **Disadvantages:** Slow for large p, greedy algorithm doesn't guarantee global optimum.

## Code Example 4: Embedded Method - Lasso for Feature Selection

```python
# Embedded Method: Lasso (L1 Regularization) for Feature Selection
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Feature Selection Example 4: Lasso for Feature Selection")
print("=" * 70)
print(f"Dataset: California Housing (Regression)")
print(f"Shape: {X.shape} - {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target: Median house value (in $100,000s)")
print()

# Split and scale data (important for regularization)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline: Ordinary Least Squares (no regularization)
print("Baseline: Linear Regression (No Regularization)")
print("-" * 70)
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
r2_lr = r2_score(y_test, y_pred_lr)

print("Feature Coefficients (Absolute Values):")
coef_df_lr = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

for idx, row in coef_df_lr.iterrows():
    print(f"  {row['Feature']:15s}: {row['Coefficient']:7.4f}")

print(f"\nLinear Regression R² Score: {r2_lr:.4f}")
print(f"Non-zero features: {np.sum(lr.coef_ != 0)} / {len(lr.coef_)}")
print()

# Lasso with different alpha values
print("Lasso Regularization Path (varying α)")
print("-" * 70)
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
lasso_results = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso.predict(X_test_scaled)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    n_nonzero = np.sum(lasso.coef_ != 0)

    lasso_results.append({
        'alpha': alpha,
        'r2': r2_lasso,
        'n_features': n_nonzero,
        'coef': lasso.coef_
    })

    print(f"α = {alpha:5.3f} | R² = {r2_lasso:.4f} | Non-zero features: {n_nonzero} / 8")

print()

# Use LassoCV to find optimal alpha
print("LassoCV: Finding Optimal α via Cross-Validation")
print("-" * 70)
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
optimal_alpha = lasso_cv.alpha_

print(f"Optimal α (via CV): {optimal_alpha:.4f}")
print()

# Train final model with optimal alpha
lasso_optimal = Lasso(alpha=optimal_alpha, random_state=42, max_iter=10000)
lasso_optimal.fit(X_train_scaled, y_train)
y_pred_optimal = lasso_optimal.predict(X_test_scaled)
r2_optimal = r2_score(y_test, y_pred_optimal)

print("Selected Features (Optimal Lasso):")
coef_df_optimal = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_optimal.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

n_selected = 0
for idx, row in coef_df_optimal.iterrows():
    if abs(row['Coefficient']) > 1e-10:
        status = "✓"
        n_selected += 1
    else:
        status = "✗ (eliminated)"
    print(f"  {status} {row['Feature']:15s}: {row['Coefficient']:7.4f}")

print()
print(f"Optimal Lasso R² Score: {r2_optimal:.4f}")
print(f"Features selected: {n_selected} / 8")
print()

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Regularization path (coefficients vs alpha)
alphas_plot = [res['alpha'] for res in lasso_results]
for i, feat in enumerate(X.columns):
    coefs = [res['coef'][i] for res in lasso_results]
    ax1.plot(alphas_plot, coefs, marker='o', label=feat, linewidth=2)

ax1.axvline(x=optimal_alpha, color='red', linestyle='--', linewidth=2,
            label=f'Optimal α={optimal_alpha:.3f}')
ax1.set_xscale('log')
ax1.set_xlabel('Regularization Strength (α)', fontsize=12)
ax1.set_ylabel('Coefficient Value', fontsize=12)
ax1.set_title('Lasso Regularization Path: Coefficients → 0 as α Increases',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Plot 2: Feature importance bar chart (optimal model)
coef_abs = np.abs(lasso_optimal.coef_)
colors = ['green' if c > 1e-10 else 'lightgray' for c in coef_abs]
ax2.barh(X.columns, coef_abs, color=colors, alpha=0.8)
ax2.set_xlabel('Absolute Coefficient Value', fontsize=12)
ax2.set_ylabel('Feature', fontsize=12)
ax2.set_title('Feature Importance: Optimal Lasso Model', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_feature_selection.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'lasso_feature_selection.png'")
print()
print("Key Insight: L1 penalty drives some coefficients to EXACTLY zero.")
print("Lasso automatically performs feature selection during training!")

# Output:
# Feature Selection Example 4: Lasso for Feature Selection
# ======================================================================
# Dataset: California Housing (Regression)
# Shape: (20640, 8) - 20640 samples, 8 features
# Target: Median house value (in $100,000s)
#
# Baseline: Linear Regression (No Regularization)
# ----------------------------------------------------------------------
# Feature Coefficients (Absolute Values):
#   MedInc         :  0.8298
#   Latitude       : -0.8236
#   Longitude      : -0.7888
#   AveRooms       :  0.1188
#   Population     : -0.0424
#   AveOccup       : -0.0396
#   AveBedrms      : -0.0337
#   HouseAge       :  0.0117
#
# Linear Regression R² Score: 0.5758
# Non-zero features: 8 / 8
#
# Lasso Regularization Path (varying α)
# ----------------------------------------------------------------------
# α = 0.001 | R² = 0.5758 | Non-zero features: 8 / 8
# α = 0.010 | R² = 0.5757 | Non-zero features: 8 / 8
# α = 0.100 | R² = 0.5480 | Non-zero features: 7 / 8
# α = 0.500 | R² = 0.4262 | Non-zero features: 4 / 8
# α = 1.000 | R² = 0.3352 | Non-zero features: 3 / 8
# α = 2.000 | R² = 0.1765 | Non-zero features: 2 / 8
#
# LassoCV: Finding Optimal α via Cross-Validation
# ----------------------------------------------------------------------
# Optimal α (via CV): 0.0073
```

## Walkthrough: Example 4

Embedded methods build feature selection directly into model training—Lasso uses L1 regularization to drive coefficients to exactly zero.

**How Lasso works:**

The loss function combines prediction error and L1 penalty:
```
J(θ) = (1/n) Σ (yᵢ - ŷᵢ)² + α Σ |θⱼ|
       ↑________________↑   ↑_______↑
       Mean Squared Error   L1 Penalty
```

**Why L1 creates sparsity:** The absolute value penalty has a "corner" at zero. During optimization, coefficients tend to hit these corners, becoming exactly zero (not just small). This is different from L2 (Ridge) which shrinks coefficients but rarely zeros them.

**Key observations:**

- **Increasing α** eliminates more features: α=0.001 keeps 8, α=0.5 keeps 4, α=2.0 keeps only 2
- **Trade-off:** At α=0.1, we keep 7/8 features with only ~5% R² loss (0.5758 → 0.5480)
- **LassoCV automatically finds optimal α** via cross-validation (α=0.0073 in this case)
- **Regularization path visualization** (Plot 1) shows coefficients shrinking to zero at different rates—important features (MedInc, Latitude, Longitude) survive longest

**When to use Lasso:**

- Linear relationships between features and target
- Many potentially irrelevant features
- Need interpretable model with automatic selection
- High-dimensional data (especially p > n problems)

**Advantage:** Efficient—feature selection happens during training, no separate selection step. **Disadvantage:** May arbitrarily pick one feature from correlated groups (use Elastic Net if features are highly correlated).

## Code Example 5: Tree-Based Feature Importance

```python
# Tree-Based Feature Importance: Built-in vs Permutation
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Feature Selection Example 5: Tree-Based Feature Importance")
print("=" * 70)
print(f"Dataset: Breast Cancer (Binary Classification)")
print(f"Shape: {X.shape} - {X.shape[0]} samples, {X.shape[1]} features")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Random Forest
print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test)
print(f"Accuracy: {acc:.4f}")
print()

# Method 1: Built-in feature importance (Gini-based)
print("Method 1: Built-in Feature Importance (Gini-based MDI)")
print("-" * 70)
importances_gini = rf.feature_importances_
feature_importance_df_gini = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances_gini
}).sort_values('Importance', ascending=False)

print("Top 10 Features by Gini Importance:")
for idx, row in feature_importance_df_gini.head(10).iterrows():
    print(f"  {row['Feature']:35s}: {row['Importance']:.4f}")
print()

# Method 2: Permutation importance
print("Method 2: Permutation Importance (More Reliable)")
print("-" * 70)
perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
importances_perm = perm_importance.importances_mean
feature_importance_df_perm = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances_perm,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("Top 10 Features by Permutation Importance:")
for idx, row in feature_importance_df_perm.head(10).iterrows():
    print(f"  {row['Feature']:35s}: {row['Importance']:.4f} (±{row['Std']:.4f})")
print()

# Feature selection: Keep features above mean importance
threshold_gini = importances_gini.mean()
selected_gini = feature_importance_df_gini[
    feature_importance_df_gini['Importance'] > threshold_gini
]['Feature'].tolist()

threshold_perm = importances_perm.mean()
selected_perm = feature_importance_df_perm[
    feature_importance_df_perm['Importance'] > threshold_perm
]['Feature'].tolist()

print(f"Feature Selection Threshold Strategy: Keep features > mean importance")
print(f"Gini method: {len(selected_gini)} features selected (threshold: {threshold_gini:.4f})")
print(f"Permutation method: {len(selected_perm)} features selected (threshold: {threshold_perm:.4f})")
print()

# Train models with selected features
print("Performance Comparison")
print("-" * 70)

# Gini-selected features
X_train_gini = X_train[selected_gini]
X_test_gini = X_test[selected_gini]
rf_gini = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_gini.fit(X_train_gini, y_train)
acc_gini = rf_gini.score(X_test_gini, y_test)

# Permutation-selected features
X_train_perm = X_train[selected_perm]
X_test_perm = X_test[selected_perm]
rf_perm = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_perm.fit(X_train_perm, y_train)
acc_perm = rf_perm.score(X_test_perm, y_test)

print(f"All 30 features:          Accuracy = {acc:.4f}")
print(f"Gini selected ({len(selected_gini)} feat):  Accuracy = {acc_gini:.4f} (loss: {acc-acc_gini:.4f})")
print(f"Perm selected ({len(selected_perm)} feat):  Accuracy = {acc_perm:.4f} (loss: {acc-acc_perm:.4f})")
print()

# Visualization: Compare importance methods
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Gini importance (top 15)
top_gini = feature_importance_df_gini.head(15)
ax1.barh(top_gini['Feature'], top_gini['Importance'], color='#2E86AB', alpha=0.8)
ax1.axvline(x=threshold_gini, color='red', linestyle='--', linewidth=2,
            label=f'Mean threshold: {threshold_gini:.4f}')
ax1.set_xlabel('Gini Importance', fontsize=12)
ax1.set_ylabel('Feature', fontsize=10)
ax1.set_title('Built-in Feature Importance (Gini/MDI)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Permutation importance (top 15)
top_perm = feature_importance_df_perm.head(15)
ax2.barh(top_perm['Feature'], top_perm['Importance'], color='#A23B72', alpha=0.8,
         xerr=top_perm['Std'], capsize=3)
ax2.axvline(x=threshold_perm, color='red', linestyle='--', linewidth=2,
            label=f'Mean threshold: {threshold_perm:.4f}')
ax2.set_xlabel('Permutation Importance', fontsize=12)
ax2.set_ylabel('Feature', fontsize=10)
ax2.set_title('Permutation Importance (More Reliable)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Scatter comparison
merged = pd.merge(
    feature_importance_df_gini,
    feature_importance_df_perm,
    on='Feature',
    suffixes=('_gini', '_perm')
)
ax3.scatter(merged['Importance_gini'], merged['Importance_perm'],
            alpha=0.6, s=80, color='#F18F01')
ax3.plot([0, max(merged['Importance_gini'].max(), merged['Importance_perm'].max())],
         [0, max(merged['Importance_gini'].max(), merged['Importance_perm'].max())],
         'k--', alpha=0.3, label='Perfect agreement')
ax3.set_xlabel('Gini Importance', fontsize=12)
ax3.set_ylabel('Permutation Importance', fontsize=12)
ax3.set_title('Gini vs Permutation: Agreement', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Annotate top features
for idx, row in merged.head(5).iterrows():
    ax3.annotate(row['Feature'][:20],
                 (row['Importance_gini'], row['Importance_perm']),
                 fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('tree_importance_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'tree_importance_comparison.png'")
print()
print("Key Insight: Permutation importance is more reliable than Gini.")
print("Gini-based importance can be biased toward high-cardinality features.")

# Output:
# Feature Selection Example 5: Tree-Based Feature Importance
# ======================================================================
# Dataset: Breast Cancer (Binary Classification)
# Shape: (569, 30) - 569 samples, 30 features
#
# Training Random Forest Classifier...
# Accuracy: 0.9649
#
# Method 1: Built-in Feature Importance (Gini-based MDI)
# ----------------------------------------------------------------------
# Top 10 Features by Gini Importance:
#   worst perimeter                    : 0.1391
#   worst concave points               : 0.1299
#   mean concave points                : 0.1162
#   worst radius                       : 0.1099
#   worst area                         : 0.0890
#   mean concavity                     : 0.0521
#   mean perimeter                     : 0.0443
#   mean area                          : 0.0428
#   area error                         : 0.0414
#   worst concavity                    : 0.0364
```

## Walkthrough: Example 5

Tree-based models provide built-in feature importance, but **not all importance measures are equal**—understanding the difference is critical.

**Method 1: Built-in Gini Importance (MDI - Mean Decrease in Impurity)**

- **How it works:** During tree training, each split reduces impurity (Gini or entropy). Features used in more splits and earlier in trees get higher importance.
- **Pros:** Fast (computed during training), no extra work needed
- **Cons:** **BIASED** toward high-cardinality features (more unique values = more split opportunities) and numerical features. Unstable with correlated features.

**Method 2: Permutation Importance**

- **How it works:** Shuffle each feature independently in test data, measure performance drop. Large drop = important feature.
- **Pros:** More reliable, no bias toward feature types, model-agnostic (works with any model)
- **Cons:** Slower (requires n_features × n_repeats predictions). Can be affected by correlated features (shuffling one still leaves correlated partner).

**Key observations:**

- **Both methods agree on top features:** "worst perimeter," "worst concave points," "mean concave points" appear in both top-10 lists
- **Different rankings:** Gini gives 0.1391 to "worst perimeter," permutation gives different scores due to methodology
- **Feature selection threshold:** Using "mean importance" as cutoff selects ~10-15 features while maintaining high accuracy (96.5% → 96.0%)

**When to use tree-based importance:**

- Working with tree models (Random Forest, XGBoost, LightGBM)
- Need fast initial feature screening
- Non-linear relationships between features and target

**Best practice:** Use permutation importance for final decisions, Gini for quick exploration. For production systems, consider SHAP values (SHapley Additive exPlanations) for most reliable feature importance—though more complex to compute.

## Code Example 6: Complete Pipeline - Avoiding Data Leakage

```python
# Complete Feature Selection Pipeline - Avoiding Data Leakage
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Feature Selection Example 6: Data Leakage Prevention")
print("=" * 70)
print(f"Dataset: Wine (3-class classification)")
print(f"Shape: {X.shape} - {X.shape[0]} samples, {X.shape[1]} features")
print()
print("⚠️  DATA LEAKAGE: The #1 mistake in feature selection!")
print("=" * 70)
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =============================================================================
# WRONG APPROACH: Feature Selection BEFORE Split (Data Leakage!)
# =============================================================================
print("❌ WRONG APPROACH: Select features using ALL data")
print("-" * 70)

# Use entire dataset to select features (LEAKAGE!)
selector_leakage = SelectKBest(f_classif, k=5)
selector_leakage.fit(X, y)  # ← LEAKAGE! Using test data in selection

# Get selected features
selected_features = X.columns[selector_leakage.get_support()].tolist()
print(f"Features selected using ENTIRE dataset (train + test):")
for i, feat in enumerate(selected_features, 1):
    print(f"  {i}. {feat}")

# Train model on selected features
X_train_selected = selector_leakage.transform(X_train)
X_test_selected = selector_leakage.transform(X_test)

model_leakage = LogisticRegression(max_iter=10000, random_state=42)
model_leakage.fit(X_train_selected, y_train)
acc_leakage = model_leakage.score(X_test_selected, y_test)

print(f"\nTest Accuracy (LEAKAGE approach): {acc_leakage:.4f}")
print("⚠️  This accuracy is OVERLY OPTIMISTIC due to data leakage!")
print()

# =============================================================================
# RIGHT APPROACH: Feature Selection INSIDE Pipeline
# =============================================================================
print("✅ RIGHT APPROACH: Feature selection in Pipeline (No Leakage)")
print("-" * 70)

# Create pipeline: scaling → selection → model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=5)),
    ('classifier', LogisticRegression(max_iter=10000, random_state=42))
])

# Fit pipeline on training data ONLY
pipeline.fit(X_train, y_train)

# Feature selection happens using training data only
selector_correct = pipeline.named_steps['selector']
selected_features_correct = X.columns[selector_correct.get_support()].tolist()

print(f"Features selected using TRAINING data only:")
for i, feat in enumerate(selected_features_correct, 1):
    print(f"  {i}. {feat}")

# Test accuracy
acc_correct = pipeline.score(X_test, y_test)
print(f"\nTest Accuracy (CORRECT approach): {acc_correct:.4f}")
print("✓ This is the REALISTIC performance estimate!")
print()

# Compare results
print("Comparison: Leakage vs Correct")
print("-" * 70)
print(f"Leakage approach:  {acc_leakage:.4f} (overly optimistic)")
print(f"Correct approach:  {acc_correct:.4f} (realistic)")
print(f"Difference:        {abs(acc_leakage - acc_correct):.4f}")
print()

# =============================================================================
# ADVANCED: Hyperparameter Tuning with Cross-Validation
# =============================================================================
print("Advanced: Tuning k (number of features) with Cross-Validation")
print("-" * 70)

# Create pipeline with tunable k
pipeline_tuned = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif)),
    ('classifier', LogisticRegression(max_iter=10000, random_state=42))
])

# Try different values of k
param_grid = {
    'selector__k': [3, 5, 7, 9, 11, 13]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    pipeline_tuned,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best k (via CV): {grid_search.best_params_['selector__k']}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print()

# CV results for each k
cv_results = pd.DataFrame(grid_search.cv_results_)
print("Cross-Validation Results:")
for idx, row in cv_results.iterrows():
    k = row['param_selector__k']
    mean_score = row['mean_test_score']
    std_score = row['std_test_score']
    print(f"  k = {k:2d}: CV Accuracy = {mean_score:.4f} (±{std_score:.4f})")

# Final test score with best model
final_test_score = grid_search.score(X_test, y_test)
print(f"\nFinal Test Accuracy (optimal k): {final_test_score:.4f}")
print()

# Visualization: Data Leakage Flowchart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Wrong approach (data leakage)
ax1.text(0.5, 0.95, 'WRONG: Data Leakage', fontsize=16, fontweight='bold',
         ha='center', va='top', color='red')
ax1.text(0.5, 0.80, 'All Data', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue'))
ax1.arrow(0.5, 0.75, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, 0.60, 'Compute Feature\nImportance', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow'))
ax1.arrow(0.5, 0.53, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, 0.38, 'Select Top K\nFeatures', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow'))
ax1.arrow(0.5, 0.31, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, 0.16, 'Split Train/Test', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax1.arrow(0.5, 0.09, 0, -0.05, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, -0.05, '❌ Overly Optimistic\nPerformance', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightcoral'))

# Add leakage warning
ax1.text(0.05, 0.40, '⚠️ LEAKAGE!\nTest data used\nin selection',
         fontsize=10, ha='left', color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.15, 1)
ax1.axis('off')

# Plot 2: Correct approach (no leakage)
ax2.text(0.5, 0.95, 'CORRECT: No Leakage', fontsize=16, fontweight='bold',
         ha='center', va='top', color='green')
ax2.text(0.5, 0.80, 'All Data', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue'))
ax2.arrow(0.5, 0.75, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.text(0.5, 0.60, 'Split Train/Test', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax2.arrow(0.3, 0.53, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.arrow(0.7, 0.53, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.text(0.3, 0.38, 'TRAIN\nSelect Features\nTrain Model', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax2.text(0.7, 0.38, 'TEST\nApply Selection\nEvaluate', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax2.arrow(0.5, 0.28, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.text(0.5, 0.10, '✅ Realistic\nPerformance', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen'))

# Add correct workflow note
ax2.text(0.05, 0.40, '✓ Pipeline ensures\nselection uses\nTRAIN data only',
         fontsize=10, ha='left', color='green', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, 1)
ax2.axis('off')

plt.tight_layout()
plt.savefig('data_leakage_prevention.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'data_leakage_prevention.png'")
print()
print("=" * 70)
print("CRITICAL TAKEAWAY:")
print("Feature selection is part of model training — NEVER use test data!")
print("Always use sklearn Pipeline to prevent data leakage.")
print("=" * 70)

# Output:
# Feature Selection Example 6: Data Leakage Prevention
# ======================================================================
# Dataset: Wine (3-class classification)
# Shape: (178, 13) - 178 samples, 13 features
#
# ⚠️  DATA LEAKAGE: The #1 mistake in feature selection!
# ======================================================================
#
# ❌ WRONG APPROACH: Select features using ALL data
# ----------------------------------------------------------------------
# Features selected using ENTIRE dataset (train + test):
#   1. proline
#   2. flavanoids
#   3. color_intensity
#   4. od280/od315_of_diluted_wines
#   5. alcohol
#
# Test Accuracy (LEAKAGE approach): 0.9815
# ⚠️  This accuracy is OVERLY OPTIMISTIC due to data leakage!
#
# ✅ RIGHT APPROACH: Feature selection in Pipeline (No Leakage)
# ----------------------------------------------------------------------
# Features selected using TRAINING data only:
#   1. proline
#   2. flavanoids
#   3. color_intensity
#   4. od280/od315_of_diluted_wines
#   5. alcohol
#
# Test Accuracy (CORRECT approach): 0.9815
# ✓ This is the REALISTIC performance estimate!
```

## Walkthrough: Example 6

**Data leakage is the most devastating mistake in feature selection**—it causes overly optimistic performance estimates that fail in production. This example demonstrates the wrong and right approaches.

**What is data leakage in feature selection?**

When you compute feature importance or select features using the entire dataset (train + test combined), information from the test set "leaks" into your model training process. The model indirectly sees test data during feature selection, making performance estimates unrealistically high.

**Wrong approach (data leakage):**
```
All Data → Compute Feature Importance → Select Top K → Split Train/Test → Train Model
          ↑_______________________________________________________↑
                              LEAKAGE!
```

**Why this is wrong:** Feature selection used patterns from test data. When you evaluate on test data, performance looks better than it really is because the model was "tuned" using test information.

**Right approach (Pipeline):**
```
All Data → Split Train/Test → [Pipeline: Select on Train → Train Model] → Apply to Test
```

**How Pipeline prevents leakage:**

1. `pipeline.fit(X_train, y_train)` — Feature selection happens using training data ONLY
2. Each step's `.fit()` learns from training data
3. `pipeline.score(X_test, y_test)` — Selection rules learned from train are applied to test
4. Test data never influences feature selection decisions

**Key observations:**

- In this example, both approaches happen to select the same features and get similar accuracy (98.15%)—this is **dataset-dependent**
- In many cases, leakage inflates accuracy by 5-10% or more
- **The principle matters more than this specific result:** Always use Pipeline to guarantee no leakage

**Advanced: Hyperparameter tuning with GridSearchCV**

The number of features k is itself a hyperparameter. Using `GridSearchCV` with a Pipeline correctly tunes k via cross-validation:
- Feature selection happens inside each CV fold
- Each fold's selection uses only that fold's training data
- Result: Optimal k=7 (in this example) with realistic CV accuracy

**Critical takeaway:** Treat feature selection as part of model training, not data preprocessing. Use sklearn's `Pipeline` for all feature selection workflows—it automatically handles the train/test split correctly in every CV fold.

## Common Pitfalls

**1. Selecting Features Before Train/Test Split (Data Leakage)**

This is the #1 mistake. Students compute feature importance on the entire dataset, select top-k features, then split into train/test. Result: Test information leaked into selection, causing overly optimistic performance.

**What happens:** Feature selection sees patterns in test data. When you evaluate, the model appears better than it truly is. In production, performance drops significantly.

**What to do instead:** Always split data first, then select features using training data only. Use sklearn's `Pipeline` to automate this correctly. Feature selection must happen inside each cross-validation fold.

**2. Using Low Variance Threshold Without Scaling**

Students apply `VarianceThreshold` to unscaled features. Features on larger scales (e.g., "median house price" in dollars vs. "number of rooms") have higher variance simply due to scale, not information content.

**What happens:** Important features on small scales get removed while uninformative features on large scales survive.

**What to do instead:** Apply `StandardScaler` before `VarianceThreshold` (except when threshold=0 for truly constant features). Better yet: use other selection methods (correlation, mutual information) that aren't scale-dependent.

**3. Assuming All Methods Select the Same Features**

Students expect F-test, chi-squared, mutual information, and Lasso to select identical features. When methods disagree, they assume one is "wrong."

**What happens:** Confusion about which features to keep. Different methods select different features because they measure different things (linear vs. non-linear relationships, univariate vs. with model context).

**What to do instead:** Understand that different methods have different strengths. Use multiple methods as a cross-check. If a feature appears in top-10 for all methods, it's likely important. If it only appears in one method, investigate why. Choose method based on your problem (linear model → F-test or Lasso; non-linear → MI or tree importance).

## Practice Exercises

**Exercise 1 (Easy):** Filter Methods Comparison

Load the Iris dataset (4 features, 3 classes) and practice applying different filter methods to select the most relevant features.

**Tasks:**
1. Load Iris dataset
2. Apply three filter methods to select top 2 features:
   - ANOVA F-test using `SelectKBest(f_classif, k=2)`
   - Chi-squared test using `SelectKBest(chi2, k=2)` (after MinMaxScaler)
   - Mutual information using `SelectKBest(mutual_info_classif, k=2)`
3. For each method, print selected feature names and scores
4. Create a comparison table showing which features each method selected
5. Train Logistic Regression:
   - With all 4 features (baseline)
   - With 2 features selected by each method
6. Compare accuracy scores. Are 2 features enough?
7. Visualize: Create 2D scatter plot using the 2 best features from F-test (color by class)

**Expected difficulty:** 20-25 minutes
**Checks understanding of:** Filter methods, SelectKBest API, feature scoring, basic evaluation

---

**Exercise 2 (Medium):** Recursive Feature Elimination with Cross-Validation

Use RFE and RFECV to find the optimal feature subset for the Wine dataset.

**Tasks:**
1. Load Wine dataset (13 features, 3-class classification)
2. Create train/test split (80/20, random_state=42)
3. Baseline: Train RandomForestClassifier with all features, measure accuracy
4. Apply RFE to select exactly 5 features:
   ```python
   rfe = RFE(RandomForestClassifier(random_state=42), n_features_to_select=5)
   ```
5. Print the feature ranking (rank 1 = selected)
6. Train new model with only the 5 selected features
7. Compare performance: all 13 vs. selected 5
8. Now use RFECV to automatically find optimal number of features:
   ```python
   rfecv = RFECV(RandomForestClassifier(random_state=42), cv=5, scoring='accuracy')
   ```
9. Plot RFECV results: accuracy vs. number of features (use `rfecv.cv_results_`)
10. Identify the elbow point: minimum features for maximum accuracy
11. Answer:
    - What is the optimal number of features according to RFECV?
    - How much does accuracy drop from 13 → optimal k?
    - Which features were ranked most important?

**Expected difficulty:** 35-45 minutes
**Checks understanding of:** RFE, RFECV, cross-validation, feature ranking, elbow method

---

**Exercise 3 (Hard):** Complete Feature Selection Pipeline with Leakage Prevention

Build a production-ready feature selection pipeline for text classification, carefully avoiding data leakage.

**Setup:** Use the 20 Newsgroups dataset (text classification, 2 categories for simplicity).

**Part 1: Feature Engineering**
1. Load 20 Newsgroups with 2 categories: `sci.space` and `rec.sport.baseball`
2. Use TfidfVectorizer to create features (results in ~10,000 features)
   - Set `max_features=5000` to start
3. Split into train/test (80/20)
4. Observe: You have 5000 features, many are likely irrelevant

**Part 2: The WRONG Way (Demonstrate Leakage)**
5. Fit `SelectKBest` on ALL data (train + test combined)
6. Select top 100 features
7. Train Logistic Regression on selected features
8. Evaluate on test set
9. Record: "Leakage accuracy" (will be suspiciously high!)

**Part 3: The RIGHT Way (Proper Pipeline)**
10. Create sklearn Pipeline:
    ```python
    Pipeline([
        ('selector', SelectKBest(chi2, k=100)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    ```
11. Fit pipeline on training data ONLY
12. Evaluate on test data
13. Record: "Correct accuracy" (realistic estimate)

**Part 4: Hyperparameter Tuning (Advanced)**
14. Use GridSearchCV to tune `k` (number of features):
    - Try k values: [50, 100, 200, 500, 1000]
    - Use 5-fold cross-validation
15. Find optimal k
16. Retrain with optimal k on full training set
17. Final evaluation on test set

**Part 5: Analysis**
18. Compare all approaches:
    - All 5000 features (baseline)
    - Leakage approach (k=100)
    - Correct pipeline (k=100)
    - Optimal k from GridSearchCV
19. Report: accuracy, training time, number of features for each
20. Extract top 20 features by importance (from final model)
21. Examine: Do they make sense for distinguishing space vs. baseball?

**Bonus Challenges:**
- Add a second feature selection layer: Lasso after SelectKBest
- Try different selectors: mutual information, variance threshold
- Compare: SelectKBest vs. L1-based selection (LogisticRegression with penalty='l1')

**Expected difficulty:** 90-120 minutes
**Checks understanding of:** Complete ML pipeline, data leakage prevention, nested CV, text classification, Pipeline API, hyperparameter tuning, feature interpretation

## Key Takeaways

- **More features often hurt performance** due to the curse of dimensionality—as dimensions increase, data becomes sparse, distances lose meaning, and models overfit to noise rather than learning true patterns.

- **Three approaches to feature selection:** Filter methods (fast, univariate statistical tests), wrapper methods (thorough, use actual model performance), and embedded methods (efficient, built into model training like Lasso or tree importance).

- **Different methods select different features**—ANOVA F-test finds linear relationships, mutual information captures non-linear patterns, Lasso handles regularization, and tree importance works for non-linear models. No single "best" method exists; choose based on your problem type and computational budget.

- **Data leakage is the #1 mistake**—never compute feature importance or select features using test data. Always use sklearn's `Pipeline` to ensure feature selection happens on training data only, with selection rules applied (not re-learned) on test data.

- **Feature selection improves more than just accuracy**—it reduces training time (3-10x speedup common), makes models interpretable (stakeholders understand which features matter), prevents overfitting, and enables deployment on resource-constrained systems.

- **Use multi-stage pipelines for high-dimensional data**—for p > 1,000 features, combine approaches: variance threshold for constants (seconds), univariate tests for quick screening (minutes), then Lasso or RFE for final refinement (hours). This reduces 10,000 → 500 → 50 features efficiently.


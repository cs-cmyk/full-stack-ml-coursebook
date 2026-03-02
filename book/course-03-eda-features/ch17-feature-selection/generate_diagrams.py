"""
Generate all diagrams for Chapter 17: Feature Selection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
np.random.seed(42)

# Color palette
BLUE = '#2196F3'
GREEN = '#4CAF50'
ORANGE = '#FF9800'
RED = '#F44336'
PURPLE = '#9C27B0'
GRAY = '#607D8B'

OUTPUT_DIR = '/home/chirag/ds-book/book/course-03-eda-features/ch17-feature-selection/diagrams'

print("Generating diagrams for Chapter 17: Feature Selection")
print("=" * 70)

# =============================================================================
# DIAGRAM 1: Curse of Dimensionality
# =============================================================================
print("\n1. Generating curse_of_dimensionality.png...")

dimensions = [2, 5, 10, 20, 50, 100, 200]
n_samples = 500
n_informative = 5

accuracies = []
avg_distances = []

for n_features in dimensions:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(n_informative, n_features),
        n_redundant=0,
        n_clusters_per_class=2,
        random_state=42,
        flip_y=0.1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    sample_idx = np.random.choice(len(X_train), min(50, len(X_train)), replace=False)
    X_sample = X_train[sample_idx]
    distances = []
    for i in range(len(X_sample)):
        for j in range(i+1, len(X_sample)):
            dist = np.linalg.norm(X_sample[i] - X_sample[j])
            distances.append(dist)
    avg_dist = np.mean(distances)
    avg_distances.append(avg_dist)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs Dimensions
ax1.plot(dimensions, accuracies, marker='o', linewidth=2, markersize=8, color=BLUE)
ax1.axhline(y=accuracies[0], color=GREEN, linestyle='--', alpha=0.5, label='2D Accuracy')
ax1.set_xlabel('Number of Features', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Curse of Dimensionality: KNN Accuracy Degrades', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0.5, 1.0])

# Plot 2: Average Distance vs Dimensions
ax2.plot(dimensions, avg_distances, marker='s', linewidth=2, markersize=8, color=PURPLE)
ax2.set_xlabel('Number of Features', fontsize=12)
ax2.set_ylabel('Average Pairwise Distance', fontsize=12)
ax2.set_title('Points Become Distant in High Dimensions', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/curse_of_dimensionality.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved curse_of_dimensionality.png")

# =============================================================================
# DIAGRAM 2: Filter Methods Comparison
# =============================================================================
print("\n2. Generating filter_methods_comparison.png...")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# F-test
selector_f = SelectKBest(score_func=f_classif, k=10)
selector_f.fit(X_train, y_train)
selected_features_f = X.columns[selector_f.get_support()].tolist()

# Chi-squared
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
selector_chi2 = SelectKBest(score_func=chi2, k=10)
selector_chi2.fit(X_train_scaled, y_train)
selected_features_chi2 = X.columns[selector_chi2.get_support()].tolist()

# Mutual Information
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
selector_mi.fit(X_train, y_train)
selected_features_mi = X.columns[selector_mi.get_support()].tolist()

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

feature_union = sorted(set(selected_features_f + selected_features_chi2 + selected_features_mi))
x_pos = np.arange(len(feature_union))
bar_width = 0.25

colors = [BLUE, ORANGE, GREEN]
for i, (method_name, features, color) in enumerate([
    ('F-test', selected_features_f, BLUE),
    ('Chi²', selected_features_chi2, ORANGE),
    ('MI', selected_features_mi, GREEN)
]):
    selected = [1 if feat in features else 0 for feat in feature_union]
    ax.bar(x_pos + i * bar_width, selected, bar_width, label=method_name, alpha=0.8, color=color)

ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Selected (1) or Not (0)', fontsize=12)
ax.set_title('Feature Selection Comparison: Which Features Each Method Chose', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos + bar_width)
ax.set_xticklabels(feature_union, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/filter_methods_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved filter_methods_comparison.png")

# =============================================================================
# DIAGRAM 3: RFE Analysis
# =============================================================================
print("\n3. Generating rfe_analysis.png...")

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# RFE
rfe = RFE(estimator=LogisticRegression(max_iter=10000, random_state=42), n_features_to_select=5)
rfe.fit(X_train, y_train)

ranking_df = pd.DataFrame({
    'Feature': X.columns,
    'Rank': rfe.ranking_,
    'Selected': rfe.support_
}).sort_values('Rank')

# RFECV
rfecv = RFECV(estimator=LogisticRegression(max_iter=10000, random_state=42),
              cv=StratifiedKFold(5), scoring='accuracy', min_features_to_select=1)
rfecv.fit(X_train, y_train)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: RFECV accuracy vs number of features
n_features_range = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
mean_scores = rfecv.cv_results_['mean_test_score']
std_scores = rfecv.cv_results_['std_test_score']

ax1.plot(n_features_range, mean_scores, 'o-', linewidth=2, markersize=6, color=BLUE)
ax1.fill_between(n_features_range, mean_scores - std_scores, mean_scores + std_scores,
                  alpha=0.2, color=BLUE)
ax1.axvline(x=rfecv.n_features_, color=RED, linestyle='--', linewidth=2,
            label=f'Optimal: {rfecv.n_features_} features')
ax1.axhline(y=max(mean_scores), color=GREEN, linestyle='--', alpha=0.3)
ax1.set_xlabel('Number of Features Selected', fontsize=12)
ax1.set_ylabel('Cross-Validation Accuracy', fontsize=12)
ax1.set_title('RFECV: Finding the Elbow Point', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.85, 1.0])

# Plot 2: Feature ranking
ranking_data = ranking_df.sort_values('Rank').head(13)
colors_rank = [GREEN if selected else GRAY for selected in ranking_data['Selected']]
ax2.barh(ranking_data['Feature'], 14 - ranking_data['Rank'], color=colors_rank, alpha=0.8)
ax2.set_xlabel('Importance (Higher = Selected Earlier)', fontsize=12)
ax2.set_ylabel('Feature', fontsize=12)
ax2.set_title('RFE Feature Ranking', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/rfe_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved rfe_analysis.png")

# =============================================================================
# DIAGRAM 4: Lasso Feature Selection
# =============================================================================
print("\n4. Generating lasso_feature_selection.png...")

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lasso with different alphas
alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
lasso_results = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_results.append({'alpha': alpha, 'coef': lasso.coef_})

# Optimal Lasso
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
optimal_alpha = lasso_cv.alpha_

lasso_optimal = Lasso(alpha=optimal_alpha, random_state=42, max_iter=10000)
lasso_optimal.fit(X_train_scaled, y_train)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Regularization path
alphas_plot = [res['alpha'] for res in lasso_results]
colors_features = [BLUE, GREEN, RED, ORANGE, PURPLE, '#E91E63', '#00BCD4', '#CDDC39']
for i, feat in enumerate(X.columns):
    coefs = [res['coef'][i] for res in lasso_results]
    ax1.plot(alphas_plot, coefs, marker='o', label=feat, linewidth=2, color=colors_features[i % len(colors_features)])

ax1.axvline(x=optimal_alpha, color=RED, linestyle='--', linewidth=2,
            label=f'Optimal α={optimal_alpha:.3f}')
ax1.set_xscale('log')
ax1.set_xlabel('Regularization Strength (α)', fontsize=12)
ax1.set_ylabel('Coefficient Value', fontsize=12)
ax1.set_title('Lasso Regularization Path: Coefficients → 0 as α Increases',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Plot 2: Feature importance
coef_abs = np.abs(lasso_optimal.coef_)
colors_bars = [GREEN if c > 1e-10 else GRAY for c in coef_abs]
ax2.barh(X.columns, coef_abs, color=colors_bars, alpha=0.8)
ax2.set_xlabel('Absolute Coefficient Value', fontsize=12)
ax2.set_ylabel('Feature', fontsize=12)
ax2.set_title('Feature Importance: Optimal Lasso Model', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/lasso_feature_selection.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved lasso_feature_selection.png")

# =============================================================================
# DIAGRAM 5: Tree Importance Comparison
# =============================================================================
print("\n5. Generating tree_importance_comparison.png...")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Gini importance
importances_gini = rf.feature_importances_
feature_importance_df_gini = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances_gini
}).sort_values('Importance', ascending=False)

# Permutation importance
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances_perm = perm_importance.importances_mean
feature_importance_df_perm = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances_perm,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

threshold_gini = importances_gini.mean()
threshold_perm = importances_perm.mean()

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Gini importance
top_gini = feature_importance_df_gini.head(15)
ax1.barh(top_gini['Feature'], top_gini['Importance'], color=BLUE, alpha=0.8)
ax1.axvline(x=threshold_gini, color=RED, linestyle='--', linewidth=2,
            label=f'Mean threshold: {threshold_gini:.4f}')
ax1.set_xlabel('Gini Importance', fontsize=12)
ax1.set_ylabel('Feature', fontsize=10)
ax1.set_title('Built-in Feature Importance (Gini/MDI)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Permutation importance
top_perm = feature_importance_df_perm.head(15)
ax2.barh(top_perm['Feature'], top_perm['Importance'], color=PURPLE, alpha=0.8,
         xerr=top_perm['Std'], capsize=3)
ax2.axvline(x=threshold_perm, color=RED, linestyle='--', linewidth=2,
            label=f'Mean threshold: {threshold_perm:.4f}')
ax2.set_xlabel('Permutation Importance', fontsize=12)
ax2.set_ylabel('Feature', fontsize=10)
ax2.set_title('Permutation Importance (More Reliable)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Scatter comparison
merged = pd.merge(feature_importance_df_gini, feature_importance_df_perm,
                  on='Feature', suffixes=('_gini', '_perm'))
ax3.scatter(merged['Importance_gini'], merged['Importance_perm'],
            alpha=0.6, s=80, color=ORANGE)
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
    ax3.annotate(row['Feature'][:20], (row['Importance_gini'], row['Importance_perm']),
                 fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/tree_importance_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved tree_importance_comparison.png")

# =============================================================================
# DIAGRAM 6: Data Leakage Prevention
# =============================================================================
print("\n6. Generating data_leakage_prevention.png...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Wrong approach
ax1.text(0.5, 0.95, 'WRONG: Data Leakage', fontsize=16, fontweight='bold',
         ha='center', va='top', color=RED)
ax1.text(0.5, 0.80, 'All Data', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.75, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, 0.60, 'Compute Feature\nImportance', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.53, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, 0.38, 'Select Top K\nFeatures', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='yellow', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.31, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, 0.16, 'Split Train/Test', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
ax1.arrow(0.5, 0.09, 0, -0.05, head_width=0.05, head_length=0.03, fc='black')
ax1.text(0.5, -0.05, '❌ Overly Optimistic\nPerformance', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2))

# Leakage warning
ax1.text(0.05, 0.40, '⚠️ LEAKAGE!\nTest data used\nin selection',
         fontsize=10, ha='left', color=RED, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7, edgecolor=RED, linewidth=2))

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.15, 1)
ax1.axis('off')

# Plot 2: Correct approach
ax2.text(0.5, 0.95, 'CORRECT: No Leakage', fontsize=16, fontweight='bold',
         ha='center', va='top', color=GREEN)
ax2.text(0.5, 0.80, 'All Data', fontsize=12, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
ax2.arrow(0.5, 0.75, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.text(0.5, 0.60, 'Split Train/Test', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
ax2.arrow(0.3, 0.53, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.arrow(0.7, 0.53, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.text(0.3, 0.38, 'TRAIN\nSelect Features\nTrain Model', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
ax2.text(0.7, 0.38, 'TEST\nApply Selection\nEvaluate', fontsize=10, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
ax2.arrow(0.5, 0.28, 0, -0.08, head_width=0.05, head_length=0.03, fc='black')
ax2.text(0.5, 0.10, '✅ Realistic\nPerformance', fontsize=11, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))

# Correct workflow note
ax2.text(0.05, 0.40, '✓ Pipeline ensures\nselection uses\nTRAIN data only',
         fontsize=10, ha='left', color=GREEN, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5, edgecolor=GREEN, linewidth=2))

ax2.set_xlim(0, 1)
ax2.set_ylim(-0.05, 1)
ax2.axis('off')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/data_leakage_prevention.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved data_leakage_prevention.png")

print("\n" + "=" * 70)
print("✓ All diagrams generated successfully!")
print(f"  Location: {OUTPUT_DIR}/")
print("=" * 70)

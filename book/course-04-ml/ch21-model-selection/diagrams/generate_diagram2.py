#!/usr/bin/env python3
"""Generate comparison: Single Split Variance vs CV Fold Scores"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Part 1: Generate single split accuracies
accuracies = []
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate
    model = LogisticRegression(random_state=42, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    acc = model.score(X_test_scaled, y_test)
    accuracies.append(acc)

accuracies = np.array(accuracies)

# Part 2: Generate CV scores
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42, max_iter=10000)
)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Histogram of single split variances
axes[0].hist(accuracies, bins=15, edgecolor='black', alpha=0.7, color='#2196F3')
axes[0].axvline(accuracies.mean(), color='#F44336', linestyle='--', linewidth=2,
                label=f'Mean: {accuracies.mean():.3f}')
axes[0].set_xlabel('Accuracy', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Single Split Accuracy (20 trials)\nHigh Variance', fontsize=13)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Right: CV fold scores
x_pos = np.arange(1, 6)
axes[1].bar(x_pos, cv_scores, color='#4CAF50', edgecolor='black', alpha=0.8)
axes[1].axhline(cv_scores.mean(), color='#F44336', linestyle='--', linewidth=2,
                label=f'Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
axes[1].fill_between([0.5, 5.5], cv_scores.mean() - cv_scores.std(),
                       cv_scores.mean() + cv_scores.std(),
                       alpha=0.2, color='#F44336', label='± 1 Std Dev')
axes[1].set_xlabel('Fold', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('5-Fold Cross-Validation\nLower Variance, More Reliable', fontsize=13)
axes[1].set_xticks(x_pos)
axes[1].set_ylim([0.90, 1.0])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-04-ml/ch21-model-selection/diagrams/single_split_vs_cv.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: single_split_vs_cv.png")
plt.close()

print(f"\nStats:")
print(f"  Single split std: {accuracies.std():.4f}")
print(f"  CV estimate std: {cv_scores.std():.4f}")

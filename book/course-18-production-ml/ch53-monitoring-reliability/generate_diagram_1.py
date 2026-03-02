import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import os

# Change to the chapter directory
os.chdir('/home/chirag/ds-book/book/course-18/ch53')

# Set random seed for reproducibility
np.random.seed(42)

# Use consistent color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Create 2x2 subplot for distribution shift types
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.patch.set_facecolor('white')

# Helper function to plot decision boundary
def plot_decision_boundary(ax, X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[y==0, 0], X[y==0, 1], c=colors['blue'], alpha=0.6, label='Class 0', s=30)
    ax.scatter(X[y==1, 0], X[y==1, 1], c=colors['red'], alpha=0.6, label='Class 1', s=30)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)

# 1. Baseline: Original training data
X_baseline, y_baseline = make_classification(n_samples=300, n_features=2,
                                              n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1,
                                              class_sep=1.5, random_state=42)
model_baseline = LogisticRegression(random_state=42)
model_baseline.fit(X_baseline, y_baseline)
plot_decision_boundary(axes[0, 0], X_baseline, y_baseline, model_baseline,
                       '1. Baseline (Training Data)')

# 2. Data Drift: Shifted feature distribution, same decision boundary
X_data_drift = X_baseline + np.array([2.5, 1.5])  # Shift features
y_data_drift = y_baseline  # Same labels
plot_decision_boundary(axes[0, 1], X_data_drift, y_data_drift, model_baseline,
                       '2. Data Drift: P(X) changed, P(y|X) same\nModel still works!')

# 3. Concept Drift: Same feature distribution, rotated decision boundary
# Rotate the relationship between X and y
angle = np.pi / 4  # 45 degrees
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_concept_drift = X_baseline.copy()
# Create new labels based on rotated boundary
X_rotated = X_baseline @ rotation_matrix.T
y_concept_drift = (X_rotated[:, 0] + X_rotated[:, 1] > 0).astype(int)
plot_decision_boundary(axes[1, 0], X_concept_drift, y_concept_drift,
                       model_baseline,
                       '3. Concept Drift: P(y|X) changed\nModel fails!')

# 4. Prior Shift: Same features and boundary, but class balance changed
# Oversample class 1 to change P(y)
class_1_indices = np.where(y_baseline == 1)[0]
class_0_indices = np.where(y_baseline == 0)[0]
# Take all class 1 samples and fewer class 0 samples
selected_indices = np.concatenate([
    np.random.choice(class_0_indices, size=50, replace=False),
    class_1_indices
])
X_prior_shift = X_baseline[selected_indices]
y_prior_shift = y_baseline[selected_indices]
plot_decision_boundary(axes[1, 1], X_prior_shift, y_prior_shift,
                       model_baseline,
                       f'4. Prior Shift: P(y) changed\nClass ratio: {y_prior_shift.mean():.2f} vs {y_baseline.mean():.2f}')

plt.tight_layout()
plt.savefig('diagrams/distribution_shifts.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/distribution_shifts.png")
plt.close()

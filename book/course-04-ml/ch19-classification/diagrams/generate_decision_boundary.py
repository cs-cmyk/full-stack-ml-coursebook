import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Use features 0 and 1 (mean radius and mean texture)
X_2d = X[:, [0, 1]]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

# Train model on 2D data
model_2d = LogisticRegression(random_state=42, max_iter=10000)
model_2d.fit(X_train_2d, y_train_2d)

# Create mesh for decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict probability across the mesh
Z = model_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
cbar = plt.colorbar(label='P(benign)')
cbar.ax.tick_params(labelsize=11)

# Plot data points
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d,
                     cmap='RdYlBu_r', edgecolors='black', s=50, alpha=0.8, linewidths=1.5)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)

plt.xlabel(data.feature_names[0], fontsize=13)
plt.ylabel(data.feature_names[1], fontsize=13)
plt.title('Logistic Regression Decision Boundary\n(Using 2 Features for Visualization)',
          fontsize=14, weight='bold')
legend = plt.legend(*scatter.legend_elements(), title='Class', labels=['Malignant', 'Benign'], fontsize=11)
legend.get_title().set_fontsize(12)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-04-ml/ch19-classification/diagrams/decision_boundary_2d.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: decision_boundary_2d.png")

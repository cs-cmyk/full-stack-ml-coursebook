import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True, ax=axes[0],
            xticklabels=data.target_names,
            yticklabels=data.target_names,
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_xlabel('Predicted Class', fontsize=13, weight='bold')
axes[0].set_ylabel('Actual Class', fontsize=13, weight='bold')
axes[0].set_title('3×3 Confusion Matrix for Iris Classification', fontsize=14, weight='bold')

# Plot 2: Decision boundaries using 2 features (petal length and petal width)
X_2d = X[:, [2, 3]]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, random_state=42, stratify=y
)

model_2d = LogisticRegression(random_state=42, max_iter=1000)
model_2d.fit(X_train_2d, y_train_2d)

# Create mesh
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Predict class for each point in mesh
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
axes[1].contourf(xx, yy, Z, alpha=0.4, cmap='viridis', levels=[0, 1, 2, 3])

# Plot test points
scatter = axes[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d,
                         cmap='viridis', edgecolors='black', s=80, alpha=0.9,
                         linewidths=1.5)

axes[1].set_xlabel(data.feature_names[2], fontsize=13, weight='bold')
axes[1].set_ylabel(data.feature_names[3], fontsize=13, weight='bold')
axes[1].set_title('Multi-Class Decision Boundaries\n(Using Petal Features)',
                 fontsize=14, weight='bold')

# Create custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.viridis(i/2),
                              markersize=10, label=data.target_names[i],
                              markeredgecolor='black', markeredgewidth=1.5)
                  for i in range(3)]
legend = axes[1].legend(handles=legend_elements, title='Species', loc='upper left', fontsize=11)
legend.get_title().set_fontsize(12)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-04-ml/ch19-classification/diagrams/multiclass_classification.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: multiclass_classification.png")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Change to the chapter directory
os.chdir('/home/chirag/ds-book/book/course-18/ch53')

# Use consistent color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

np.random.seed(42)

# Generate baseline data with clear decision boundary
X_train, y_train = make_classification(
    n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
    n_clusters_per_class=1, class_sep=2.0, random_state=42
)

# Train baseline model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Track performance over time with gradual concept drift
time_steps = 50
accuracies = []
rotation_angles = np.linspace(0, np.pi/2, time_steps)  # Gradual rotation from 0° to 90°

for t, angle in enumerate(rotation_angles):
    # Simulate gradual rotation of decision boundary
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    X_rotated = X_train @ rotation_matrix.T
    y_drifted = (X_rotated[:, 0] > 0).astype(int)

    # Evaluate model on drifted data
    accuracy = accuracy_score(y_drifted, model.predict(X_train))
    accuracies.append(accuracy)

# Plot performance degradation over time
plt.figure(figsize=(10, 6))
plt.gcf().patch.set_facecolor('white')
plt.plot(range(time_steps), accuracies, linewidth=2.5, color=colors['blue'])
plt.axhline(y=0.5, color=colors['red'], linestyle='--', linewidth=2,
            label='Random Baseline')
plt.axhline(y=0.85, color=colors['orange'], linestyle='--', linewidth=2,
            label='Alert Threshold (85%)')
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Model Accuracy', fontsize=12)
plt.title('Model Performance Under Gradual Concept Drift\n(Decision boundary rotates 0° → 90°)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.ylim([0.4, 1.0])
plt.tight_layout()
plt.savefig('diagrams/concept_drift_degradation.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/concept_drift_degradation.png")
plt.close()

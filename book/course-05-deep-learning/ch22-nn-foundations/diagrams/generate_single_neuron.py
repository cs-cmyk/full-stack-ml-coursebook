#!/usr/bin/env python3
"""Generate single neuron decision boundary and loss curve."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2D linearly separable binary classification data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.0,
    class_sep=2.0,
    random_state=42
)

# Sigmoid activation function
def sigmoid(z):
    """Compute sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Binary cross-entropy loss
def binary_cross_entropy(y_true, y_pred):
    """Compute binary cross-entropy loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialize weights and bias
n_features = X.shape[1]
w = np.random.randn(n_features) * 0.01
b = 0.0

# Training parameters
learning_rate = 0.1
n_iterations = 100
losses = []

# Training loop (gradient descent)
for iteration in range(n_iterations):
    # Forward pass
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)

    # Compute loss
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)

    # Compute gradients
    dz = y_pred - y
    dw = np.dot(X.T, dz) / len(y)
    db = np.mean(dz)

    # Update weights and bias
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Visualize decision boundary and loss curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Decision boundary
ax1.scatter(X[y==0, 0], X[y==0, 1], c='#2196F3', label='Class 0', s=50, alpha=0.7)
ax1.scatter(X[y==1, 0], X[y==1, 1], c='#F44336', label='Class 1', s=50, alpha=0.7)

# Create meshgrid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
z_grid = np.dot(grid, w) + b
probs = sigmoid(z_grid).reshape(xx.shape)

# Plot decision boundary (where probability = 0.5)
ax1.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
ax1.contourf(xx, yy, probs, levels=[0, 0.5, 1], colors=['#2196F3', '#F44336'], alpha=0.3)
ax1.set_xlabel('Feature 1', fontsize=11)
ax1.set_ylabel('Feature 2', fontsize=11)
ax1.set_title('Single Neuron Decision Boundary (Linear)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Loss curve
ax2.plot(range(1, n_iterations+1), losses, linewidth=2, color='#2196F3')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Binary Cross-Entropy Loss', fontsize=11)
ax2.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('single_neuron_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: single_neuron_results.png")
plt.close()

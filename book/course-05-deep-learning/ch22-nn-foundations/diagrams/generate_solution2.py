#!/usr/bin/env python3
"""Generate Solution 2: Make Moons Decision Boundaries."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate data
X, y = make_moons(n_samples=400, noise=0.15, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Activation functions
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Initialize and train network
def initialize_network(hidden_size=8):
    W1 = np.random.randn(2, hidden_size) * 0.5
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * 0.5
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

def train_network(X_train, y_train, hidden_size=8, epochs=500, batch_size=32, lr=0.1, reg_lambda=0.0):
    W1, b1, W2, b2 = initialize_network(hidden_size)
    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Forward pass
            z1 = np.dot(X_batch, W1) + b1
            a1 = tanh(z1)
            z2 = np.dot(a1, W2) + b2
            a2 = sigmoid(z2)

            # Backward pass
            m = X_batch.shape[0]
            dz2 = a2 - y_batch
            dW2 = (np.dot(a1.T, dz2) + reg_lambda * W2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m

            da1 = np.dot(dz2, W2.T)
            dz1 = da1 * tanh_derivative(z1)
            dW1 = (np.dot(X_batch.T, dz1) + reg_lambda * W1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            # Update weights
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

    return W1, b1, W2, b2

# Train without regularization
W1, b1, W2, b2 = train_network(X_train, y_train, hidden_size=8, epochs=500, lr=0.1, reg_lambda=0.0)

# Train with L2 regularization
W1_reg, b1_reg, W2_reg, b2_reg = train_network(X_train, y_train, hidden_size=8, epochs=500, lr=0.1, reg_lambda=0.01)

# Visualize decision boundaries
def plot_decision_boundary(W1, b1, W2, b2, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    z1 = np.dot(grid, W1) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    probs = a2.reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=[0, 0.5, 1], colors=['#2196F3', '#F44336'], alpha=0.4)
    ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X[y.ravel()==0, 0], X[y.ravel()==0, 1], c='#2196F3', label='Class 0', s=30, alpha=0.7)
    ax.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], c='#F44336', label='Class 1', s=30, alpha=0.7)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_decision_boundary(W1, b1, W2, b2, X_train, y_train,
                      'Decision Boundary (No Regularization)', ax1)
plot_decision_boundary(W1_reg, b1_reg, W2_reg, b2_reg, X_train, y_train,
                      'Decision Boundary (With L2 Regularization)', ax2)
plt.tight_layout()
plt.savefig('solution2_boundaries.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: solution2_boundaries.png")
plt.close()

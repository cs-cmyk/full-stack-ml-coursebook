#!/usr/bin/env python3
"""Generate XOR problem solution with decision boundary."""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# XOR dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Two-layer neural network class
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, activation='relu'):
        scale = np.sqrt(2.0 / input_size) if activation == 'relu' else np.sqrt(1.0 / input_size)

        self.W1 = np.random.randn(input_size, hidden_size) * scale
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * scale
        self.b2 = np.zeros((1, output_size))

        self.activation = activation
        if activation == 'relu':
            self.act_fn = relu
            self.act_derivative = relu_derivative
        else:
            self.act_fn = sigmoid
            self.act_derivative = sigmoid_derivative

    def forward(self, X):
        """Forward propagation through both layers"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.act_fn(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        """Backpropagation: compute gradients and update weights"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.act_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, learning_rate):
        """Training loop"""
        losses = []
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clipped) +
                           (1 - y) * np.log(1 - y_pred_clipped))
            losses.append(loss)

            # Backward pass
            self.backward(X, y, learning_rate)

        return losses

# Train network on XOR
model = TwoLayerNet(input_size=2, hidden_size=2, output_size=1, activation='relu')
losses = model.train(X_xor, y_xor, epochs=1000, learning_rate=0.1)

# Visualize XOR solution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Non-linear decision boundary
colors = ['#2196F3' if y[0] == 0 else '#F44336' for y in y_xor]
ax1.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=300, edgecolors='black',
           linewidth=3, zorder=10)
for i in range(4):
    ax1.text(X_xor[i, 0], X_xor[i, 1], f'  {y_xor[i, 0]}',
            fontsize=12, fontweight='bold')

# Create meshgrid for decision boundary
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.forward(grid).reshape(xx.shape)

# Plot decision boundary
ax1.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=3)
ax1.contourf(xx, yy, probs, levels=[0, 0.5, 1], colors=['#2196F3', '#F44336'], alpha=0.4)
ax1.set_xlabel('x₁', fontsize=12)
ax1.set_ylabel('x₂', fontsize=12)
ax1.set_title('XOR Solved: Non-Linear Decision Boundary', fontsize=13, fontweight='bold')
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.grid(True, alpha=0.3)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])

# Plot 2: Loss curve
ax2.plot(range(1, len(losses)+1), losses, linewidth=2, color='#2196F3')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
ax2.set_title('Training Loss: XOR Problem', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('xor_solution.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: xor_solution.png")
plt.close()

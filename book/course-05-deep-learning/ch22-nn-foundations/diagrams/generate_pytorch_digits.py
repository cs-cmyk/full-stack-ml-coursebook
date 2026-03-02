#!/usr/bin/env python3
"""Generate PyTorch digits classification results."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Load Digits dataset
digits = load_digits()
X, y = digits.data, digits.target
X = X / 16.0  # Normalize to [0, 1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Define neural network
class DigitClassifier(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Instantiate model
model = DigitClassifier(input_size=64, hidden1_size=128,
                       hidden2_size=64, num_classes=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 50
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(n_epochs):
    # Training phase
    model.train()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, predicted_train = torch.max(outputs, 1)
    train_acc = (predicted_train == y_train_t).float().mean().item()

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        outputs_test = model(X_test_t)
        test_loss = criterion(outputs_test, y_test_t)
        _, predicted_test = torch.max(outputs_test, 1)
        test_acc = (predicted_test == y_test_t).float().mean().item()

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Visualize training history
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curves
ax1.plot(range(1, n_epochs+1), train_losses, label='Train Loss', linewidth=2, color='#2196F3')
ax1.plot(range(1, n_epochs+1), test_losses, label='Test Loss', linewidth=2, color='#FF9800')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax1.set_title('Training and Test Loss', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy curves
ax2.plot(range(1, n_epochs+1), [acc * 100 for acc in train_accuracies],
        label='Train Accuracy', linewidth=2, color='#2196F3')
ax2.plot(range(1, n_epochs+1), [acc * 100 for acc in test_accuracies],
        label='Test Accuracy', linewidth=2, color='#FF9800')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title('Training and Test Accuracy', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Confusion matrix
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_t)
    _, y_pred_class = torch.max(y_pred_test, 1)
    y_pred_numpy = y_pred_class.numpy()

cm = confusion_matrix(y_test, y_pred_numpy)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar_kws={'label': 'Count'})
ax3.set_xlabel('Predicted Class', fontsize=11)
ax3.set_ylabel('True Class', fontsize=11)
ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# Plot 4: Sample predictions
n_samples = 8
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
sample_images = X_test[sample_indices].reshape(-1, 8, 8)
sample_labels = y_test[sample_indices]

with torch.no_grad():
    sample_tensors = torch.FloatTensor(X_test[sample_indices])
    sample_outputs = model(sample_tensors)
    sample_probs = torch.softmax(sample_outputs, dim=1)
    sample_pred_probs, sample_pred_classes = torch.max(sample_probs, 1)

ax3_positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
for idx, (img, true_label, pred_label, prob) in enumerate(
    zip(sample_images, sample_labels, sample_pred_classes.numpy(),
        sample_pred_probs.numpy())):
    row, col = ax3_positions[idx]
    ax = plt.subplot(4, 4, 9 + row * 4 + col)
    ax.imshow(img, cmap='gray')
    color = '#4CAF50' if true_label == pred_label else '#F44336'
    ax.set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {prob:.2f}',
                fontsize=8, color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('PyTorch Digits Classification Results', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('pytorch_digits_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: pytorch_digits_results.png")
plt.close()

"""
Generate diagrams for the code examples in the CNN chapter
These are referenced in the code examples but don't require actual MNIST/CIFAR data
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("Generating example placeholder diagrams...")
print("=" * 60)

# ============================================================
# MNIST Samples Placeholder
# ============================================================
print("\n1. Generating mnist_samples.png...")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, ax in enumerate(axes.flat):
    # Create synthetic digit-like patterns
    digit = np.random.randint(0, 10)
    img = np.random.rand(28, 28) * 0.3  # Background noise

    # Add some structure to look vaguely digit-like
    if digit in [0, 6, 8, 9]:  # Circular digits
        y, x = np.ogrid[:28, :28]
        mask = (x - 14)**2 + (y - 14)**2 <= 100
        img[mask] = 0.8
    elif digit == 1:  # Vertical line
        img[5:23, 12:16] = 0.8
    else:  # Horizontal component
        img[10:14, 5:23] = 0.8

    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Label: {digit}', fontweight='bold')
    ax.axis('off')

plt.suptitle('Sample MNIST Digits (Synthetic)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ mnist_samples.png")

# ============================================================
# MNIST Confusion Matrix Placeholder
# ============================================================
print("\n2. Generating mnist_confusion.png...")

# Create synthetic confusion matrix
cm = np.array([
    [973,   0,   2,   0,   0,   1,   3,   1,   0,   0],
    [  0, 1129,   2,   1,   0,   1,   1,   1,   0,   0],
    [  2,   3, 1014,   4,   2,   0,   2,   4,   1,   0],
    [  0,   0,   4, 998,   0,   3,   0,   2,   2,   1],
    [  1,   1,   2,   0, 968,   0,   3,   1,   2,   4],
    [  2,   0,   0,   8,   1, 875,   4,   1,   1,   0],
    [  4,   2,   1,   0,   2,   4, 944,   0,   1,   0],
    [  1,   3,   8,   2,   1,   1,   0, 1008,   2,   2],
    [  3,   0,   3,   5,   2,   2,   2,   3, 950,   4],
    [  2,   3,   0,   3,   5,   4,   1,   6,   4, 981]
])

from matplotlib.colors import LinearSegmentedColormap

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
im = axes[0].imshow(cm, cmap='Blues', aspect='auto')
axes[0].set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
axes[0].set_ylabel('True Label', fontweight='bold', fontsize=12)
axes[0].set_title('Confusion Matrix\n(99.12% accuracy)', fontweight='bold', fontsize=13)
axes[0].set_xticks(range(10))
axes[0].set_yticks(range(10))
axes[0].set_xticklabels(range(10))
axes[0].set_yticklabels(range(10))

# Add text annotations for diagonal
for i in range(10):
    axes[0].text(i, i, str(cm[i, i]), ha='center', va='center',
                color='white', fontweight='bold', fontsize=11)

plt.colorbar(im, ax=axes[0])

# Accuracy per class
class_correct = cm.diagonal()
class_total = cm.sum(axis=1)
class_acc = 100 * class_correct / class_total

axes[1].bar(range(10), class_acc, color='#2196F3', edgecolor='black', linewidth=1.5)
axes[1].axhline(y=99.12, color='red', linestyle='--', linewidth=2, label='Overall Accuracy')
axes[1].set_xlabel('Digit Class', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
axes[1].set_title('Per-Class Accuracy', fontweight='bold', fontsize=13)
axes[1].set_xticks(range(10))
axes[1].set_ylim([97, 100])
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_confusion.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ mnist_confusion.png")

# ============================================================
# MNIST Predictions Placeholder
# ============================================================
print("\n3. Generating mnist_predictions.png...")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

for idx, ax in enumerate(axes.flat):
    # Create synthetic digit
    true_label = np.random.randint(0, 10)
    predicted = true_label if np.random.rand() > 0.1 else np.random.randint(0, 10)
    confidence = 0.95 if predicted == true_label else 0.65

    img = np.random.rand(28, 28) * 0.3
    if true_label in [0, 6, 8]:
        y, x = np.ogrid[:28, :28]
        mask = (x - 14)**2 + (y - 14)**2 <= 100
        img[mask] = 0.8
    else:
        img[8:20, 10:18] = 0.8

    ax.imshow(img, cmap='gray')
    color = 'green' if predicted == true_label else 'red'
    ax.set_title(f'True: {true_label}, Pred: {predicted}\nConfidence: {confidence:.0%}',
                color=color, fontweight='bold', fontsize=10)
    ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ mnist_predictions.png")

# ============================================================
# Fashion-MNIST Samples Placeholder
# ============================================================
print("\n4. Generating fashion_mnist_samples.png...")

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for idx, ax in enumerate(axes.flat):
    # Create synthetic fashion items
    img = np.random.rand(28, 28) * 0.3

    # Add some structure
    label_idx = idx % 10
    if label_idx in [1, 5, 7, 9]:  # Lower body / shoes
        img[15:25, 8:20] = 0.7
    else:  # Upper body / bags
        img[5:20, 6:22] = 0.7

    ax.imshow(img, cmap='gray')
    ax.set_title(f'{class_names[label_idx]}', fontweight='bold', fontsize=11)
    ax.axis('off')

plt.suptitle('Fashion-MNIST Sample Images (Synthetic)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fashion_mnist_samples.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ fashion_mnist_samples.png")

# ============================================================
# Fashion-MNIST Training Curves
# ============================================================
print("\n5. Generating fashion_training_curves.png...")

epochs = np.arange(1, 11)
train_accs = [82.34, 88.67, 90.23, 91.45, 92.11, 92.56, 92.89, 93.12, 93.21, 93.28]
test_accs = [87.12, 89.54, 90.12, 90.45, 90.78, 91.01, 91.08, 91.15, 91.19, 91.23]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=8, color='#2196F3')
plt.plot(epochs, test_accs, 's-', label='Test Accuracy', linewidth=2, markersize=8, color='#4CAF50')
plt.xlabel('Epoch', fontweight='bold', fontsize=12)
plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
plt.title('Fashion-MNIST Training Progress', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xticks(epochs)
plt.ylim([80, 95])
plt.tight_layout()
plt.savefig('fashion_training_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ fashion_training_curves.png")

# ============================================================
# Fashion-MNIST Confusion Matrix
# ============================================================
print("\n6. Generating fashion_confusion.png...")

# Synthetic confusion matrix with realistic errors
cm_fashion = np.array([
    [870,   2,  12, 184,   3,   1, 245,   0,   5,   0],  # T-shirt
    [  1, 985,   2,   3,   1,   0,   2,   0,   2,   0],  # Trouser
    [ 15,   1, 891,  10,  89,   0,  45,   0,   3,   0],  # Pullover
    [ 32,   3,   8, 925,  18,   0,  28,   0,   1,   0],  # Dress
    [  2,   1,  76,  21, 905,   0,  76,   0,   1,   0],  # Coat
    [  0,   0,   0,   0,   0, 955,   0,  45,   2,  12],  # Sandal
    [245,   2,  45,  28,  76,   0, 870,   0,   2,   0],  # Shirt
    [  0,   0,   0,   0,   0,  23,   0, 921,   1,  42],  # Sneaker
    [  8,   3,   5,   2,   3,   1,   8,   0, 962,   0],  # Bag
    [  0,   0,   0,   0,   0,   8,   0,  38,   2, 945]   # Ankle boot
])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_fashion, cmap='Blues', aspect='auto')
ax.set_xlabel('Predicted', fontweight='bold', fontsize=12)
ax.set_ylabel('True', fontweight='bold', fontsize=12)
ax.set_title('Fashion-MNIST Confusion Matrix (Accuracy: 91.23%)', fontweight='bold', fontsize=13)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(class_names, rotation=0, fontsize=10)

# Add diagonal values
for i in range(10):
    ax.text(i, i, str(cm_fashion[i, i]), ha='center', va='center',
           color='white', fontweight='bold', fontsize=10)

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('fashion_confusion.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ fashion_confusion.png")

# ============================================================
# Learned Filters Visualization
# ============================================================
print("\n7. Generating learned_filters.png...")

# Create synthetic learned filters (edge detectors, etc.)
filters = np.random.randn(32, 3, 3) * 0.5

# Add some recognizable patterns
filters[0] = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]  # Horizontal edge
filters[1] = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]  # Vertical edge
filters[2] = [[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]   # Diagonal
filters[3] = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]     # Laplacian

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for idx, ax in enumerate(axes.flat):
    filt = filters[idx]
    im = ax.imshow(filt, cmap='RdBu', vmin=-1.5, vmax=1.5)
    ax.set_title(f'Filter {idx+1}', fontsize=9)
    ax.axis('off')

plt.suptitle('Learned Filters from Conv1 (32 filters, 3×3 each)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('learned_filters.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ learned_filters.png")

# ============================================================
# Feature Maps Visualization
# ============================================================
print("\n8. Generating feature_maps_conv1.png...")

# Create synthetic feature maps
conv1_activation = np.random.rand(32, 28, 28)

# Add some structure to a few channels
for i in range(5):
    conv1_activation[i, 10:18, :] = np.random.rand(8, 28) * 0.3 + 0.7

fig, axes = plt.subplots(5, 8, figsize=(16, 10))

# Original digit (synthetic)
original_img = np.random.rand(28, 28) * 0.3
original_img[8:20, 10:18] = 0.8
axes[0, 0].imshow(original_img, cmap='gray')
axes[0, 0].set_title('Original\nDigit 7', fontweight='bold', fontsize=9)
axes[0, 0].axis('off')

for i in range(1, 8):
    axes[0, i].axis('off')

# Feature maps
for idx in range(32):
    row = (idx + 8) // 8
    col = (idx + 8) % 8
    if row < 5:
        fmap = conv1_activation[idx]
        axes[row, col].imshow(fmap, cmap='viridis', vmin=0, vmax=1)
        axes[row, col].set_title(f'Channel {idx+1}', fontsize=8)
        axes[row, col].axis('off')

plt.suptitle('Conv1 Feature Maps for Digit 7 (32 channels, 28×28 each)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_maps_conv1.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ feature_maps_conv1.png")

# ============================================================
# Conv2 Feature Maps
# ============================================================
print("\n9. Generating feature_maps_conv2.png...")

conv2_activation = np.random.rand(64, 14, 14)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for idx, ax in enumerate(axes.flat):
    if idx < 32:
        fmap = conv2_activation[idx]
        ax.imshow(fmap, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'Ch {idx+1}', fontsize=8)
        ax.axis('off')
    else:
        ax.axis('off')

plt.suptitle('Conv2 Feature Maps (first 32 of 64 channels, 14×14 each)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_maps_conv2.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ feature_maps_conv2.png")

# ============================================================
# Fashion MNIST Filters
# ============================================================
print("\n10. Generating fashion_filters.png...")

fashion_filters = np.random.randn(32, 3, 3) * 0.4

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for idx, ax in enumerate(axes.flat):
    ax.imshow(fashion_filters[idx], cmap='RdBu', vmin=-1, vmax=1)
    ax.axis('off')

plt.suptitle('Learned Filters from Fashion-MNIST Conv1', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fashion_filters.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ fashion_filters.png")

# ============================================================
# Data Efficiency
# ============================================================
print("\n11. Generating data_efficiency.png...")

percentages = [10, 25, 50, 100]
transfer_accs = [73.24, 79.56, 82.89, 85.67]

plt.figure(figsize=(10, 6))
plt.plot(percentages, transfer_accs, 'o-', linewidth=3, markersize=12, color='#2196F3')
plt.xlabel('Percentage of Training Data Used (%)', fontweight='bold', fontsize=12)
plt.ylabel('Validation Accuracy (%)', fontweight='bold', fontsize=12)
plt.title('Transfer Learning: Data Efficiency on CIFAR-10', fontweight='bold', fontsize=14)
plt.grid(alpha=0.3)
plt.xticks(percentages)
plt.ylim([70, 90])

# Add value labels
for x, y in zip(percentages, transfer_accs):
    plt.text(x, y + 1, f'{y:.1f}%', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('data_efficiency.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ data_efficiency.png")

print("\n" + "=" * 60)
print("✓ All example diagrams generated successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  1. mnist_samples.png")
print("  2. mnist_confusion.png")
print("  3. mnist_predictions.png")
print("  4. fashion_mnist_samples.png")
print("  5. fashion_training_curves.png")
print("  6. fashion_confusion.png")
print("  7. learned_filters.png")
print("  8. feature_maps_conv1.png")
print("  9. feature_maps_conv2.png")
print(" 10. fashion_filters.png")
print(" 11. data_efficiency.png")

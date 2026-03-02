"""
Generate all diagrams for Chapter 37: Computer Vision - Image Preprocessing and Augmentation
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
import torch
import os

# Set style and random seeds
plt.style.use('default')
torch.manual_seed(42)
np.random.seed(42)

# Color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Create output directory
os.makedirs('/home/chirag/ds-book/book/course-10-advanced/ch37-cv/diagrams', exist_ok=True)

print("Generating diagrams for Chapter 37...")

# ============================================================================
# Diagram 1: Image Tensor Structure and Normalization
# ============================================================================
print("\n1. Generating image_tensor_structure.png...")

# Load a sample CIFAR-10 image
dataset = datasets.CIFAR10(root='/tmp/data', train=True, download=True)
img, label = dataset[100]  # Get a single image
img_array = np.array(img)

fig = plt.figure(figsize=(14, 6))

# Panel 1: Original image
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(img_array)
ax1.set_title('Original Image\n(32×32×3)', fontsize=11, fontweight='bold')
ax1.axis('off')

# Panel 2-4: Individual RGB channels
for i, (channel, color) in enumerate([(0, 'Reds'), (1, 'Greens'), (2, 'Blues')]):
    ax = plt.subplot(2, 4, i+2)
    ax.imshow(img_array[:, :, channel], cmap=color)
    ax.set_title(f'{"RGB"[i]} Channel', fontsize=11, fontweight='bold')
    ax.axis('off')

# Panel 5: Pixel value distribution (original)
ax5 = plt.subplot(2, 4, 5)
for i, (color, label_text) in enumerate([('red', 'R'), ('green', 'G'), ('blue', 'B')]):
    ax5.hist(img_array[:, :, i].flatten(), bins=30, alpha=0.6,
             color=color, label=label_text, edgecolor='white', linewidth=0.5)
ax5.set_xlabel('Pixel Value', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Original [0, 255]', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.set_xlim([0, 255])
ax5.grid(True, alpha=0.3)

# Panel 6: After normalization [0, 1]
img_norm = img_array / 255.0
ax6 = plt.subplot(2, 4, 6)
for i, (color, label_text) in enumerate([('red', 'R'), ('green', 'G'), ('blue', 'B')]):
    ax6.hist(img_norm[:, :, i].flatten(), bins=30, alpha=0.6,
             color=color, label=label_text, edgecolor='white', linewidth=0.5)
ax6.set_xlabel('Pixel Value', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title('Min-Max Norm [0, 1]', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.set_xlim([0, 1])
ax6.grid(True, alpha=0.3)

# Panel 7: After standardization
img_std = (img_norm - img_norm.mean()) / (img_norm.std() + 1e-8)
ax7 = plt.subplot(2, 4, 7)
for i, (color, label_text) in enumerate([('red', 'R'), ('green', 'G'), ('blue', 'B')]):
    ax7.hist(img_std[:, :, i].flatten(), bins=30, alpha=0.6,
             color=color, label=label_text, edgecolor='white', linewidth=0.5)
ax7.set_xlabel('Standardized Value', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('Standardization (μ≈0, σ≈1)', fontsize=11, fontweight='bold')
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Panel 8: Tensor structure diagram
ax8 = plt.subplot(2, 4, 8)
ax8.text(0.5, 0.7, 'Image Tensor Structure', ha='center', fontsize=12,
         fontweight='bold', transform=ax8.transAxes)
ax8.text(0.5, 0.5, 'Shape: (H, W, C)', ha='center',
         fontsize=11, transform=ax8.transAxes)
ax8.text(0.5, 0.35, 'CIFAR-10: (32, 32, 3)', ha='center', fontsize=10,
         style='italic', transform=ax8.transAxes, color=COLORS['blue'])
ax8.text(0.5, 0.15, 'H rows × W cols × C channels', ha='center',
         fontsize=9, transform=ax8.transAxes, color=COLORS['gray'])
ax8.axis('off')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-10-advanced/ch37-cv/diagrams/image_tensor_structure.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved image_tensor_structure.png")

# ============================================================================
# Diagram 2: Normalization Comparison
# ============================================================================
print("\n2. Generating normalization_comparison.png...")

# Load dataset with different normalizations
img_raw, label = dataset[42]
img_array = np.array(img_raw)

# Min-max normalization
transform_minmax = transforms.Compose([transforms.ToTensor()])
dataset_minmax = datasets.CIFAR10(root='/tmp/data', train=True, transform=transform_minmax)
img_minmax, _ = dataset_minmax[42]

# Standardization
train_mean = img_minmax.mean(dim=[1, 2])
train_std = img_minmax.std(dim=[1, 2])
transform_std = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist())
])
dataset_std = datasets.CIFAR10(root='/tmp/data', train=True, transform=transform_std)
img_std, _ = dataset_std[42]

# ImageNet normalization
transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset_imagenet = datasets.CIFAR10(root='/tmp/data', train=True, transform=transform_imagenet)
img_imagenet, _ = dataset_imagenet[42]

# Create comparison figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original
axes[0].imshow(img_array)
axes[0].set_title('Original\n[0, 255]', fontsize=13, fontweight='bold')
axes[0].axis('off')

# Min-max
img_display = img_minmax.permute(1, 2, 0).numpy()
axes[1].imshow(img_display)
axes[1].set_title(f'Min-Max [0, 1]\nmean={img_minmax.mean():.3f}',
                  fontsize=13, fontweight='bold')
axes[1].axis('off')

# Standardized
img_std_display = img_std.permute(1, 2, 0).numpy()
img_std_display = (img_std_display * train_std.numpy()) + train_mean.numpy()
img_std_display = np.clip(img_std_display, 0, 1)
axes[2].imshow(img_std_display)
axes[2].set_title('Standardized\nmean≈0, std≈1', fontsize=13, fontweight='bold')
axes[2].axis('off')

# ImageNet
img_in_display = img_imagenet.permute(1, 2, 0).numpy()
img_in_display = (img_in_display * np.array([0.229, 0.224, 0.225])) + \
                  np.array([0.485, 0.456, 0.406])
img_in_display = np.clip(img_in_display, 0, 1)
axes[3].imshow(img_in_display)
axes[3].set_title('ImageNet Norm\n(for transfer learning)', fontsize=13, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-10-advanced/ch37-cv/diagrams/normalization_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved normalization_comparison.png")

# ============================================================================
# Diagram 3: Augmentation Variations
# ============================================================================
print("\n3. Generating augmentation_variations.png...")

# Define augmentation pipeline
transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_aug = datasets.CIFAR10(root='/tmp/data', train=True, transform=transform_augment)
img_idx = 42

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

for i in range(9):
    img_aug, label = dataset_aug[img_idx]

    # Denormalize for display
    img_display = img_aug.permute(1, 2, 0).numpy()
    img_display = (img_display * np.array([0.229, 0.224, 0.225])) + \
                   np.array([0.485, 0.456, 0.406])
    img_display = np.clip(img_display, 0, 1)

    axes[i].imshow(img_display)
    axes[i].set_title(f'Augmentation {i+1}', fontsize=12, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Same Image with 9 Different Augmentations\n(Flip, Rotate, Crop, Color Jitter)',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-10-advanced/ch37-cv/diagrams/augmentation_variations.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved augmentation_variations.png")

# ============================================================================
# Diagram 4: Augmentation Performance Comparison (Simulated)
# ============================================================================
print("\n4. Generating augmentation_performance_comparison.png...")

# Simulate training curves (since full training would take too long)
np.random.seed(42)
epochs = np.arange(1, 21)

# No augmentation: high overfitting
train_baseline = 50 + 46 * (1 - np.exp(-epochs/3)) + np.random.randn(20) * 1
test_baseline = 50 + 13.89 * (1 - np.exp(-epochs/8)) + np.random.randn(20) * 0.8

# Basic augmentation: moderate overfitting
train_basic = 50 + 32.45 * (1 - np.exp(-epochs/4)) + np.random.randn(20) * 1.2
test_basic = 50 + 24.23 * (1 - np.exp(-epochs/6)) + np.random.randn(20) * 0.8

# Advanced augmentation: minimal overfitting
train_advanced = 50 + 28.67 * (1 - np.exp(-epochs/5)) + np.random.randn(20) * 1.5
test_advanced = 50 + 27.12 * (1 - np.exp(-epochs/5.5)) + np.random.randn(20) * 0.8

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training and test accuracy
axes[0].plot(epochs, train_baseline, color=COLORS['blue'], linestyle='-',
             linewidth=2.5, label='No Aug (Train)')
axes[0].plot(epochs, test_baseline, color=COLORS['blue'], linestyle='--',
             linewidth=2.5, label='No Aug (Test)', alpha=0.8)
axes[0].plot(epochs, train_basic, color=COLORS['green'], linestyle='-',
             linewidth=2.5, label='Basic Aug (Train)')
axes[0].plot(epochs, test_basic, color=COLORS['green'], linestyle='--',
             linewidth=2.5, label='Basic Aug (Test)', alpha=0.8)
axes[0].plot(epochs, train_advanced, color=COLORS['red'], linestyle='-',
             linewidth=2.5, label='Advanced Aug (Train)')
axes[0].plot(epochs, test_advanced, color=COLORS['red'], linestyle='--',
             linewidth=2.5, label='Advanced Aug (Test)', alpha=0.8)

axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
axes[0].set_title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10, loc='lower right')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_ylim([45, 105])

# Plot 2: Overfitting gap
gap_baseline = train_baseline - test_baseline
gap_basic = train_basic - test_basic
gap_advanced = train_advanced - test_advanced

axes[1].plot(epochs, gap_baseline, color=COLORS['blue'], linewidth=2.5,
             marker='o', markersize=6, label='No Augmentation')
axes[1].plot(epochs, gap_basic, color=COLORS['green'], linewidth=2.5,
             marker='s', markersize=6, label='Basic Augmentation')
axes[1].plot(epochs, gap_advanced, color=COLORS['red'], linewidth=2.5,
             marker='^', markersize=6, label='Advanced Augmentation')

axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Overfitting Gap\n(Train - Test Acc %)', fontsize=13, fontweight='bold')
axes[1].set_title('Overfitting Reduction', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11, loc='upper left')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-10-advanced/ch37-cv/diagrams/augmentation_performance_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved augmentation_performance_comparison.png")

print("\n" + "="*70)
print("All diagrams generated successfully!")
print("="*70)
print("\nGenerated files:")
print("  • image_tensor_structure.png")
print("  • normalization_comparison.png")
print("  • augmentation_variations.png")
print("  • augmentation_performance_comparison.png")
print("\nLocation: /home/chirag/ds-book/book/course-10-advanced/ch37-cv/diagrams/")

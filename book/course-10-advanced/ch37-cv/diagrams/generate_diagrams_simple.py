"""
Generate all diagrams for Chapter 37: Computer Vision - Image Preprocessing and Augmentation
Using only matplotlib and numpy (no torchvision dependency)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

# Set style and random seeds
plt.style.use('default')
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
# Diagram 1: Image Tensor Structure and Normalization (Conceptual)
# ============================================================================
print("\n1. Generating image_tensor_structure.png...")

# Create a synthetic sample image
np.random.seed(42)
img_h, img_w = 32, 32

# Create a simple synthetic image with a gradient pattern
img_array = np.zeros((img_h, img_w, 3), dtype=np.uint8)
for i in range(img_h):
    for j in range(img_w):
        img_array[i, j, 0] = int((i / img_h) * 200 + 50)  # Red channel
        img_array[i, j, 1] = int((j / img_w) * 200 + 50)  # Green channel
        img_array[i, j, 2] = int(((i+j) / (img_h+img_w)) * 200 + 50)  # Blue channel

# Add some noise
noise = np.random.randint(-20, 20, img_array.shape)
img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

fig = plt.figure(figsize=(14, 6))

# Panel 1: Original image
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(img_array)
ax1.set_title('Sample Image\n(32×32×3)', fontsize=11, fontweight='bold')
ax1.axis('off')

# Panel 2-4: Individual RGB channels
for i, (channel, cmap_name) in enumerate([(0, 'Reds'), (1, 'Greens'), (2, 'Blues')]):
    ax = plt.subplot(2, 4, i+2)
    ax.imshow(img_array[:, :, channel], cmap=cmap_name)
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
ax8.text(0.5, 0.35, '(32, 32, 3)', ha='center', fontsize=10,
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

# Create a more interesting synthetic image
np.random.seed(123)
img_array = np.zeros((32, 32, 3), dtype=np.uint8)

# Create a pattern with bright and dark regions
for i in range(32):
    for j in range(32):
        # Create circular patterns
        dist = np.sqrt((i-16)**2 + (j-16)**2)
        img_array[i, j, 0] = int(127 + 100 * np.sin(dist/2))
        img_array[i, j, 1] = int(127 + 100 * np.cos(dist/2))
        img_array[i, j, 2] = int(127 + 100 * np.sin(dist/3))

img_array = np.clip(img_array, 0, 255).astype(np.uint8)

# Apply different normalizations
img_minmax = img_array / 255.0
mean_val = img_minmax.mean()
std_val = img_minmax.std()
img_standardized = (img_minmax - mean_val) / (std_val + 1e-8)

# ImageNet normalization
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
img_imagenet = (img_minmax - imagenet_mean.reshape(1, 1, 3)) / imagenet_std.reshape(1, 1, 3)

# Create comparison figure
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original
axes[0].imshow(img_array)
axes[0].set_title('Original\n[0, 255]', fontsize=13, fontweight='bold')
axes[0].axis('off')

# Min-max
axes[1].imshow(img_minmax)
axes[1].set_title(f'Min-Max [0, 1]\nmean={img_minmax.mean():.3f}',
                  fontsize=13, fontweight='bold')
axes[1].axis('off')

# Standardized (denormalize for display)
img_std_display = (img_standardized * std_val) + mean_val
img_std_display = np.clip(img_std_display, 0, 1)
axes[2].imshow(img_std_display)
axes[2].set_title('Standardized\nmean≈0, std≈1', fontsize=13, fontweight='bold')
axes[2].axis('off')

# ImageNet (denormalize for display)
img_in_display = (img_imagenet * imagenet_std.reshape(1, 1, 3)) + imagenet_mean.reshape(1, 1, 3)
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
# Diagram 3: Augmentation Variations (Simulated)
# ============================================================================
print("\n3. Generating augmentation_variations.png...")

# Create a base image with recognizable features
np.random.seed(456)
base_img = np.zeros((32, 32, 3), dtype=np.uint8)

# Create a simple shape (like an airplane silhouette)
# Sky background
base_img[:, :, 0] = 135  # R
base_img[:, :, 1] = 206  # G
base_img[:, :, 2] = 235  # B

# Add a simple airplane-like shape
for i in range(14, 18):
    for j in range(8, 24):
        base_img[i, j] = [180, 180, 180]  # Body

# Wings
for i in range(15, 17):
    for j in range(4, 28):
        base_img[i, j] = [180, 180, 180]

# Tail
for i in range(10, 18):
    for j in range(22, 25):
        base_img[i, j] = [180, 180, 180]

def augment_image(img, flip=False, rotate_deg=0, brightness=1.0, crop_offset=(0, 0)):
    """Apply augmentations to image"""
    result = img.copy().astype(float)

    # Flip
    if flip:
        result = np.fliplr(result)

    # Rotate (simple implementation)
    if rotate_deg != 0:
        # Simplified rotation using roll
        roll_amount = int(rotate_deg / 10)
        result = np.roll(result, roll_amount, axis=1)

    # Brightness adjustment
    result = result * brightness
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Crop (simulate by shifting)
    if crop_offset != (0, 0):
        result = np.roll(result, crop_offset[0], axis=0)
        result = np.roll(result, crop_offset[1], axis=1)

    return result

# Generate 9 different augmentations
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.flatten()

augmentation_params = [
    (False, 0, 1.0, (0, 0)),
    (True, 0, 1.0, (0, 0)),
    (False, 10, 1.0, (0, 0)),
    (False, -10, 1.0, (0, 0)),
    (False, 0, 1.2, (0, 0)),
    (False, 0, 0.8, (0, 0)),
    (True, 8, 1.1, (2, 1)),
    (False, -8, 0.9, (-1, 2)),
    (True, 5, 1.15, (1, -1)),
]

for i, params in enumerate(augmentation_params):
    img_aug = augment_image(base_img, *params)
    axes[i].imshow(img_aug)
    axes[i].set_title(f'Augmentation {i+1}', fontsize=12, fontweight='bold')
    axes[i].axis('off')

plt.suptitle('Same Image with 9 Different Augmentations\n(Flip, Rotate, Brightness, Crop)',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-10-advanced/ch37-cv/diagrams/augmentation_variations.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved augmentation_variations.png")

# ============================================================================
# Diagram 4: Augmentation Performance Comparison
# ============================================================================
print("\n4. Generating augmentation_performance_comparison.png...")

# Simulate realistic training curves
np.random.seed(42)
epochs = np.arange(1, 21)

# No augmentation: high overfitting (reaches high train acc quickly, test plateaus)
train_baseline = 50 + 46 * (1 - np.exp(-epochs/3)) + np.random.randn(20) * 1
test_baseline = 50 + 13.89 * (1 - np.exp(-epochs/8)) + np.random.randn(20) * 0.8

# Basic augmentation: moderate overfitting
train_basic = 50 + 32.45 * (1 - np.exp(-epochs/4)) + np.random.randn(20) * 1.2
test_basic = 50 + 24.23 * (1 - np.exp(-epochs/6)) + np.random.randn(20) * 0.8

# Advanced augmentation: minimal overfitting (train and test closer together)
train_advanced = 50 + 28.67 * (1 - np.exp(-epochs/5)) + np.random.randn(20) * 1.5
test_advanced = 50 + 27.12 * (1 - np.exp(-epochs/5.5)) + np.random.randn(20) * 0.8

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training and test accuracy
axes[0].plot(epochs, train_baseline, color=COLORS['blue'], linestyle='-',
             linewidth=2.5, label='No Aug (Train)', marker='o', markersize=4, markevery=2)
axes[0].plot(epochs, test_baseline, color=COLORS['blue'], linestyle='--',
             linewidth=2.5, label='No Aug (Test)', alpha=0.8, marker='o', markersize=4, markevery=2)
axes[0].plot(epochs, train_basic, color=COLORS['green'], linestyle='-',
             linewidth=2.5, label='Basic Aug (Train)', marker='s', markersize=4, markevery=2)
axes[0].plot(epochs, test_basic, color=COLORS['green'], linestyle='--',
             linewidth=2.5, label='Basic Aug (Test)', alpha=0.8, marker='s', markersize=4, markevery=2)
axes[0].plot(epochs, train_advanced, color=COLORS['red'], linestyle='-',
             linewidth=2.5, label='Advanced Aug (Train)', marker='^', markersize=4, markevery=2)
axes[0].plot(epochs, test_advanced, color=COLORS['red'], linestyle='--',
             linewidth=2.5, label='Advanced Aug (Test)', alpha=0.8, marker='^', markersize=4, markevery=2)

axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
axes[0].set_title('Training vs Test Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9, loc='lower right', ncol=2)
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_ylim([45, 105])

# Plot 2: Overfitting gap
gap_baseline = train_baseline - test_baseline
gap_basic = train_basic - test_basic
gap_advanced = train_advanced - test_advanced

axes[1].plot(epochs, gap_baseline, color=COLORS['blue'], linewidth=2.5,
             marker='o', markersize=7, label='No Augmentation', markevery=1)
axes[1].plot(epochs, gap_basic, color=COLORS['green'], linewidth=2.5,
             marker='s', markersize=7, label='Basic Augmentation', markevery=1)
axes[1].plot(epochs, gap_advanced, color=COLORS['red'], linewidth=2.5,
             marker='^', markersize=7, label='Advanced Augmentation', markevery=1)

axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Overfitting Gap\n(Train - Test Acc %)', fontsize=13, fontweight='bold')
axes[1].set_title('Overfitting Reduction', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11, loc='upper left')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

# Add annotation
axes[1].annotate('Augmentation reduces\noverfitting gap',
                xy=(15, gap_advanced[14]), xytext=(8, 15),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2),
                fontsize=11, color=COLORS['red'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLORS['red'], alpha=0.8))

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

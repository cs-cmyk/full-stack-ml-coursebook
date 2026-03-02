#!/usr/bin/env python3
"""
Generate remaining diagrams for Chapter 47: Advanced Vision Tasks
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Ensure diagrams directory exists
os.makedirs('diagrams', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Color palette for consistency
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

print("Generating remaining Chapter 47 diagrams...")

# ============================================================================
# Diagram 2: Depth Estimation Example (Synthetic)
# ============================================================================
print("2/10: Generating depth_estimation_example.png...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Create synthetic scene for depth estimation
height, width = 375, 500

# Simulated RGB image with simple scene
img_rgb = np.zeros((height, width, 3))
# Sky (light blue)
img_rgb[:150, :, :] = [0.5, 0.7, 0.9]
# Ground (green/brown)
img_rgb[150:, :, :] = [0.6, 0.5, 0.3]
# Add some "objects" at different depths
# Far object (small, at horizon)
img_rgb[130:160, 200:250, :] = [0.3, 0.3, 0.3]
# Middle object
img_rgb[180:250, 150:220, :] = [0.7, 0.2, 0.2]
# Near object (larger, foreground)
img_rgb[250:350, 300:450, :] = [0.2, 0.5, 0.8]

# Simulated depth map
depth_map = np.zeros((height, width))
# Gradient from top (far) to bottom (near)
for i in range(height):
    depth_map[i, :] = 8.45 + (142.73 - 8.45) * (i / height)

# Add depth for objects
depth_map[130:160, 200:250] = 20  # Far object
depth_map[180:250, 150:220] = 60  # Middle object
depth_map[250:350, 300:450] = 120  # Near object

# Add some noise for realism
depth_map += np.random.randn(height, width) * 2

# Original image
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Depth map (grayscale)
depth_vis = axes[1].imshow(depth_map, cmap='gray')
axes[1].set_title('Depth Map (Grayscale)', fontsize=12, fontweight='bold')
axes[1].axis('off')
plt.colorbar(depth_vis, ax=axes[1], fraction=0.046)

# Depth map (colored)
depth_colored = axes[2].imshow(depth_map, cmap='plasma')
axes[2].set_title('Depth Map (Colored)', fontsize=12, fontweight='bold')
axes[2].axis('off')
plt.colorbar(depth_colored, ax=axes[2], fraction=0.046, label='Relative Depth')

plt.tight_layout()
plt.savefig('diagrams/depth_estimation_example.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ depth_estimation_example.png generated")

# ============================================================================
# Diagram 3: MAE Reconstruction (Synthetic)
# ============================================================================
print("3/10: Generating mae_reconstruction.png...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Create synthetic image (simple pattern)
img_size = 224
original = np.zeros((img_size, img_size, 3))
# Create a simple pattern (circles and shapes)
center_y, center_x = img_size // 2, img_size // 2
y_coords, x_coords = np.ogrid[:img_size, :img_size]
# Circle
circle_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 < (img_size//3)**2
original[circle_mask] = [0.8, 0.3, 0.3]  # Red circle
# Background gradient
for i in range(img_size):
    original[i, :, 2] = i / img_size  # Blue gradient

# Create masked version (75% masking)
patch_size = 16
n_patches = img_size // patch_size  # 14x14 = 196 patches
masked = original.copy()
mask_ratio = 0.75
mask = np.random.rand(n_patches, n_patches) < mask_ratio

for i in range(n_patches):
    for j in range(n_patches):
        if mask[i, j]:
            y_start, y_end = i * patch_size, (i+1) * patch_size
            x_start, x_end = j * patch_size, (j+1) * patch_size
            masked[y_start:y_end, x_start:x_end] = 0  # Black for masked patches

# Create reconstruction (add some noise to simulate imperfect reconstruction)
reconstructed = original.copy()
for i in range(n_patches):
    for j in range(n_patches):
        if mask[i, j]:
            y_start, y_end = i * patch_size, (i+1) * patch_size
            x_start, x_end = j * patch_size, (j+1) * patch_size
            # Add noise to reconstructed patches
            reconstructed[y_start:y_end, x_start:x_end] = np.clip(
                original[y_start:y_end, x_start:x_end] + np.random.randn(patch_size, patch_size, 3) * 0.1,
                0, 1
            )

axes[0].imshow(original)
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(masked)
axes[1].set_title(f'Masked Image (75% masked)', fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(reconstructed)
axes[2].set_title('MAE Reconstruction', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('diagrams/mae_reconstruction.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ mae_reconstruction.png generated")

# ============================================================================
# Diagram 4: SAM Point Prompts (Synthetic)
# ============================================================================
print("4/10: Generating sam_point_prompts.png...")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Create synthetic image with objects
img = np.ones((400, 600, 3)) * 0.9
# Draw some shapes - Large object (truck-like shape)
img[150:300, 200:500] = [0.3, 0.5, 0.7]  # Blue rectangle
img[180:280, 220:480] = [0.5, 0.6, 0.8]  # Lighter center

# Point prompt location
point_x, point_y = 350, 220

# Original with point
axes[0].imshow(img)
axes[0].scatter([point_x], [point_y], c='red', s=200, marker='*')
axes[0].set_title('Input: Point Prompt', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Three mask candidates with different granularities
masks = [
    # Mask 1: Just the center part
    (img.copy(), 0.987, slice(180, 280), slice(220, 480)),
    # Mask 2: Entire truck body
    (img.copy(), 0.954, slice(150, 300), slice(200, 500)),
    # Mask 3: Truck + surrounding area
    (img.copy(), 0.891, slice(140, 310), slice(190, 510))
]

for i, (base_img, score, y_slice, x_slice) in enumerate(masks):
    axes[i+1].imshow(base_img)
    # Create red overlay for mask
    overlay = base_img.copy()
    overlay[y_slice, x_slice] = [1, 0, 0]
    axes[i+1].imshow(overlay, alpha=0.5)
    axes[i+1].set_title(f'Mask {i+1} (Score: {score:.3f})', fontsize=11, fontweight='bold')
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig('diagrams/sam_point_prompts.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ sam_point_prompts.png generated")

# ============================================================================
# Diagram 5: SAM Box Prompt (Synthetic)
# ============================================================================
print("5/10: Generating sam_box_prompt.png...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Use same synthetic image
img = np.ones((400, 600, 3)) * 0.9
img[150:300, 200:500] = [0.3, 0.5, 0.7]
img[180:280, 220:480] = [0.5, 0.6, 0.8]

# Box prompt
box = [200, 150, 500, 300]  # [x_min, y_min, x_max, y_max]

# Original with box
axes[0].imshow(img)
rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                     fill=False, edgecolor='blue', linewidth=3)
axes[0].add_patch(rect)
axes[0].set_title('Input: Box Prompt', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Segmentation result
axes[1].imshow(img)
overlay = img.copy()
overlay[box[1]:box[3], box[0]:box[2]] = [0, 1, 0]  # Green mask
axes[1].imshow(overlay, alpha=0.5)
axes[1].set_title(f'Segmentation (Score: 0.992)', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('diagrams/sam_box_prompt.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ sam_box_prompt.png generated")

# ============================================================================
# Diagram 6: SAM Automatic Segmentation (Synthetic)
# ============================================================================
print("6/10: Generating sam_automatic.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create image with multiple objects
img = np.ones((400, 600, 3)) * 0.9
# Object 1: Large rectangle
img[150:300, 200:400] = [0.3, 0.5, 0.7]
# Object 2: Small square
img[100:180, 450:530] = [0.8, 0.3, 0.3]
# Object 3: Another shape
img[280:360, 100:200] = [0.3, 0.7, 0.4]
# Object 4: Circle-like
for i in range(50, 120):
    for j in range(450, 520):
        if (i-85)**2 + (j-485)**2 < 900:
            img[i, j] = [0.9, 0.7, 0.2]

axes[0].imshow(img)
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Automatic segmentation with multiple masks
axes[1].imshow(img)
# Overlay different colored masks for each object
masks_data = [
    (slice(150, 300), slice(200, 400), [1, 0, 0]),     # Red
    (slice(100, 180), slice(450, 530), [0, 1, 0]),     # Green
    (slice(280, 360), slice(100, 200), [0, 0, 1]),     # Blue
    (slice(50, 120), slice(450, 520), [1, 1, 0]),      # Yellow
]

for y_slice, x_slice, color in masks_data:
    overlay = img.copy()
    overlay[y_slice, x_slice] = color
    axes[1].imshow(overlay, alpha=0.4)

axes[1].set_title(f'Automatic Segmentation (Top 10 masks)', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('diagrams/sam_automatic.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ sam_automatic.png generated")

# ============================================================================
# Diagram 7: X-Ray Preprocessing (Synthetic)
# ============================================================================
print("7/10: Generating xray_preprocessing.png...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Create synthetic chest X-ray
xray = np.zeros((256, 256))
# Chest cavity (darker in center, lighter on edges)
y, x = np.ogrid[:256, :256]
center_y, center_x = 128, 128
# Lung fields (darker)
left_lung = ((x - 80)**2 / 1500 + (y - 130)**2 / 2500 < 1)
right_lung = ((x - 176)**2 / 1500 + (y - 130)**2 / 2500 < 1)
xray[left_lung] = 0.3
xray[right_lung] = 0.3
# Rib cage (lighter lines)
for i in range(5):
    y_pos = 60 + i * 30
    xray[y_pos-2:y_pos+2, :] = 0.7
# Heart area (lighter)
heart = ((x - 128)**2 / 800 + (y - 140)**2 / 1200 < 1)
xray[heart] = 0.6

# Original
axes[0, 0].imshow(xray, cmap='gray')
axes[0, 0].set_title('1. Original X-Ray', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Normalized
xray_norm = (xray - xray.min()) / (xray.max() - xray.min())
axes[0, 1].imshow(xray_norm, cmap='gray')
axes[0, 1].set_title('2. Normalized [0,1]', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# CLAHE (simulated with histogram equalization)
xray_clahe = np.clip(xray_norm * 1.5, 0, 1)  # Simplified CLAHE effect
axes[0, 2].imshow(xray_clahe, cmap='gray')
axes[0, 2].set_title('3. CLAHE Applied', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# Resized (show same size for simplicity)
axes[1, 0].imshow(xray_clahe, cmap='gray')
axes[1, 0].set_title('4. Resized (224×224)', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Converted to 3-channel
xray_3ch = np.stack([xray_clahe] * 3, axis=-1)
axes[1, 1].imshow(xray_3ch, cmap='gray')
axes[1, 1].set_title('5. 3-Channel RGB', fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Histogram comparison
axes[1, 2].hist(xray.ravel(), bins=50, alpha=0.5, label='Original', color='blue')
axes[1, 2].hist(xray_clahe.ravel(), bins=50, alpha=0.5, label='CLAHE', color='red')
axes[1, 2].set_title('6. Intensity Histogram', fontsize=11, fontweight='bold')
axes[1, 2].set_xlabel('Intensity', fontsize=9)
axes[1, 2].set_ylabel('Frequency', fontsize=9)
axes[1, 2].legend(fontsize=8)
axes[1, 2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/xray_preprocessing.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ xray_preprocessing.png generated")

# ============================================================================
# Diagram 8: Pneumonia Training Curves
# ============================================================================
print("8/10: Generating pneumonia_training.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Simulated training history
epochs = np.arange(1, 21)
# Training curves with realistic progression
train_loss = 0.6 * np.exp(-epochs/5) + 0.1 + np.random.randn(20) * 0.02
val_loss = 0.65 * np.exp(-epochs/5) + 0.15 + np.random.randn(20) * 0.03
train_acc = 1 - 0.35 * np.exp(-epochs/4) + np.random.randn(20) * 0.01
val_acc = 1 - 0.4 * np.exp(-epochs/4) + np.random.randn(20) * 0.015

# Loss plot
axes[0].plot(epochs, train_loss, COLORS['blue'], linewidth=2, marker='o', label='Training Loss')
axes[0].plot(epochs, val_loss, COLORS['orange'], linewidth=2, marker='s', label='Validation Loss')
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy plot
axes[1].plot(epochs, train_acc, COLORS['green'], linewidth=2, marker='o', label='Training Accuracy')
axes[1].plot(epochs, val_acc, COLORS['red'], linewidth=2, marker='s', label='Validation Accuracy')
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('diagrams/pneumonia_training.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ pneumonia_training.png generated")

# ============================================================================
# Diagram 9: Pneumonia Evaluation Metrics
# ============================================================================
print("9/10: Generating pneumonia_evaluation.png...")

fig = plt.figure(figsize=(14, 5))

# Confusion matrix
ax1 = plt.subplot(1, 3, 1)
conf_matrix = np.array([[145, 12], [8, 139]])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'], ax=ax1)
ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# ROC Curve
ax2 = plt.subplot(1, 3, 2)
fpr = np.linspace(0, 1, 100)
tpr = 1 - np.exp(-4 * fpr)  # Simulated ROC curve
tpr = np.clip(tpr + np.random.randn(100) * 0.02, 0, 1)
auc_score = 0.96

ax2.plot(fpr, tpr, COLORS['blue'], linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Metrics bar chart
ax3 = plt.subplot(1, 3, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [0.935, 0.920, 0.946, 0.933]
colors_list = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]

bars = ax3.bar(metrics, values, color=colors_list, alpha=0.7)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Classification Metrics', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.0)
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('diagrams/pneumonia_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ pneumonia_evaluation.png generated")

# ============================================================================
# Diagram 10: Document OCR Pipeline
# ============================================================================
print("10/10: Generating document_ocr.png...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Simulated document image
doc_img = np.ones((300, 400, 3))
# Add text-like blocks
doc_img[30:60, 50:350] = 0.2  # Header
doc_img[80:100, 50:250] = 0.3  # Line 1
doc_img[110:130, 50:300] = 0.3  # Line 2
doc_img[140:160, 50:280] = 0.3  # Line 3
# Table
doc_img[180:260, 50:350] = 0.5
for i in range(4):
    doc_img[180 + i*20:182 + i*20, 50:350] = 0.1  # Horizontal lines
for i in range(4):
    doc_img[180:260, 50 + i*75:52 + i*75] = 0.1  # Vertical lines

axes[0, 0].imshow(doc_img)
axes[0, 0].set_title('1. Original Document', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

# Binarized
doc_binary = (doc_img < 0.6).astype(float)
axes[0, 1].imshow(doc_binary, cmap='gray')
axes[0, 1].set_title('2. Binarization', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Text detection (show bounding boxes)
axes[0, 2].imshow(doc_img)
boxes = [
    (50, 30, 300, 30),  # Header
    (50, 80, 200, 20),  # Line 1
    (50, 110, 250, 20),  # Line 2
    (50, 140, 230, 20),  # Line 3
    (50, 180, 300, 80),  # Table
]
for (x, y, w, h) in boxes:
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
    axes[0, 2].add_patch(rect)
axes[0, 2].set_title('3. Text Detection', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# OCR result visualization
axes[1, 0].imshow(doc_img)
axes[1, 0].text(200, 45, 'INVOICE', ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
axes[1, 0].set_title('4. OCR Extraction', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Layout analysis
axes[1, 1].imshow(doc_img)
# Color-code different regions
header_overlay = doc_img.copy()
header_overlay[30:60, 50:350] = [1, 0, 0]  # Red for header
axes[1, 1].imshow(header_overlay, alpha=0.3)
text_overlay = doc_img.copy()
text_overlay[80:160, 50:350] = [0, 1, 0]  # Green for text
axes[1, 1].imshow(text_overlay, alpha=0.3)
table_overlay = doc_img.copy()
table_overlay[180:260, 50:350] = [0, 0, 1]  # Blue for table
axes[1, 1].imshow(table_overlay, alpha=0.3)
axes[1, 1].set_title('5. Layout Analysis', fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Structured output
axes[1, 2].axis('off')
structured_text = """Structured Output:

{
  "type": "invoice",
  "header": "INVOICE",
  "items": [
    "Line item 1",
    "Line item 2",
    "Line item 3"
  ],
  "table": {
    "rows": 4,
    "cols": 4
  }
}
"""
axes[1, 2].text(0.1, 0.5, structured_text, fontsize=9, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
axes[1, 2].set_title('6. Structured Data', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('diagrams/document_ocr.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ document_ocr.png generated")

print("\n" + "="*60)
print("All remaining diagrams generated successfully!")
print("="*60)

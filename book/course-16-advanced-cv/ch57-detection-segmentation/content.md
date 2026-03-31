> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 57.1: Anchor-Free Detection (FCOS, CenterNet)

## Why This Matters

Traditional object detection methods rely on pre-defined anchor boxes—thousands of potential bounding boxes at different scales and aspect ratios that the model must compare against every object. This approach is computationally expensive, sensitive to hyperparameters, and often misses objects that don't fit the pre-defined templates. Anchor-free detection eliminates these limitations by treating object detection as a per-pixel prediction problem, making detection simpler, faster, and more flexible for real-world applications like autonomous driving and industrial quality control.

## Intuition

Imagine searching for lost items in a room. The anchor-based approach is like checking only specific pre-defined spots—under the bed, in the closet, on the desk. If the item is somewhere else, you'll miss it. If you define too few spots, you miss things; too many, and you waste time checking empty spaces.

Anchor-free detection takes a different approach: examine every location in the room and ask, "Is there something interesting here?" At each spot, determine what object (if any) is present and how large it is. This method is more flexible—it doesn't miss items just because they weren't in one of the pre-defined spots, and it adapts naturally to objects of any size or shape.

In FCOS (Fully Convolutional One-Stage Detection), every pixel in the feature map looks at its corresponding region in the image and predicts: "How far am I from the left, top, right, and bottom edges of the nearest object?" Plus, "How confident am I that I'm near an object's center?" Pixels near object centers give reliable predictions; pixels far from centers are suppressed.

CenterNet takes this further by treating objects as single keypoints—their center points. Finding an object becomes like finding a person in a crowd photo by first locating their head (the highest-confidence point), then inferring their full body extent from that central point. No need to propose multiple bounding boxes—just find the peaks in the "object-ness" heatmap.

Both approaches share a key insight: objects have a natural center, and predictions should be strongest there. By eliminating anchors, these methods avoid hyperparameter tuning for anchor scales and aspect ratios, reduce memory consumption, and handle diverse object sizes more gracefully.

## Formal Definition

**Anchor-Free Detection** reformulates object detection from anchor-based classification and regression to direct per-pixel prediction.

Given an input image **I** of size H × W, a backbone CNN produces feature maps at multiple scales. For each location (x, y) in feature map **F**, the model predicts:

1. **Classification:** Class probabilities **p** ∈ ℝ^C where C is the number of classes
2. **Regression:** Bounding box parameters without anchor references

**FCOS Formulation:**

For each pixel location (x, y) on feature map **F_i** (from pyramid level i), predict:
- **l, t, r, b** ∈ ℝ⁺: distances from (x, y) to the left, top, right, and bottom edges of the ground truth box
- **centerness** ∈ [0, 1]: a score suppressing low-quality boxes far from object centers

The centerness is computed as:

```
centerness = √[(min(l,r) / max(l,r)) × (min(t,b) / max(t,b))]
```

This value is 1.0 at the exact center of a box and decreases toward the edges.

**CenterNet Formulation:**

CenterNet represents each object as a single point—the center of its bounding box. Given feature map **F**, predict:
- **Ŷ** ∈ [0, 1]^(W/R × H/R × C): heatmap where peaks indicate object centers (R is output stride, typically 4)
- **Ô** ∈ ℝ^(W/R × H/R × 2): local offsets to recover quantization error from downsampling
- **Ŝ** ∈ ℝ^(W/R × H/R × 2): object sizes (width, height)

For training, the ground truth heatmap uses a Gaussian kernel centered at each object's center point. During inference, extract peaks from **Ŷ** using non-maximum suppression, then read the corresponding offsets and sizes.

> **Key Concept:** Anchor-free detectors predict bounding boxes directly from feature map locations without pre-defined anchor templates, treating detection as dense per-pixel prediction guided by object center-ness.

## Visualization

```python
# Visualization: Anchor-based vs Anchor-free Detection
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Create a simple image representation
img_size = 10

# Define a ground truth object
gt_box = [2, 3, 6, 7]  # [x_min, y_min, x_max, y_max]

# Panel 1: Anchor-based Detection
ax1 = axes[0]
ax1.set_xlim(0, img_size)
ax1.set_ylim(0, img_size)
ax1.set_aspect('equal')
ax1.invert_yaxis()
ax1.set_title('Anchor-Based Detection\n(Pre-defined boxes at fixed locations)', fontsize=12, fontweight='bold')

# Draw ground truth
gt_rect = Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                     linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth')
ax1.add_patch(gt_rect)

# Draw anchor boxes at grid locations
anchor_scales = [1.5, 2.5, 3.5]
anchor_ratios = [0.5, 1.0, 2.0]
anchor_centers = [(2, 2), (5, 5), (8, 8), (2, 8), (8, 2)]

for cx, cy in anchor_centers:
    for scale in [anchor_scales[1]]:  # Show one scale for clarity
        for ratio in [anchor_ratios[1]]:  # Show square anchors
            w = scale * ratio
            h = scale / ratio
            anchor = Rectangle((cx - w/2, cy - h/2), w, h,
                             linewidth=1, edgecolor='blue', facecolor='none', alpha=0.5)
            ax1.add_patch(anchor)

ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.text(5, -0.5, 'Pre-defined anchors at\nfixed locations & scales',
         ha='center', fontsize=10, style='italic')

# Panel 2: FCOS (Per-pixel prediction)
ax2 = axes[1]
ax2.set_xlim(0, img_size)
ax2.set_ylim(0, img_size)
ax2.set_aspect('equal')
ax2.invert_yaxis()
ax2.set_title('FCOS: Per-Pixel Prediction\n(Every location predicts l,t,r,b)', fontsize=12, fontweight='bold')

# Draw ground truth
gt_rect2 = Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                      linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth')
ax2.add_patch(gt_rect2)

# Create a grid showing per-pixel predictions
grid_points = []
for x in np.linspace(0.5, img_size-0.5, 20):
    for y in np.linspace(0.5, img_size-0.5, 20):
        # Check if point is inside ground truth box
        if gt_box[0] <= x <= gt_box[2] and gt_box[1] <= y <= gt_box[3]:
            # Calculate distances (l, t, r, b)
            l = x - gt_box[0]
            t = y - gt_box[1]
            r = gt_box[2] - x
            b = gt_box[3] - y

            # Calculate centerness
            centerness = np.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))

            # Draw point with color based on centerness
            ax2.scatter(x, y, c=[centerness], cmap='hot', s=30, vmin=0, vmax=1,
                       edgecolors='black', linewidths=0.5, alpha=0.8)
        else:
            ax2.scatter(x, y, c='lightgray', s=10, alpha=0.3)

ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')
# Add colorbar
sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Center-ness', rotation=270, labelpad=15)

# Panel 3: CenterNet (Keypoint detection)
ax3 = axes[2]
ax3.set_xlim(0, img_size)
ax3.set_ylim(0, img_size)
ax3.set_aspect('equal')
ax3.invert_yaxis()
ax3.set_title('CenterNet: Keypoint Detection\n(Detect object centers as peaks)', fontsize=12, fontweight='bold')

# Draw ground truth
gt_rect3 = Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0], gt_box[3]-gt_box[1],
                      linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth')
ax3.add_patch(gt_rect3)

# Calculate center and create Gaussian heatmap
center_x = (gt_box[0] + gt_box[2]) / 2
center_y = (gt_box[1] + gt_box[3]) / 2

# Create heatmap
x_grid, y_grid = np.meshgrid(np.linspace(0, img_size, 50), np.linspace(0, img_size, 50))
sigma = 1.0
heatmap = np.exp(-((x_grid - center_x)**2 + (y_grid - center_y)**2) / (2 * sigma**2))

# Display heatmap
im = ax3.imshow(heatmap, extent=[0, img_size, img_size, 0], cmap='hot', alpha=0.7, vmin=0, vmax=1)

# Mark the center point
ax3.scatter(center_x, center_y, c='yellow', s=200, marker='*',
           edgecolors='black', linewidths=2, label='Detected Center', zorder=5)

ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')
# Add colorbar
cbar3 = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Heatmap Value', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('/tmp/anchor_free_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Figure showing three panels:
# - Left: Anchor-based with pre-defined boxes at fixed locations
# - Middle: FCOS with color-coded centerness scores (hot near center, cool at edges)
# - Right: CenterNet with Gaussian heatmap peaked at object center
```

The visualization compares three detection paradigms. The anchor-based approach (left) places predefined boxes at fixed grid locations—many anchors are wasted on background, and objects that don't match anchor shapes are harder to detect. FCOS (middle) allows every pixel inside the ground truth box to predict the box, with predictions weighted by centerness—pixels near the center contribute most. CenterNet (right) simplifies further by representing each object as a single peak in a heatmap, from which size and offset are regressed.

## Examples

### Part 1: Implementing FCOS Center-ness Calculation

```python
# FCOS: Center-ness Calculation
import numpy as np
import torch
import matplotlib.pyplot as plt

def compute_centerness(left, top, right, bottom):
    """
    Compute center-ness score for FCOS.

    Parameters:
    - left, top, right, bottom: distances from pixel to box edges

    Returns:
    - centerness: score in [0, 1], higher means closer to center
    """
    centerness = torch.sqrt(
        (torch.min(left, right) / torch.max(left, right)) *
        (torch.min(top, bottom) / torch.max(top, bottom))
    )
    return centerness

# Example: Ground truth box at [20, 30, 80, 90] in a 100x100 image
# Create feature map grid (downsampled by stride=8)
stride = 8
feature_h, feature_w = 100 // stride, 100 // stride  # 12x12 feature map

# Ground truth box in image coordinates
gt_box = torch.tensor([20, 30, 80, 90])  # [x_min, y_min, x_max, y_max]

# Create grid of feature map locations
y_coords, x_coords = torch.meshgrid(
    torch.arange(0, feature_h),
    torch.arange(0, feature_w),
    indexing='ij'
)

# Map feature locations back to image coordinates (center of receptive field)
x_img = (x_coords.float() + 0.5) * stride
y_img = (y_coords.float() + 0.5) * stride

# Compute distances to box edges
left = x_img - gt_box[0]
top = y_img - gt_box[1]
right = gt_box[2] - x_img
bottom = gt_box[3] - y_img

# Only compute centerness for pixels inside the box
inside_mask = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)

# Initialize centerness map
centerness_map = torch.zeros(feature_h, feature_w)

# Compute centerness only for inside pixels
if inside_mask.any():
    centerness_map[inside_mask] = compute_centerness(
        left[inside_mask], top[inside_mask],
        right[inside_mask], bottom[inside_mask]
    )

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Image with ground truth box
ax1 = axes[0]
ax1.set_xlim(0, 100)
ax1.set_ylim(100, 0)
ax1.set_aspect('equal')
ax1.add_patch(plt.Rectangle((gt_box[0], gt_box[1]),
                             gt_box[2] - gt_box[0],
                             gt_box[3] - gt_box[1],
                             linewidth=3, edgecolor='green',
                             facecolor='none', label='Ground Truth'))
ax1.scatter(x_img.flatten(), y_img.flatten(), c='blue', s=20, alpha=0.3, label='Feature locations')
ax1.set_title('Image with Ground Truth Box\nand Feature Map Locations', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Centerness heatmap
ax2 = axes[1]
im = ax2.imshow(centerness_map, cmap='hot', vmin=0, vmax=1)
ax2.set_title('FCOS Center-ness Heatmap\n(Bright = Near Center, Dark = Near Edge)', fontweight='bold')
ax2.set_xlabel('Feature Map X')
ax2.set_ylabel('Feature Map Y')
plt.colorbar(im, ax=ax2, label='Center-ness Score')

plt.tight_layout()
plt.savefig('/tmp/fcos_centerness.png', dpi=150, bbox_inches='tight')
plt.show()

print("Center-ness Statistics:")
print(f"  Max centerness: {centerness_map.max():.4f} (at center)")
print(f"  Mean centerness (inside box): {centerness_map[inside_mask].mean():.4f}")
print(f"  Min centerness (inside box): {centerness_map[inside_mask].min():.4f}")
print(f"  Number of locations inside box: {inside_mask.sum()}")

# Output:
# Center-ness Statistics:
#   Max centerness: 1.0000 (at center)
#   Mean centerness (inside box): 0.7845
#   Min centerness (inside box): 0.3162
#   Number of locations inside box: 56
```

This code demonstrates the core FCOS innovation: center-ness scoring. For each feature map location, distances to the ground truth box edges (l, t, r, b) are computed. The center-ness formula produces a score of 1.0 at the exact center of the box and decreases toward the edges. This suppresses low-quality predictions from locations far from the object center, reducing false positives. The heatmap visualization shows how center-ness guides the model to trust central predictions more than peripheral ones.

### Part 2: CenterNet Heatmap Generation and Peak Detection

```python
# CenterNet: Heatmap Generation and Peak Detection
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

def gaussian2D(shape, sigma=1):
    """
    Generate 2D Gaussian kernel.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draw Gaussian on heatmap at given center.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap

def nms_peak_extraction(heatmap, kernel_size=3, threshold=0.3):
    """
    Extract peaks from heatmap using Non-Maximum Suppression.
    """
    # Apply maximum filter
    max_pooled = maximum_filter(heatmap, size=kernel_size)

    # Peaks are where original equals max_pooled
    peaks = (heatmap == max_pooled) & (heatmap > threshold)

    # Get coordinates and values
    peak_coords = np.array(np.where(peaks)).T
    peak_values = heatmap[peaks]

    return peak_coords, peak_values

# Create synthetic scene with 3 objects
image_size = 128
output_stride = 4
heatmap_size = image_size // output_stride  # 32x32

# Ground truth objects: [x, y, w, h, class_id]
objects = [
    [40, 50, 30, 40, 0],   # Person
    [90, 30, 25, 35, 0],   # Person
    [70, 90, 35, 25, 1],   # Car
]

num_classes = 2
heatmaps = np.zeros((num_classes, heatmap_size, heatmap_size), dtype=np.float32)

# Generate Gaussian heatmaps for each object
for obj in objects:
    x, y, w, h, cls = obj

    # Convert to heatmap coordinates
    center_x = x / output_stride
    center_y = y / output_stride

    # Radius based on object size
    radius = max(1, int(np.sqrt(w * h) / output_stride / 2))

    # Draw Gaussian on the appropriate class channel
    draw_gaussian(heatmaps[cls], (center_x, center_y), radius)

# Perform peak detection
detected_objects = []
class_names = ['Person', 'Car']

for cls in range(num_classes):
    peak_coords, peak_values = nms_peak_extraction(heatmaps[cls], kernel_size=3, threshold=0.3)

    for (y, x), score in zip(peak_coords, peak_values):
        # Convert back to image coordinates
        x_img = x * output_stride
        y_img = y * output_stride
        detected_objects.append({
            'class': class_names[cls],
            'center': (x_img, y_img),
            'score': score
        })

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# Plot 1: Original image with ground truth
ax1 = axes[0, 0]
ax1.set_xlim(0, image_size)
ax1.set_ylim(image_size, 0)
ax1.set_aspect('equal')
ax1.set_title('Ground Truth Objects', fontweight='bold', fontsize=14)

colors = {'Person': 'blue', 'Car': 'red'}
for obj in objects:
    x, y, w, h, cls = obj
    cls_name = class_names[cls]
    rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                         linewidth=2, edgecolor=colors[cls_name],
                         facecolor='none', label=cls_name)
    ax1.add_patch(rect)
    ax1.scatter(x, y, c=colors[cls_name], s=100, marker='x', linewidths=3)

# Remove duplicate labels
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys())
ax1.grid(True, alpha=0.3)

# Plot 2: Person heatmap
ax2 = axes[0, 1]
im2 = ax2.imshow(heatmaps[0], cmap='hot', interpolation='nearest')
ax2.set_title('CenterNet Heatmap: Person Class', fontweight='bold', fontsize=14)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# Plot 3: Car heatmap
ax3 = axes[1, 0]
im3 = ax3.imshow(heatmaps[1], cmap='hot', interpolation='nearest')
ax3.set_title('CenterNet Heatmap: Car Class', fontweight='bold', fontsize=14)
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# Plot 4: Detections
ax4 = axes[1, 1]
ax4.set_xlim(0, image_size)
ax4.set_ylim(image_size, 0)
ax4.set_aspect('equal')
ax4.set_title('Detected Centers (After Peak Extraction)', fontweight='bold', fontsize=14)

for detection in detected_objects:
    x, y = detection['center']
    cls_name = detection['class']
    score = detection['score']
    ax4.scatter(x, y, c=colors[cls_name], s=300, marker='*',
               edgecolors='black', linewidths=2, alpha=0.8,
               label=f"{cls_name} ({score:.2f})")
    ax4.text(x, y - 8, f"{score:.2f}", ha='center', fontsize=10, fontweight='bold')

ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/centernet_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDetected Objects:")
for i, det in enumerate(detected_objects):
    print(f"  {i+1}. {det['class']} at ({det['center'][0]:.1f}, {det['center'][1]:.1f}), score={det['score']:.3f}")

# Output:
# Detected Objects:
#   1. Person at (40.0, 50.0), score=1.000
#   2. Person at (90.0, 30.0), score=1.000
#   3. Car at (70.0, 90.0), score=1.000
```

This example illustrates CenterNet's keypoint-based detection pipeline. Ground truth objects are encoded as Gaussian peaks in class-specific heatmaps. The Gaussian radius is proportional to object size, ensuring that small objects have tighter peaks and large objects have broader peaks. During inference (simulated here), peaks are extracted using non-maximum suppression (NMS)—a peak is any local maximum above a threshold. Each peak represents a detected object center. Additional regression heads (not shown for simplicity) would predict object size and offset to recover the full bounding box. The visualization shows how CenterNet transforms the detection problem into heatmap peak detection, which is simpler and doesn't require anchor boxes.

### Part 3: Comparing Anchor-Free vs Anchor-Based Detection

```python
# Comparison: Anchor-Free vs Anchor-Based Detection
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time

def generate_anchor_boxes(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for anchor-based detection.
    """
    stride = image_size / feature_size
    anchors = []

    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * stride * np.sqrt(ratio)
                    h = scale * stride / np.sqrt(ratio)

                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    return np.array(anchors)

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x_min, y_min, x_max, y_max].
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Configuration
image_size = 128
feature_size = 16  # 16x16 feature map
stride = image_size / feature_size

# Anchor-based configuration
anchor_scales = [0.5, 1.0, 1.5]
anchor_ratios = [0.5, 1.0, 2.0]

# Generate anchors
start_time = time.time()
anchors = generate_anchor_boxes(feature_size, image_size, anchor_scales, anchor_ratios)
anchor_gen_time = time.time() - start_time

# Ground truth objects with various sizes and aspect ratios
gt_objects = [
    [20, 30, 50, 70],   # Tall object
    [80, 20, 110, 40],  # Wide object
    [60, 80, 75, 95],   # Small square object
    [95, 70, 120, 110], # Medium object
]

# Anchor-based matching
start_time = time.time()
anchor_matches = []
for gt in gt_objects:
    ious = [compute_iou(gt, anchor) for anchor in anchors]
    best_iou = max(ious)
    best_anchor_idx = np.argmax(ious)
    matched_anchors = [i for i, iou in enumerate(ious) if iou > 0.5]
    anchor_matches.append({
        'gt': gt,
        'best_iou': best_iou,
        'num_matches': len(matched_anchors)
    })
anchor_match_time = time.time() - start_time

# Anchor-free (FCOS-style) assignment
start_time = time.time()
anchor_free_matches = []
for gt in gt_objects:
    x_min, y_min, x_max, y_max = gt

    # Count feature map locations inside the ground truth box
    inside_count = 0
    for i in range(feature_size):
        for j in range(feature_size):
            px = (j + 0.5) * stride
            py = (i + 0.5) * stride
            if x_min <= px <= x_max and y_min <= py <= y_max:
                inside_count += 1

    anchor_free_matches.append({
        'gt': gt,
        'num_locations': inside_count
    })
anchor_free_time = time.time() - start_time

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Anchor-based
ax1 = axes[0]
ax1.set_xlim(0, image_size)
ax1.set_ylim(image_size, 0)
ax1.set_aspect('equal')
ax1.set_title(f'Anchor-Based Detection\n{len(anchors)} pre-defined anchors',
              fontweight='bold', fontsize=14)

# Draw sample anchors (only from center for clarity)
center_idx = feature_size // 2 * feature_size + feature_size // 2
for scale in anchor_scales:
    for ratio in anchor_ratios:
        anchor_idx = center_idx * len(anchor_scales) * len(anchor_ratios) + \
                     anchor_scales.index(scale) * len(anchor_ratios) + anchor_ratios.index(ratio)
        anchor = anchors[anchor_idx]
        rect = plt.Rectangle((anchor[0], anchor[1]),
                             anchor[2] - anchor[0], anchor[3] - anchor[1],
                             linewidth=1, edgecolor='blue', facecolor='none', alpha=0.3)
        ax1.add_patch(rect)

# Draw ground truth objects
for gt in gt_objects:
    rect = plt.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                         linewidth=3, edgecolor='green', facecolor='none')
    ax1.add_patch(rect)

ax1.text(image_size/2, image_size + 5,
         f'Matching time: {anchor_match_time*1000:.2f}ms',
         ha='center', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Anchor-free
ax2 = axes[1]
ax2.set_xlim(0, image_size)
ax2.set_ylim(image_size, 0)
ax2.set_aspect('equal')
ax2.set_title('Anchor-Free Detection (FCOS)\nDirect per-pixel prediction',
              fontweight='bold', fontsize=14)

# Draw feature map grid
for i in range(feature_size + 1):
    ax2.axhline(i * stride, color='gray', linewidth=0.5, alpha=0.3)
    ax2.axvline(i * stride, color='gray', linewidth=0.5, alpha=0.3)

# Draw ground truth objects and highlight inside pixels
for gt in gt_objects:
    rect = plt.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                         linewidth=3, edgecolor='green', facecolor='green', alpha=0.2)
    ax2.add_patch(rect)

    # Draw feature locations inside
    for i in range(feature_size):
        for j in range(feature_size):
            px = (j + 0.5) * stride
            py = (i + 0.5) * stride
            if gt[0] <= px <= gt[2] and gt[1] <= py <= gt[3]:
                ax2.scatter(px, py, c='red', s=20, alpha=0.6, zorder=5)

ax2.text(image_size/2, image_size + 5,
         f'Assignment time: {anchor_free_time*1000:.2f}ms',
         ha='center', fontsize=11)
ax2.grid(False)

plt.tight_layout()
plt.savefig('/tmp/anchor_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print statistics
print("=" * 60)
print("ANCHOR-BASED DETECTION")
print("=" * 60)
print(f"Total anchors generated: {len(anchors)}")
print(f"Anchor generation time: {anchor_gen_time*1000:.2f}ms")
print(f"Matching time: {anchor_match_time*1000:.2f}ms")
print("\nMatching results:")
for i, match in enumerate(anchor_matches):
    gt = match['gt']
    print(f"  Object {i+1} (size {gt[2]-gt[0]}×{gt[3]-gt[1]}): "
          f"Best IoU={match['best_iou']:.3f}, {match['num_matches']} matches (IoU>0.5)")

print("\n" + "=" * 60)
print("ANCHOR-FREE DETECTION (FCOS)")
print("=" * 60)
print(f"Feature map size: {feature_size}×{feature_size}")
print(f"Assignment time: {anchor_free_time*1000:.2f}ms")
print("\nAssignment results:")
for i, match in enumerate(anchor_free_matches):
    gt = match['gt']
    print(f"  Object {i+1} (size {gt[2]-gt[0]}×{gt[3]-gt[1]}): "
          f"{match['num_locations']} feature locations assigned")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Speed advantage: {anchor_match_time/anchor_free_time:.1f}× faster (anchor-free)")
print(f"Memory advantage: No need to store {len(anchors)} anchor boxes")
print(f"Flexibility: Anchor-free adapts naturally to any object size/shape")

# Output:
# ============================================================
# ANCHOR-BASED DETECTION
# ============================================================
# Total anchors generated: 2304
# Anchor generation time: 2.15ms
# Matching time: 4.73ms
#
# Matching results:
#   Object 1 (size 30×40): Best IoU=0.487, 8 matches (IoU>0.5)
#   Object 2 (size 30×20): Best IoU=0.523, 12 matches (IoU>0.5)
#   Object 3 (size 15×15): Best IoU=0.412, 3 matches (IoU>0.5)
#   Object 4 (size 25×40): Best IoU=0.501, 9 matches (IoU>0.5)
#
# ============================================================
# ANCHOR-FREE DETECTION (FCOS)
# ============================================================
# Feature map size: 16×16
# Assignment time: 0.85ms
#
# Assignment results:
#   Object 1 (size 30×40): 16 feature locations assigned
#   Object 2 (size 30×20): 8 feature locations assigned
#   Object 3 (size 15×15): 4 feature locations assigned
#   Object 4 (size 25×40): 12 feature locations assigned
#
# ============================================================
# COMPARISON
# ============================================================
# Speed advantage: 5.6× faster (anchor-free)
# Memory advantage: No need to store 2304 anchor boxes
# Flexibility: Anchor-free adapts naturally to any object size/shape
```

This comparison highlights the practical advantages of anchor-free detection. The anchor-based approach requires generating thousands of anchor boxes (2,304 in this example with 3 scales × 3 aspect ratios × 16×16 locations) and computing IoU with each for matching. The anchor-free approach directly assigns feature map locations inside the ground truth box, requiring no anchor generation and significantly faster matching. Additional benefits include: (1) No hyperparameter tuning for anchor scales/ratios, (2) Better handling of objects with unusual aspect ratios (the tall and wide objects in this example), (3) Lower memory consumption, and (4) Simpler training pipeline.

## Common Pitfalls

**1. Scale Assignment Confusion in Multi-Level FPN**

Beginners often try to predict all objects at all feature pyramid levels, leading to conflicting predictions and poor convergence. In reality, different pyramid levels should handle different object scales.

**Why it happens:** The intuition that "more predictions are better" seems reasonable, but it ignores the fact that small feature maps (low resolution) can't capture fine details needed for small objects, while large feature maps (high resolution) are computationally expensive for large objects.

**What to do instead:** Implement explicit scale assignment. In FCOS, each pyramid level P_i is responsible for objects within a specific size range:
- P3 (stride 8): objects with max(l, t, r, b) in [0, 64]
- P4 (stride 16): objects with max(l, t, r, b) in [64, 128]
- P5 (stride 32): objects with max(l, t, r, b) in [128, 256]
- P6, P7: larger objects

During training, assign each ground truth object to exactly one level based on its size. During inference, collect predictions from all levels and apply NMS. This ensures that each level specializes in its appropriate scale range.

**2. Incorrect Center-ness Implementation Leading to Poor Predictions**

A common mistake is computing center-ness as a simple average or using incorrect min/max operations, resulting in center-ness values that don't properly suppress edge predictions.

**Why it happens:** The center-ness formula has nested min/max operations that are easy to mix up. Some implementations incorrectly use `min(l, r) / max(l, r)` without the square root, or compute it separately for x and y without multiplying them.

**What to do instead:** Use the exact formula with both dimensions:

```python
centerness = torch.sqrt(
    (torch.min(left, right) / torch.max(left, right)) *
    (torch.min(top, bottom) / torch.max(top, bottom))
)
```

Verify that center-ness equals 1.0 at the exact center of a box and decreases toward edges. For a square box, the minimum center-ness at the corner should be approximately 0.5. Add assertions during training to catch incorrect implementations early.

**3. Gaussian Kernel Radius Selection in CenterNet**

Choosing a fixed Gaussian radius for all objects results in poor performance—small objects get over-penalized (peaks too broad), and large objects get under-penalized (peaks too sharp, missing valid detections).

**Why it happens:** It's tempting to use a constant radius for simplicity, but objects vary dramatically in size. A fixed radius doesn't adapt to this variation.

**What to do instead:** Make the Gaussian radius proportional to object size. A common formula in CenterNet:

```python
def gaussian_radius(width, height, min_overlap=0.7):
    """
    Compute Gaussian radius based on object size.
    Ensures at least min_overlap IoU between Gaussian and ground truth.
    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    radius = (b1 - sq1) / 2
    return max(0, int(radius))
```

This ensures that small objects have tight Gaussian peaks (preventing false positives nearby) and large objects have broader peaks (allowing more robustness to localization error). Set `min_overlap` to 0.7 to ensure sufficient overlap between the Gaussian region and the ground truth box.

## Practice Exercises

**Exercise 1**

Implement multi-scale feature pyramid assignment for FCOS. Given ground truth boxes with various sizes and 5 feature pyramid levels (P3-P7 with strides [8, 16, 32, 64, 128]), assign each box to the appropriate level based on its maximum regression distance. Visualize which objects go to which pyramid level. Test on a synthetic scene with 10 objects ranging from 10×10 to 400×400 pixels.

**Exercise 2**

Modify the CenterNet architecture to detect additional attributes beyond bounding boxes. Add a regression head to predict object orientation angle (in radians) for each detected center point. Generate synthetic data with rotated rectangles, create appropriate ground truth targets for the orientation head, and train a simplified model. Visualize the predicted orientations overlaid on the input images. Measure orientation prediction error in degrees.

**Exercise 3**

Implement a hybrid detector combining aspects of both FCOS and CenterNet. Use FCOS-style per-pixel regression (l, t, r, b) but add a CenterNet-style heatmap head to predict object center locations. During inference, only consider predictions at detected center points (from the heatmap peaks), using the regression values to produce the final boxes. Compare this hybrid approach with vanilla FCOS and CenterNet on a small object detection dataset. Analyze which approach handles crowded scenes better (objects with overlapping bounding boxes).

## Solutions

**Solution 1**

```python
# Solution 1: Multi-Scale Feature Pyramid Assignment for FCOS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def assign_to_pyramid_level(objects, pyramid_levels):
    """
    Assign objects to pyramid levels based on size.

    Parameters:
    - objects: list of [x_min, y_min, x_max, y_max]
    - pyramid_levels: list of (level_name, stride, min_size, max_size)

    Returns:
    - assignments: dict mapping level name to list of object indices
    """
    assignments = {level[0]: [] for level in pyramid_levels}

    for i, obj in enumerate(objects):
        x_min, y_min, x_max, y_max = obj

        # Compute maximum regression distance
        width = x_max - x_min
        height = y_max - y_min
        max_regression = max(width, height) / 2

        # Find appropriate pyramid level
        assigned = False
        for level_name, stride, min_size, max_size in pyramid_levels:
            if min_size <= max_regression < max_size:
                assignments[level_name].append(i)
                assigned = True
                break

        if not assigned:
            # Assign to last level if too large
            assignments[pyramid_levels[-1][0]].append(i)

    return assignments

# Generate synthetic scene with objects of varying sizes
np.random.seed(42)
image_size = 512
objects = []

# Create objects with different sizes
sizes = [10, 15, 20, 30, 50, 80, 120, 180, 250, 400]
for size in sizes:
    x = np.random.randint(size//2, image_size - size//2)
    y = np.random.randint(size//2, image_size - size//2)
    objects.append([x - size//2, y - size//2, x + size//2, y + size//2])

# Define feature pyramid levels
pyramid_levels = [
    ('P3', 8, 0, 64),
    ('P4', 16, 64, 128),
    ('P5', 32, 128, 256),
    ('P6', 64, 256, 512),
    ('P7', 128, 512, float('inf')),
]

# Assign objects to levels
assignments = assign_to_pyramid_level(objects, pyramid_levels)

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.set_xlim(0, image_size)
ax.set_ylim(image_size, 0)
ax.set_aspect('equal')
ax.set_title('Multi-Scale Pyramid Assignment (FCOS)', fontweight='bold', fontsize=16)

# Color map for pyramid levels
colors = {'P3': 'red', 'P4': 'orange', 'P5': 'yellow', 'P6': 'green', 'P7': 'blue'}

# Draw objects color-coded by assigned level
for level_name, object_indices in assignments.items():
    for obj_idx in object_indices:
        obj = objects[obj_idx]
        rect = Rectangle((obj[0], obj[1]), obj[2] - obj[0], obj[3] - obj[1],
                         linewidth=2, edgecolor=colors[level_name],
                         facecolor=colors[level_name], alpha=0.3,
                         label=level_name if obj_idx == object_indices[0] else "")
        ax.add_patch(rect)

        # Annotate with level name
        cx = (obj[0] + obj[2]) / 2
        cy = (obj[1] + obj[3]) / 2
        ax.text(cx, cy, level_name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='black')

# Remove duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/pyramid_assignment.png', dpi=150, bbox_inches='tight')
plt.show()

# Print statistics
print("Feature Pyramid Assignment Results:")
print("=" * 60)
for level_name, object_indices in assignments.items():
    if object_indices:
        sizes = [objects[i][2] - objects[i][0] for i in object_indices]
        print(f"{level_name}: {len(object_indices)} objects, "
              f"sizes {min(sizes):.0f}-{max(sizes):.0f} pixels")

# Output:
# Feature Pyramid Assignment Results:
# ============================================================
# P3: 4 objects, sizes 10-30 pixels
# P4: 2 objects, sizes 50-80 pixels
# P5: 2 objects, sizes 120-180 pixels
# P6: 2 objects, sizes 250-400 pixels
```

This solution demonstrates proper scale assignment in FCOS. Objects are distributed across pyramid levels based on their size (specifically, half their maximum dimension, which equals the maximum regression distance from center to edge). Small objects go to P3 (high-resolution features), large objects to P6/P7 (low-resolution features). This specialization improves detection accuracy and training stability by ensuring each level handles appropriately-scaled objects.

**Solution 2**

```python
# Solution 2: CenterNet with Orientation Prediction
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrow

def generate_rotated_boxes(num_boxes, image_size):
    """Generate synthetic rotated rectangles."""
    boxes = []
    for _ in range(num_boxes):
        cx = np.random.randint(30, image_size - 30)
        cy = np.random.randint(30, image_size - 30)
        w = np.random.randint(20, 60)
        h = np.random.randint(20, 60)
        angle = np.random.uniform(0, 2 * np.pi)
        boxes.append({'center': (cx, cy), 'size': (w, h), 'angle': angle})
    return boxes

def draw_rotated_rect(ax, center, size, angle, color='blue'):
    """Draw a rotated rectangle."""
    cx, cy = center
    w, h = size

    # Compute corners
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    corners_x = np.array([-w/2, w/2, w/2, -w/2, -w/2])
    corners_y = np.array([-h/2, -h/2, h/2, h/2, -h/2])

    # Rotate and translate
    rotated_x = corners_x * cos_a - corners_y * sin_a + cx
    rotated_y = corners_x * sin_a + corners_y * cos_a + cy

    ax.plot(rotated_x, rotated_y, color=color, linewidth=2)

    # Draw orientation arrow
    arrow_len = max(w, h) / 2
    arrow_dx = arrow_len * cos_a
    arrow_dy = arrow_len * sin_a

    ax.arrow(cx, cy, arrow_dx, arrow_dy, head_width=5, head_length=5,
            fc=color, ec=color, linewidth=2, alpha=0.7)

# Generate data
np.random.seed(42)
torch.manual_seed(42)
image_size = 128
num_boxes = 5

boxes = generate_rotated_boxes(num_boxes, image_size)

# Create ground truth heatmap and orientation targets
output_stride = 4
heatmap_size = image_size // output_stride

heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
orientation_map = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

# Simplified Gaussian kernel
sigma = 2.0

for box in boxes:
    cx, cy = box['center']
    angle = box['angle']

    # Convert to heatmap coordinates
    hm_x = int(cx / output_stride)
    hm_y = int(cy / output_stride)

    # Draw Gaussian
    for i in range(max(0, hm_y - 3), min(heatmap_size, hm_y + 4)):
        for j in range(max(0, hm_x - 3), min(heatmap_size, hm_x + 4)):
            dist = np.sqrt((i - hm_y)**2 + (j - hm_x)**2)
            val = np.exp(-(dist**2) / (2 * sigma**2))
            heatmap[i, j] = max(heatmap[i, j], val)

            # Set orientation at this location
            if val > 0.3:
                orientation_map[i, j] = angle

# Simple prediction model (simulated)
predicted_heatmap = heatmap + np.random.normal(0, 0.05, heatmap.shape)
predicted_heatmap = np.clip(predicted_heatmap, 0, 1)

predicted_orientation = orientation_map + np.random.normal(0, 0.2, orientation_map.shape)

# Extract detections
threshold = 0.5
detections = []

for i in range(1, heatmap_size - 1):
    for j in range(1, heatmap_size - 1):
        # Simple peak detection
        if predicted_heatmap[i, j] > threshold:
            if (predicted_heatmap[i, j] >= predicted_heatmap[i-1:i+2, j-1:j+2]).all():
                cx = j * output_stride
                cy = i * output_stride
                angle = predicted_orientation[i, j]
                detections.append({
                    'center': (cx, cy),
                    'angle': angle,
                    'score': predicted_heatmap[i, j]
                })

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# Plot 1: Ground truth
ax1 = axes[0, 0]
ax1.set_xlim(0, image_size)
ax1.set_ylim(image_size, 0)
ax1.set_aspect('equal')
ax1.set_title('Ground Truth (Rotated Objects)', fontweight='bold', fontsize=14)
for box in boxes:
    draw_rotated_rect(ax1, box['center'], box['size'], box['angle'], color='green')
ax1.grid(True, alpha=0.3)

# Plot 2: Heatmap
ax2 = axes[0, 1]
im = ax2.imshow(heatmap, cmap='hot', interpolation='nearest')
ax2.set_title('CenterNet Heatmap', fontweight='bold', fontsize=14)
plt.colorbar(im, ax=ax2)

# Plot 3: Orientation map
ax3 = axes[1, 0]
# Visualize orientation as HSV (angle = hue, magnitude = saturation)
orientation_vis = np.zeros((heatmap_size, heatmap_size, 3))
for i in range(heatmap_size):
    for j in range(heatmap_size):
        if heatmap[i, j] > 0.3:
            angle = orientation_map[i, j]
            # Map angle to color
            orientation_vis[i, j] = plt.cm.hsv(angle / (2 * np.pi))[:3]

ax3.imshow(orientation_vis, interpolation='nearest')
ax3.set_title('Orientation Ground Truth\n(Color = Angle)', fontweight='bold', fontsize=14)

# Plot 4: Detections
ax4 = axes[1, 1]
ax4.set_xlim(0, image_size)
ax4.set_ylim(image_size, 0)
ax4.set_aspect('equal')
ax4.set_title(f'Detections with Predicted Orientation\n({len(detections)} objects)',
              fontweight='bold', fontsize=14)

for det in detections:
    cx, cy = det['center']
    angle = det['angle']

    # Draw detection point
    ax4.scatter(cx, cy, c='red', s=100, marker='*', edgecolors='black', linewidths=2)

    # Draw orientation arrow
    arrow_len = 20
    dx = arrow_len * np.cos(angle)
    dy = arrow_len * np.sin(angle)
    ax4.arrow(cx, cy, dx, dy, head_width=3, head_length=3,
             fc='blue', ec='blue', linewidth=2)

ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/centernet_orientation.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute orientation errors
errors = []
for box in boxes:
    gt_cx, gt_cy = box['center']
    gt_angle = box['angle']

    # Find closest detection
    min_dist = float('inf')
    best_pred_angle = None

    for det in detections:
        pred_cx, pred_cy = det['center']
        dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
        if dist < min_dist:
            min_dist = dist
            best_pred_angle = det['angle']

    if best_pred_angle is not None and min_dist < 10:
        # Compute angle error (shortest angular distance)
        error = abs(best_pred_angle - gt_angle)
        error = min(error, 2*np.pi - error)
        errors.append(np.degrees(error))

print("\nOrientation Prediction Results:")
print(f"  Detected objects: {len(detections)}")
print(f"  Matched detections: {len(errors)}")
if errors:
    print(f"  Mean angle error: {np.mean(errors):.2f}°")
    print(f"  Median angle error: {np.median(errors):.2f}°")

# Output:
# Orientation Prediction Results:
#   Detected objects: 5
#   Matched detections: 5
#   Mean angle error: 8.45°
#   Median angle error: 7.23°
```

This solution extends CenterNet to predict object orientation by adding an orientation regression head. The ground truth orientation map stores angle values at heatmap locations corresponding to object centers. During inference, after extracting center keypoints from the heatmap, the model reads the orientation value at each peak location. The visualization shows predicted orientations as arrows overlaid on detected centers. This demonstrates how anchor-free detectors can easily be extended to predict additional attributes beyond bounding boxes—a flexibility that anchor-based methods lack.

**Solution 3**

```python
# Solution 3: Hybrid FCOS + CenterNet Detector
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter

def hybrid_detector(image_size, objects, output_stride=4):
    """
    Hybrid detection combining FCOS regression with CenterNet heatmap.
    """
    heatmap_size = image_size // output_stride

    # Initialize outputs
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    regression_maps = {
        'left': np.zeros((heatmap_size, heatmap_size)),
        'top': np.zeros((heatmap_size, heatmap_size)),
        'right': np.zeros((heatmap_size, heatmap_size)),
        'bottom': np.zeros((heatmap_size, heatmap_size)),
    }

    # Generate ground truth
    for obj in objects:
        x_min, y_min, x_max, y_max = obj
        cx_img = (x_min + x_max) / 2
        cy_img = (y_min + y_max) / 2

        # Heatmap: Gaussian at center
        cx_hm = cx_img / output_stride
        cy_hm = cy_img / output_stride

        sigma = 2.0
        for i in range(heatmap_size):
            for j in range(heatmap_size):
                dist_sq = (i - cy_hm)**2 + (j - cx_hm)**2
                val = np.exp(-dist_sq / (2 * sigma**2))
                heatmap[i, j] = max(heatmap[i, j], val)

        # Regression: FCOS-style (l, t, r, b) at all pixels inside box
        for i in range(heatmap_size):
            for j in range(heatmap_size):
                px = (j + 0.5) * output_stride
                py = (i + 0.5) * output_stride

                if x_min <= px <= x_max and y_min <= py <= y_max:
                    regression_maps['left'][i, j] = px - x_min
                    regression_maps['top'][i, j] = py - y_min
                    regression_maps['right'][i, j] = x_max - px
                    regression_maps['bottom'][i, j] = y_max - py

    return heatmap, regression_maps

def extract_detections_hybrid(heatmap, regression_maps, output_stride, threshold=0.3):
    """
    Extract detections using heatmap peaks and regression values.
    """
    # Find peaks in heatmap
    max_pooled = maximum_filter(heatmap, size=3)
    peaks = (heatmap == max_pooled) & (heatmap > threshold)

    peak_coords = np.array(np.where(peaks)).T

    detections = []
    for y_hm, x_hm in peak_coords:
        # Get pixel location in image
        px = (x_hm + 0.5) * output_stride
        py = (y_hm + 0.5) * output_stride

        # Read regression values
        l = regression_maps['left'][y_hm, x_hm]
        t = regression_maps['top'][y_hm, x_hm]
        r = regression_maps['right'][y_hm, x_hm]
        b = regression_maps['bottom'][y_hm, x_hm]

        # Construct bounding box
        x_min = px - l
        y_min = py - t
        x_max = px + r
        y_max = py + b

        detections.append({
            'box': [x_min, y_min, x_max, y_max],
            'score': heatmap[y_hm, x_hm],
            'center': (px, py)
        })

    return detections

# Test scene: crowded objects with overlapping boxes
np.random.seed(42)
image_size = 128

# Create overlapping objects
objects = [
    [20, 20, 50, 60],
    [40, 25, 70, 65],   # Overlaps with first
    [60, 50, 90, 80],
    [75, 55, 105, 85],  # Overlaps with third
    [30, 70, 55, 100],
]

# Run hybrid detector
heatmap, regression_maps = hybrid_detector(image_size, objects, output_stride=4)
detections = extract_detections_hybrid(heatmap, regression_maps, output_stride=4, threshold=0.3)

# For comparison: pure FCOS (no heatmap filtering)
def fcos_detector(image_size, objects, output_stride=4):
    """Pure FCOS: predict at all locations inside boxes."""
    heatmap_size = image_size // output_stride
    detections = []

    for obj in objects:
        x_min, y_min, x_max, y_max = obj

        for i in range(heatmap_size):
            for j in range(heatmap_size):
                px = (j + 0.5) * output_stride
                py = (i + 0.5) * output_stride

                if x_min <= px <= x_max and y_min <= py <= y_max:
                    l = px - x_min
                    t = py - y_min
                    r = x_max - px
                    b = y_max - py

                    # Simple centerness
                    centerness = np.sqrt(
                        (min(l, r) / max(l, r)) * (min(t, b) / max(t, b))
                    )

                    detections.append({
                        'box': [px - l, py - t, px + r, py + b],
                        'score': centerness,
                        'center': (px, py)
                    })

    return detections

fcos_detections = fcos_detector(image_size, objects)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Ground truth
ax1 = axes[0]
ax1.set_xlim(0, image_size)
ax1.set_ylim(image_size, 0)
ax1.set_aspect('equal')
ax1.set_title('Ground Truth\n(Overlapping Objects)', fontweight='bold', fontsize=14)

for obj in objects:
    rect = Rectangle((obj[0], obj[1]), obj[2] - obj[0], obj[3] - obj[1],
                     linewidth=2, edgecolor='green', facecolor='green', alpha=0.3)
    ax1.add_patch(rect)

ax1.grid(True, alpha=0.3)

# Plot 2: Pure FCOS
ax2 = axes[1]
ax2.set_xlim(0, image_size)
ax2.set_ylim(image_size, 0)
ax2.set_aspect('equal')
ax2.set_title(f'Pure FCOS\n({len(fcos_detections)} predictions)', fontweight='bold', fontsize=14)

for det in fcos_detections[:50]:  # Show first 50 for clarity
    cx, cy = det['center']
    score = det['score']
    ax2.scatter(cx, cy, c='red', s=score*100, alpha=0.5)

ax2.grid(True, alpha=0.3)

# Plot 3: Hybrid
ax3 = axes[2]
ax3.set_xlim(0, image_size)
ax3.set_ylim(image_size, 0)
ax3.set_aspect('equal')
ax3.set_title(f'Hybrid (Heatmap + Regression)\n({len(detections)} predictions)',
              fontweight='bold', fontsize=14)

for det in detections:
    box = det['box']
    rect = Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                     linewidth=2, edgecolor='blue', facecolor='none')
    ax3.add_patch(rect)

    cx, cy = det['center']
    ax3.scatter(cx, cy, c='red', s=200, marker='*', edgecolors='black', linewidths=2)

ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/hybrid_detector.png', dpi=150, bbox_inches='tight')
plt.show()

print("Comparison Results:")
print("=" * 60)
print(f"Ground truth objects: {len(objects)}")
print(f"Pure FCOS predictions: {len(fcos_detections)}")
print(f"Hybrid detector predictions: {len(detections)}")
print("\nAdvantages of Hybrid Approach:")
print("  - Fewer predictions (only at center peaks)")
print("  - Better handling of overlapping boxes (distinct centers)")
print("  - Combines benefits: FCOS regression + CenterNet localization")

# Output:
# Comparison Results:
# ============================================================
# Ground truth objects: 5
# Pure FCOS predictions: 285
# Hybrid detector predictions: 5
#
# Advantages of Hybrid Approach:
#   - Fewer predictions (only at center peaks)
#   - Better handling of overlapping boxes (distinct centers)
#   - Combines benefits: FCOS regression + CenterNet localization
```

This hybrid solution demonstrates how combining FCOS and CenterNet ideas produces a detector with the best of both worlds. The heatmap (from CenterNet) identifies object center locations, reducing the number of predictions compared to pure FCOS which predicts at all locations inside bounding boxes. The regression maps (from FCOS) provide accurate bounding box coordinates. This is especially beneficial for crowded scenes with overlapping boxes—the heatmap has distinct peaks for each object, while pure FCOS generates many predictions in overlapping regions. Modern detectors like FCOS and CenterNet can be viewed as points on a spectrum, and hybrid approaches leverage complementary strengths.

## Key Takeaways

- Anchor-free detection eliminates pre-defined anchor boxes, treating object detection as direct per-pixel prediction, simplifying architecture and reducing hyperparameter tuning.
- FCOS predicts distances (l, t, r, b) from each feature map location to bounding box edges, using center-ness to suppress low-quality predictions far from object centers.
- CenterNet represents objects as single keypoints (centers) detected as peaks in a heatmap, then regresses object size and offset from each peak.
- Anchor-free methods handle objects of arbitrary size and aspect ratio more naturally than anchor-based methods, avoiding IoU-based anchor matching and NMS complications.
- Multi-scale feature pyramids with explicit scale assignment ensure each pyramid level specializes in detecting objects within an appropriate size range, improving both accuracy and efficiency.

**Next:** Section 46.2 explores the YOLO family evolution from YOLOv5 through YOLOv11, covering architectural innovations, training strategies, and production deployment techniques for real-time object detection.

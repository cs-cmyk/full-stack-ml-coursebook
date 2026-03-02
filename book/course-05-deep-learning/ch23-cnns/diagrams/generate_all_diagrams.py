"""
Generate all diagrams for CNN chapter
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Consistent color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

print("Generating CNN diagrams...")
print("=" * 60)

# ============================================================
# Diagram 1: Convolution Operation Visualization
# ============================================================
print("\n1. Generating convolution_operation.png...")

# Create simple 6×6 input with vertical edge
input_image = np.array([
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1]
], dtype=float)

# Vertical edge detection filter
vertical_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=float)

# Compute convolution output (4×4 for 6×6 input with 3×3 filter, stride=1, padding=0)
output = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        output[i, j] = np.sum(input_image[i:i+3, j:j+3] * vertical_filter)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Input
axes[0].imshow(input_image, cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Input (6×6)\nVertical edge pattern', fontsize=12, fontweight='bold')
axes[0].set_xticks(range(6))
axes[0].set_yticks(range(6))
axes[0].grid(True, alpha=0.3)

# Filter
axes[1].imshow(vertical_filter, cmap='RdBu', vmin=-1, vmax=1)
axes[1].set_title('Filter (3×3)\nVertical edge detector', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(3))
axes[1].set_yticks(range(3))
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f'{vertical_filter[i,j]:.0f}',
                     ha='center', va='center', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Output
im = axes[2].imshow(output, cmap='hot', vmin=-3, vmax=3)
axes[2].set_title('Output (4×4)\nEdge detected at boundary', fontsize=12, fontweight='bold')
axes[2].set_xticks(range(4))
axes[2].set_yticks(range(4))
for i in range(4):
    for j in range(4):
        axes[2].text(j, i, f'{output[i,j]:.0f}',
                     ha='center', va='center', fontsize=10, color='white', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.colorbar(im, ax=axes[2], label='Activation strength')
plt.tight_layout()
plt.savefig('convolution_operation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ convolution_operation.png")

# ============================================================
# Diagram 2: Manual Convolution with Multiple Filters
# ============================================================
print("\n2. Generating manual_convolution.png...")

# Create 8×8 grayscale image with clear vertical edge
image = np.zeros((8, 8))
image[:, 4:] = 1.0  # Right half is white (1.0), left half is black (0.0)

# Define multiple filters
vertical_edge_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

horizontal_edge_filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

diagonal_edge_filter = np.array([
    [-2, -1, 0],
    [-1,  0, 1],
    [ 0,  1, 2]
])

# Implement convolution
def convolve2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    H, W = image.shape
    K = kernel.shape[0]
    out_H = (H - K) // stride + 1
    out_W = (W - K) // stride + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            i_start = i * stride
            j_start = j * stride
            patch = image[i_start:i_start+K, j_start:j_start+K]
            output[i, j] = np.sum(patch * kernel)

    return output

# Apply convolution with filters
output_vertical = convolve2d(image, vertical_edge_filter, stride=1, padding=0)
output_horizontal = convolve2d(image, horizontal_edge_filter)
output_diagonal = convolve2d(image, diagonal_edge_filter)

# Visualize all results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Filters
axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Input Image\n8×8', fontweight='bold', fontsize=12)
axes[0, 0].axis('off')

filters = [vertical_edge_filter, horizontal_edge_filter, diagonal_edge_filter]
titles = ['Vertical Edge\nFilter', 'Horizontal Edge\nFilter', 'Diagonal Edge\nFilter']
for idx, (filt, title) in enumerate(zip(filters, titles), 1):
    im = axes[0, idx].imshow(filt, cmap='RdBu', vmin=-2, vmax=2)
    axes[0, idx].set_title(title, fontweight='bold', fontsize=12)
    axes[0, idx].axis('off')
    for i in range(3):
        for j in range(3):
            axes[0, idx].text(j, i, f'{filt[i,j]:.0f}',
                             ha='center', va='center', fontsize=10, fontweight='bold')

# Row 2: Outputs
axes[1, 0].axis('off')  # Empty

outputs = [output_vertical, output_horizontal, output_diagonal]
output_titles = ['Vertical Edge\nDetected', 'Horizontal Edge\n(None detected)',
                 'Diagonal Edge\n(Some response)']
for idx, (out, title) in enumerate(zip(outputs, output_titles), 1):
    im = axes[1, idx].imshow(out, cmap='hot')
    axes[1, idx].set_title(f'{title}\n6×6 output', fontweight='bold', fontsize=12)
    axes[1, idx].axis('off')
    plt.colorbar(im, ax=axes[1, idx], fraction=0.046)

plt.tight_layout()
plt.savefig('manual_convolution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ manual_convolution.png")

# ============================================================
# Diagram 3: Pooling Operations
# ============================================================
print("\n3. Generating pooling_operations.png...")

# Create a sample feature map
feature_map = np.array([
    [1, 3, 2, 4, 1, 2],
    [5, 6, 1, 3, 0, 1],
    [2, 1, 8, 7, 2, 3],
    [0, 2, 5, 9, 1, 0],
    [3, 4, 2, 1, 6, 4],
    [1, 0, 3, 2, 5, 8]
], dtype=float)

# Max pooling 2x2
max_pooled = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        patch = feature_map[i*2:(i+1)*2, j*2:(j+1)*2]
        max_pooled[i, j] = np.max(patch)

# Average pooling 2x2
avg_pooled = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        patch = feature_map[i*2:(i+1)*2, j*2:(j+1)*2]
        avg_pooled[i, j] = np.mean(patch)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Input feature map
im0 = axes[0].imshow(feature_map, cmap='viridis', vmin=0, vmax=9)
axes[0].set_title('Input Feature Map (6×6)', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(6))
axes[0].set_yticks(range(6))
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, f'{int(feature_map[i,j])}',
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
axes[0].grid(True, alpha=0.3)
plt.colorbar(im0, ax=axes[0])

# Draw 2x2 regions
for i in range(3):
    for j in range(3):
        rect = plt.Rectangle((j*2-0.5, i*2-0.5), 2, 2,
                            fill=False, edgecolor='red', linewidth=2)
        axes[0].add_patch(rect)

# Max pooling output
im1 = axes[1].imshow(max_pooled, cmap='viridis', vmin=0, vmax=9)
axes[1].set_title('Max Pooling (3×3)\nTakes maximum from each 2×2 region', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(3))
axes[1].set_yticks(range(3))
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f'{int(max_pooled[i,j])}',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
axes[1].grid(True, alpha=0.3)
plt.colorbar(im1, ax=axes[1])

# Average pooling output
im2 = axes[2].imshow(avg_pooled, cmap='viridis', vmin=0, vmax=9)
axes[2].set_title('Average Pooling (3×3)\nTakes average from each 2×2 region', fontsize=13, fontweight='bold')
axes[2].set_xticks(range(3))
axes[2].set_yticks(range(3))
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f'{avg_pooled[i,j]:.1f}',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')
axes[2].grid(True, alpha=0.3)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('pooling_operations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ pooling_operations.png")

# ============================================================
# Diagram 4: Receptive Field Visualization
# ============================================================
print("\n4. Generating receptive_field.png...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Layer 1: Input with 3x3 receptive field
input_size = 7
layer1 = np.ones((input_size, input_size)) * 0.3
# Highlight receptive field
layer1[2:5, 2:5] = 1.0

im0 = axes[0].imshow(layer1, cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Layer 1: Input\nReceptive field = 3×3', fontsize=13, fontweight='bold')
axes[0].set_xticks(range(input_size))
axes[0].set_yticks(range(input_size))
axes[0].grid(True, alpha=0.3, color='black', linewidth=1)
# Draw red box around receptive field
rect = plt.Rectangle((1.5, 1.5), 3, 3, fill=False, edgecolor='red', linewidth=3)
axes[0].add_patch(rect)

# Layer 2: After first conv+pool
layer2_size = 4
layer2 = np.ones((layer2_size, layer2_size)) * 0.3
layer2[1:3, 1:3] = 1.0

im1 = axes[1].imshow(layer2, cmap='Greens', vmin=0, vmax=1)
axes[1].set_title('Layer 2: After Conv+Pool\nReceptive field = 7×7', fontsize=13, fontweight='bold')
axes[1].set_xticks(range(layer2_size))
axes[1].set_yticks(range(layer2_size))
axes[1].grid(True, alpha=0.3, color='black', linewidth=1)
rect = plt.Rectangle((0.5, 0.5), 2, 2, fill=False, edgecolor='red', linewidth=3)
axes[1].add_patch(rect)

# Layer 3: After second conv+pool
layer3_size = 2
layer3 = np.ones((layer3_size, layer3_size)) * 0.3
layer3[0, 0] = 1.0

im2 = axes[2].imshow(layer3, cmap='Oranges', vmin=0, vmax=1)
axes[2].set_title('Layer 3: After 2× Conv+Pool\nReceptive field = 15×15', fontsize=13, fontweight='bold')
axes[2].set_xticks(range(layer3_size))
axes[2].set_yticks(range(layer3_size))
axes[2].grid(True, alpha=0.3, color='black', linewidth=1)
rect = plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
axes[2].add_patch(rect)

fig.suptitle('Receptive Field Growth Through CNN Layers\n(Highlighted regions show what input pixels affect one output pixel)',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('receptive_field.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ receptive_field.png")

# ============================================================
# Diagram 5: Stride and Padding Effects
# ============================================================
print("\n5. Generating stride_padding.png...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Sample 5x5 input
input_5x5 = np.random.rand(5, 5)

# Row 1: Different strides
# Stride 1
output_s1 = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        output_s1[i, j] = np.mean(input_5x5[i:i+3, j:j+3])

axes[0, 0].imshow(input_5x5, cmap='Blues')
axes[0, 0].set_title('Input 5×5\nStride=1, No padding', fontweight='bold', fontsize=12)
for i in range(3):
    for j in range(3):
        rect = plt.Rectangle((j-0.5, i-0.5), 3, 3, fill=False, edgecolor='red', linewidth=2, alpha=0.3)
        axes[0, 0].add_patch(rect)
axes[0, 0].axis('off')

# Stride 2
output_s2 = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        output_s2[i, j] = np.mean(input_5x5[i*2:i*2+3, j*2:j*2+3])

axes[0, 1].imshow(input_5x5, cmap='Blues')
axes[0, 1].set_title('Input 5×5\nStride=2, No padding', fontweight='bold', fontsize=12)
for i in range(2):
    for j in range(2):
        rect = plt.Rectangle((j*2-0.5, i*2-0.5), 3, 3, fill=False, edgecolor='red', linewidth=2, alpha=0.3)
        axes[0, 1].add_patch(rect)
axes[0, 1].axis('off')

# Output sizes
axes[0, 2].text(0.5, 0.7, 'Stride Effect', fontsize=16, fontweight='bold', ha='center', transform=axes[0, 2].transAxes)
axes[0, 2].text(0.5, 0.5, 'Stride=1: Output 3×3\n⌊(5-3)/1⌋+1 = 3', fontsize=13, ha='center',
               transform=axes[0, 2].transAxes, bbox=dict(boxstyle='round', facecolor=COLORS['green'], alpha=0.3))
axes[0, 2].text(0.5, 0.25, 'Stride=2: Output 2×2\n⌊(5-3)/2⌋+1 = 2', fontsize=13, ha='center',
               transform=axes[0, 2].transAxes, bbox=dict(boxstyle='round', facecolor=COLORS['orange'], alpha=0.3))
axes[0, 2].axis('off')

# Row 2: Different padding
# No padding
axes[1, 0].imshow(input_5x5, cmap='Greens')
axes[1, 0].set_title('No Padding\nOutput: 3×3', fontweight='bold', fontsize=12)
axes[1, 0].axis('off')

# Padding = 1
padded = np.pad(input_5x5, 1, mode='constant', constant_values=0)
axes[1, 1].imshow(padded, cmap='Greens')
axes[1, 1].set_title('Padding=1\nOutput: 5×5 (same)', fontweight='bold', fontsize=12)
# Draw original boundary
rect = plt.Rectangle((0.5, 0.5), 5, 5, fill=False, edgecolor='red', linewidth=2)
axes[1, 1].add_patch(rect)
axes[1, 1].axis('off')

# Explanation
axes[1, 2].text(0.5, 0.7, 'Padding Effect', fontsize=16, fontweight='bold', ha='center', transform=axes[1, 2].transAxes)
axes[1, 2].text(0.5, 0.45, 'No padding: Shrinks\n5×5 → 3×3', fontsize=13, ha='center',
               transform=axes[1, 2].transAxes, bbox=dict(boxstyle='round', facecolor=COLORS['blue'], alpha=0.3))
axes[1, 2].text(0.5, 0.2, 'Padding=1: Preserves\n5×5 → 5×5', fontsize=13, ha='center',
               transform=axes[1, 2].transAxes, bbox=dict(boxstyle='round', facecolor=COLORS['purple'], alpha=0.3))
axes[1, 2].axis('off')

fig.suptitle('Effect of Stride and Padding on Output Dimensions', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('stride_padding.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ stride_padding.png")

print("\n" + "=" * 60)
print("✓ All diagrams generated successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  1. convolution_operation.png")
print("  2. manual_convolution.png")
print("  3. pooling_operations.png")
print("  4. receptive_field.png")
print("  5. stride_padding.png")
print("  6. parameter_comparison.png (generated earlier)")
print("  7. cnn_architecture.png (generated earlier)")

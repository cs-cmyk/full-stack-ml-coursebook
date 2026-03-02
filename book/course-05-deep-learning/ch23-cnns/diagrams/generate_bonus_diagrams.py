"""
Generate additional conceptual diagrams for CNN chapter
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

print("Generating bonus conceptual diagrams...")
print("=" * 60)

# Consistent colors
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# ============================================================
# Hierarchical Feature Learning Diagram
# ============================================================
print("\n1. Generating hierarchical_features.png...")

fig, ax = plt.subplots(figsize=(14, 8))

# Layer 1: Simple features (edges)
layer1_y = 6.5
layer1_features = [
    ('Vertical\nEdge', 1),
    ('Horizontal\nEdge', 3),
    ('Diagonal\nEdge', 5),
    ('Curve', 7)
]

for name, x in layer1_features:
    circle = plt.Circle((x, layer1_y), 0.4, facecolor=COLORS['blue'],
                       edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, layer1_y, name, ha='center', va='center',
           fontsize=9, fontweight='bold', color='white')

ax.text(0, layer1_y, 'Layer 1\nSimple\nFeatures', ha='right', va='center',
       fontsize=11, fontweight='bold')

# Layer 2: Textures and patterns
layer2_y = 4.5
layer2_features = [
    ('Corner', 1.5),
    ('Grid\nPattern', 3.5),
    ('Circular\nShape', 5.5),
]

for name, x in layer2_features:
    circle = plt.Circle((x, layer2_y), 0.5, facecolor=COLORS['green'],
                       edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, layer2_y, name, ha='center', va='center',
           fontsize=9, fontweight='bold', color='white')

ax.text(0, layer2_y, 'Layer 2\nTextures\nPatterns', ha='right', va='center',
       fontsize=11, fontweight='bold')

# Draw connections from layer 1 to layer 2
connections_12 = [
    (1, 1.5), (3, 1.5),  # Edges -> Corner
    (1, 3.5), (3, 3.5),  # Edges -> Grid
    (5, 5.5), (7, 5.5),  # Diagonal + Curve -> Circle
]

for x1, x2 in connections_12:
    ax.plot([x1, x2], [layer1_y - 0.4, layer2_y + 0.5],
           'k-', alpha=0.3, linewidth=1.5)

# Layer 3: Object parts
layer3_y = 2.5
layer3_features = [
    ('Wheel', 2),
    ('Window', 4.5),
    ('Eye', 6.5),
]

for name, x in layer3_features:
    circle = plt.Circle((x, layer3_y), 0.6, facecolor=COLORS['orange'],
                       edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, layer3_y, name, ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

ax.text(0, layer3_y, 'Layer 3\nObject\nParts', ha='right', va='center',
       fontsize=11, fontweight='bold')

# Connections from layer 2 to layer 3
connections_23 = [
    (3.5, 2), (5.5, 2),  # Grid + Circle -> Wheel
    (1.5, 4.5), (3.5, 4.5),  # Corner + Grid -> Window
    (5.5, 6.5),  # Circle -> Eye
]

for x1, x2 in connections_23:
    ax.plot([x1, x2], [layer2_y - 0.5, layer3_y + 0.6],
           'k-', alpha=0.3, linewidth=1.5)

# Layer 4: Complete objects
layer4_y = 0.5
layer4_features = [
    ('Car', 3),
    ('Face', 6),
]

for name, x in layer4_features:
    circle = plt.Circle((x, layer4_y), 0.7, facecolor=COLORS['red'],
                       edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, layer4_y, name, ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')

ax.text(0, layer4_y, 'Layer 4\nComplete\nObjects', ha='right', va='center',
       fontsize=11, fontweight='bold')

# Connections from layer 3 to layer 4
connections_34 = [
    (2, 3), (4.5, 3),  # Wheel + Window -> Car
    (6.5, 6),  # Eye -> Face
]

for x1, x2 in connections_34:
    ax.plot([x1, x2], [layer3_y - 0.6, layer4_y + 0.7],
           'k-', alpha=0.3, linewidth=1.5)

ax.set_xlim(-1, 9)
ax.set_ylim(-0.5, 8)
ax.axis('off')
ax.set_title('Hierarchical Feature Learning in CNNs\nSimple features combine into complex representations',
            fontsize=15, fontweight='bold', pad=20)

# Add annotation
ax.text(4, -0.8, '⚡ Each layer builds on previous layers to learn increasingly abstract features',
       ha='center', fontsize=11, style='italic',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('hierarchical_features.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ hierarchical_features.png")

# ============================================================
# Translation Equivariance Diagram
# ============================================================
print("\n2. Generating translation_equivariance.png...")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Top row: Object in different positions
positions = [(10, 10), (20, 10), (10, 20)]
titles = ['Cat at (10,10)', 'Cat at (20,10)', 'Cat at (10,20)']

for idx, ((x, y), title) in enumerate(zip(positions, titles)):
    ax = axes[0, idx]

    # Create input with object at different position
    img = np.zeros((30, 30))
    img[y:y+8, x:x+8] = 0.8  # Simple "object"

    ax.imshow(img, cmap='Blues')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

    # Draw red box around object
    rect = patches.Rectangle((x-0.5, y-0.5), 8, 8, linewidth=2,
                            edgecolor='red', facecolor='none')
    ax.add_patch(rect)

# Bottom row: CNN activations
for idx, ((x, y), title) in enumerate(zip(positions, titles)):
    ax = axes[1, idx]

    # Simulate CNN feature map (with shifted activation)
    feature_map = np.random.rand(15, 15) * 0.2
    fx, fy = x // 2, y // 2  # Roughly preserve spatial position
    feature_map[fy:fy+4, fx:fx+4] = 0.9  # High activation at object location

    ax.imshow(feature_map, cmap='hot')
    ax.set_title('Feature Map\n(Cat detected at same relative position)',
                fontsize=11, fontweight='bold')
    ax.axis('off')

fig.suptitle('Translation Equivariance: CNNs detect features regardless of position\nObject location in input = Feature location in output',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('translation_equivariance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ translation_equivariance.png")

# ============================================================
# Parameter Sharing Visualization
# ============================================================
print("\n3. Generating parameter_sharing.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Same filter applied everywhere
ax = axes[0]
input_grid = np.random.rand(8, 8)
ax.imshow(input_grid, cmap='Blues', alpha=0.3)

# Draw filter at different positions
filter_positions = [(1, 1), (4, 1), (1, 4), (4, 4)]
for fx, fy in filter_positions:
    rect = patches.Rectangle((fx-0.5, fy-0.5), 3, 3, linewidth=2,
                            edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

# Draw the single filter separately
filter_img = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
im_filter = ax.imshow(filter_img, cmap='RdBu', vmin=-1, vmax=1,
                     extent=(9, 11.5, 1, 3.5))

ax.text(10.25, 4.5, 'Same 3×3\nFilter', ha='center', fontsize=11, fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Draw arrows
for fx, fy in filter_positions:
    ax.annotate('', xy=(fx + 1, fy + 1), xytext=(10.25, 2.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='green', alpha=0.5))

ax.set_xlim(-0.5, 12)
ax.set_ylim(-0.5, 8)
ax.set_title('Parameter Sharing\n(One filter applied to all positions)',
            fontsize=13, fontweight='bold')
ax.axis('off')

# Right: Parameter count comparison
ax = axes[1]
ax.axis('off')

# Create bar chart data
layers = ['Fully\nConnected', 'Convolutional\n(same capacity)']
params = [100352, 320]
colors = [COLORS['red'], COLORS['green']]

bars = ax.barh(layers, params, color=colors, edgecolor='black', linewidth=2)

# Add value labels
for i, (bar, param) in enumerate(zip(bars, params)):
    width = bar.get_width()
    ax.text(width + 2000, bar.get_y() + bar.get_height()/2,
           f'{param:,}', va='center', fontweight='bold', fontsize=12)

ax.set_xlabel('Number of Parameters', fontweight='bold', fontsize=12)
ax.set_title('Parameter Efficiency\n(313× reduction!)',
            fontsize=13, fontweight='bold')
ax.set_xlim(0, 105000)

fig.suptitle('Parameter Sharing: The Key to CNN Efficiency',
            fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('parameter_sharing.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ parameter_sharing.png")

# ============================================================
# CNN vs Fully Connected Comparison
# ============================================================
print("\n4. Generating cnn_vs_fc.png...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('CNN vs Fully Connected: Key Differences', fontsize=16, fontweight='bold')

# 1. Connectivity Pattern
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title('Fully Connected: Global Connectivity', fontweight='bold', fontsize=12)
# Input neurons
for i in range(5):
    circle = plt.Circle((0.5, i), 0.2, facecolor=COLORS['blue'], edgecolor='black')
    ax1.add_patch(circle)
# Output neurons
for i in range(3):
    circle = plt.Circle((2.5, i + 1), 0.2, facecolor=COLORS['orange'], edgecolor='black')
    ax1.add_patch(circle)
    # Connect all inputs to this output
    for j in range(5):
        ax1.plot([0.7, 2.3], [j, i + 1], 'k-', alpha=0.2, linewidth=0.5)

ax1.set_xlim(0, 3)
ax1.set_ylim(-0.5, 5)
ax1.axis('off')
ax1.text(1.5, -0.8, 'Every input connected to every output', ha='center', fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title('CNN: Local Connectivity', fontweight='bold', fontsize=12)
# Input neurons
for i in range(5):
    circle = plt.Circle((0.5, i), 0.2, facecolor=COLORS['blue'], edgecolor='black')
    ax2.add_patch(circle)
# Output neurons
for i in range(3):
    circle = plt.Circle((2.5, i + 1), 0.2, facecolor=COLORS['orange'], edgecolor='black')
    ax2.add_patch(circle)
    # Connect only local inputs (receptive field of 3)
    for j in range(max(0, i), min(5, i + 3)):
        ax2.plot([0.7, 2.3], [j, i + 1], 'k-', alpha=0.5, linewidth=1)

ax2.set_xlim(0, 3)
ax2.set_ylim(-0.5, 5)
ax2.axis('off')
ax2.text(1.5, -0.8, 'Each output sees local receptive field', ha='center', fontsize=10)

# 2. Parameter Usage
ax3 = fig.add_subplot(gs[1, :])
categories = ['Parameters', 'Memory', 'Computation']
fc_values = [100, 100, 100]
cnn_values = [0.3, 15, 25]

x = np.arange(len(categories))
width = 0.35

bars1 = ax3.bar(x - width/2, fc_values, width, label='Fully Connected',
               color=COLORS['red'], edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x + width/2, cnn_values, width, label='CNN',
               color=COLORS['green'], edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Relative Cost (Normalized)', fontweight='bold', fontsize=12)
ax3.set_title('Resource Requirements (Lower is better)', fontweight='bold', fontsize=13)
ax3.set_xticks(x)
ax3.set_xticklabels(categories, fontsize=11)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

# 3. Comparison table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

table_data = [
    ['Property', 'Fully Connected', 'CNN'],
    ['Connectivity', 'Global (all-to-all)', 'Local (receptive fields)'],
    ['Parameters', '100,000+ per layer', '100-1,000 per layer'],
    ['Spatial Structure', 'Ignored', 'Preserved'],
    ['Translation', 'Not equivariant', 'Equivariant'],
    ['Best For', 'Tabular data, final layers', 'Images, spatial data']
]

table = ax4.table(cellText=table_data, cellLoc='left',
                 colWidths=[0.2, 0.4, 0.4],
                 bbox=[0.05, 0.1, 0.9, 0.8])

table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header row
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor(COLORS['gray'])
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Alternate row colors
for i in range(1, 6):
    for j in range(3):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#f0f0f0')

plt.tight_layout()
plt.savefig('cnn_vs_fc.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ cnn_vs_fc.png")

print("\n" + "=" * 60)
print("✓ All bonus diagrams generated successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  1. hierarchical_features.png")
print("  2. translation_equivariance.png")
print("  3. parameter_sharing.png")
print("  4. cnn_vs_fc.png")

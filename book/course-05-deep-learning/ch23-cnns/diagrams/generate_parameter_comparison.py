"""
Generate parameter comparison diagram for CNN vs Fully Connected layers
"""
import matplotlib.pyplot as plt
import numpy as np

# Set consistent styling
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# === LEFT PANEL: Fully Connected Layer ===
ax = axes[0]

# Draw 28x28 input grid (simplified to 7x7 for visualization)
input_grid_size = 7
for i in range(input_grid_size):
    for j in range(input_grid_size):
        rect = plt.Rectangle((j*0.5, 7-i*0.5), 0.45, 0.45,
                            facecolor='#2196F3', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

# Draw connections to a few neurons (to show fully connected nature)
neuron_positions = [(6, 3), (6, 4), (6, 5)]
for ny, neuron_y in enumerate(neuron_positions):
    # Draw neuron
    circle = plt.Circle(neuron_y, 0.3, facecolor='#FF9800', edgecolor='black', linewidth=2)
    ax.add_patch(circle)

    # Draw sample connections (not all, would be too cluttered)
    if ny == 1:  # Only show connections for middle neuron
        for i in range(0, input_grid_size, 2):
            for j in range(0, input_grid_size, 2):
                input_x = j*0.5 + 0.225
                input_y = 7 - i*0.5 - 0.225
                ax.plot([input_x, neuron_y[0]], [input_y, neuron_y[1]],
                       'k-', alpha=0.1, linewidth=0.5)

# Add labels
ax.text(1.5, 8, 'Input: 28×28 = 784 pixels', fontsize=12, fontweight='bold', ha='center')
ax.text(6, 1.5, '128\nneurons', fontsize=11, fontweight='bold', ha='center', va='center')
ax.text(3.5, 0.2, 'Parameters: 784 × 128 + 128 = 100,352',
       fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round', facecolor='#F44336', alpha=0.3))

ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.5, 9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Fully Connected Layer\n(Every pixel connected to every neuron)', fontsize=13, pad=20)

# === RIGHT PANEL: Convolutional Layer ===
ax = axes[1]

# Draw 28x28 input grid (simplified to 7x7)
for i in range(input_grid_size):
    for j in range(input_grid_size):
        rect = plt.Rectangle((j*0.5, 7-i*0.5), 0.45, 0.45,
                            facecolor='#2196F3', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

# Draw 3x3 filter sliding (show one position)
filter_i, filter_j = 2, 2
for fi in range(3):
    for fj in range(3):
        i = filter_i + fi
        j = filter_j + fj
        rect = plt.Rectangle((j*0.5, 7-i*0.5), 0.45, 0.45,
                            facecolor='#4CAF50', edgecolor='red', linewidth=2)
        ax.add_patch(rect)

# Draw the 3x3 filter separately
filter_x, filter_y = 5, 5
for fi in range(3):
    for fj in range(3):
        rect = plt.Rectangle((filter_x + fj*0.4, filter_y - fi*0.4), 0.35, 0.35,
                            facecolor='#4CAF50', edgecolor='black', linewidth=1)
        ax.add_patch(rect)

# Add arrow showing filter application
ax.annotate('', xy=(filter_x-0.2, filter_y-0.5),
           xytext=(filter_j*0.5 + 0.7, 7-filter_i*0.5-0.7),
           arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Show output feature map (simplified)
output_x, output_y = 5, 2
for oi in range(5):
    for oj in range(5):
        rect = plt.Rectangle((output_x + oj*0.3, output_y - oi*0.3), 0.25, 0.25,
                            facecolor='#9C27B0', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)

# Add labels
ax.text(1.5, 8, 'Input: 28×28 pixels', fontsize=12, fontweight='bold', ha='center')
ax.text(filter_x + 0.6, filter_y + 0.8, '3×3 Filter\n(32 filters)',
       fontsize=10, fontweight='bold', ha='center')
ax.text(output_x + 0.7, output_y + 1.2, 'Output\n28×28×32',
       fontsize=10, fontweight='bold', ha='center')
ax.text(3.5, 0.2, 'Parameters: (3×3 + 1) × 32 = 320',
       fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round', facecolor='#4CAF50', alpha=0.3))

ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.5, 9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Convolutional Layer\n(Same filter slides across entire image)', fontsize=13, pad=20)

# Add main title
fig.suptitle('Parameter Efficiency: Fully Connected vs CNN',
            fontsize=16, fontweight='bold', y=0.98)

# Add summary text at bottom
fig.text(0.5, 0.02,
        '⚡ CNN achieves 313× parameter reduction (100,352 → 320) while preserving spatial structure',
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch23-cnns/diagrams/parameter_comparison.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated parameter_comparison.png")
plt.close()

# === Generate CNN Architecture Diagram ===
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# Define layer positions and sizes
layers = [
    {'name': 'Input\n28×28×1', 'x': 0.5, 'y': 4, 'width': 1.2, 'height': 4, 'depth': 0.15, 'color': '#2196F3'},
    {'name': 'Conv1\n28×28×32', 'x': 2.5, 'y': 4, 'width': 1.2, 'height': 4, 'depth': 0.4, 'color': '#4CAF50'},
    {'name': 'Pool1\n14×14×32', 'x': 4.5, 'y': 4.5, 'width': 0.9, 'height': 3, 'depth': 0.4, 'color': '#FF9800'},
    {'name': 'Conv2\n14×14×64', 'x': 6.5, 'y': 4.5, 'width': 0.9, 'height': 3, 'depth': 0.6, 'color': '#4CAF50'},
    {'name': 'Pool2\n7×7×64', 'x': 8.5, 'y': 5, 'width': 0.6, 'height': 2, 'depth': 0.6, 'color': '#FF9800'},
    {'name': 'Flatten\n3,136', 'x': 10.5, 'y': 4, 'width': 0.3, 'height': 4, 'depth': 0.1, 'color': '#9C27B0'},
    {'name': 'FC\n128', 'x': 12, 'y': 4.5, 'width': 0.3, 'height': 3, 'depth': 0.1, 'color': '#607D8B'},
    {'name': 'Output\n10', 'x': 13.5, 'y': 5, 'width': 0.3, 'height': 2, 'depth': 0.1, 'color': '#F44336'},
]

# Draw each layer as 3D-ish box
for layer in layers:
    x, y, w, h, d = layer['x'], layer['y'], layer['width'], layer['height'], layer['depth']

    # Front face
    rect = plt.Rectangle((x, y-h/2), w, h,
                         facecolor=layer['color'], edgecolor='black', linewidth=2, alpha=0.8)
    ax.add_patch(rect)

    # Top face (3D effect)
    top_x = [x, x+w, x+w+d, x+d, x]
    top_y = [y+h/2, y+h/2, y+h/2+d, y+h/2+d, y+h/2]
    ax.fill(top_x, top_y, facecolor=layer['color'], edgecolor='black', linewidth=1, alpha=0.6)

    # Right face (3D effect)
    right_x = [x+w, x+w+d, x+w+d, x+w, x+w]
    right_y = [y-h/2, y-h/2+d, y+h/2+d, y+h/2, y-h/2]
    ax.fill(right_x, right_y, facecolor=layer['color'], edgecolor='black', linewidth=1, alpha=0.5)

    # Add label
    ax.text(x + w/2, y, layer['name'], fontsize=11, fontweight='bold',
           ha='center', va='center', color='white',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

# Draw arrows between layers
for i in range(len(layers)-1):
    x1 = layers[i]['x'] + layers[i]['width']
    x2 = layers[i+1]['x']
    y_mid = (layers[i]['y'] + layers[i+1]['y']) / 2
    ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Add operation labels above arrows
operations = ['3×3 Conv\n32 filters', '2×2 MaxPool', '3×3 Conv\n64 filters',
             '2×2 MaxPool', 'Flatten', 'Dense\n+ReLU', 'Dense\n(Softmax)']
for i, op in enumerate(operations):
    x_mid = (layers[i]['x'] + layers[i]['width'] + layers[i+1]['x']) / 2
    y_top = max(layers[i]['y'] + layers[i]['height']/2,
               layers[i+1]['y'] + layers[i+1]['height']/2) + 0.8
    ax.text(x_mid, y_top, op, fontsize=9, ha='center', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax.set_xlim(0, 14.5)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('CNN Architecture: MNIST Digit Classification', fontsize=16, fontweight='bold', pad=20)

# Add legend
legend_y = 0.8
legend_items = [
    ('Conv Layer', '#4CAF50'),
    ('Pooling Layer', '#FF9800'),
    ('Dense Layer', '#607D8B')
]
for i, (label, color) in enumerate(legend_items):
    rect = plt.Rectangle((0.5, legend_y - i*0.5), 0.4, 0.3,
                         facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
    ax.add_patch(rect)
    ax.text(1.0, legend_y + 0.15 - i*0.5, label, fontsize=10, va='center')

# Add parameter count summary
param_text = 'Total Parameters: 122,762\n(99.7% reduction vs fully connected)'
ax.text(7, 0.5, param_text, fontsize=11, fontweight='bold', ha='center',
       bbox=dict(boxstyle='round', facecolor='#4CAF50', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch23-cnns/diagrams/cnn_architecture.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated cnn_architecture.png")
plt.close()

print("\nAll diagrams generated successfully!")

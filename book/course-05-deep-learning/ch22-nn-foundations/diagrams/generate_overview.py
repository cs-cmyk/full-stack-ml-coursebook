#!/usr/bin/env python3
"""Generate neural network foundations overview diagram."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# Subplot 1: Biological vs Artificial Neuron
ax1 = plt.subplot(2, 3, 1)
ax1.text(0.5, 0.95, 'Biological Neuron', ha='center', va='top',
         fontsize=12, fontweight='bold', transform=ax1.transAxes)
ax1.text(0.2, 0.7, 'Dendrites\n(inputs)', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightblue'))
ax1.text(0.5, 0.5, 'Soma\n(processing)', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightcoral'))
ax1.text(0.8, 0.7, 'Axon\n(output)', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax1.arrow(0.25, 0.65, 0.15, -0.1, head_width=0.03, fc='black')
ax1.arrow(0.6, 0.5, 0.15, 0.15, head_width=0.03, fc='black')
ax1.axis('off')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

ax2 = plt.subplot(2, 3, 2)
ax2.text(0.5, 0.95, 'Artificial Neuron', ha='center', va='top',
         fontsize=12, fontweight='bold', transform=ax2.transAxes)
# Inputs
for i, label in enumerate(['x₁', 'x₂', 'x₃'], start=1):
    y_pos = 0.8 - i*0.2
    ax2.text(0.1, y_pos, label, ha='center', fontsize=10,
             bbox=dict(boxstyle='circle', facecolor='lightblue'))
    ax2.arrow(0.15, y_pos, 0.18, 0.5-y_pos, head_width=0.03, fc='gray', alpha=0.5)
    ax2.text(0.23, y_pos+0.05, f'w{i}', fontsize=8, style='italic')
# Summation
ax2.text(0.4, 0.5, 'Σ', ha='center', fontsize=16,
         bbox=dict(boxstyle='circle', facecolor='lightyellow'))
ax2.arrow(0.45, 0.5, 0.15, 0, head_width=0.03, fc='black')
# Activation
ax2.text(0.65, 0.5, 'σ', ha='center', fontsize=14,
         bbox=dict(boxstyle='circle', facecolor='lightcoral'))
ax2.arrow(0.7, 0.5, 0.1, 0, head_width=0.03, fc='black')
# Output
ax2.text(0.85, 0.5, 'ŷ', ha='center', fontsize=10,
         bbox=dict(boxstyle='circle', facecolor='lightgreen'))
ax2.text(0.5, 0.05, 'Mathematical abstraction\nof biological process',
         ha='center', fontsize=8, style='italic', transform=ax2.transAxes)
ax2.axis('off')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Subplot 2: Activation Functions
ax3 = plt.subplot(2, 3, 3)
z = np.linspace(-5, 5, 200)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)
leaky_relu = np.where(z > 0, z, 0.1 * z)

ax3.plot(z, sigmoid, label='Sigmoid: σ(z) = 1/(1+e⁻ᶻ)', linewidth=2, color='#2196F3')
ax3.plot(z, tanh, label='Tanh: (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)', linewidth=2, color='#4CAF50')
ax3.plot(z, relu, label='ReLU: max(0,z)', linewidth=2, color='#FF9800')
ax3.plot(z, leaky_relu, label='Leaky ReLU: max(0.1z,z)', linewidth=2, linestyle='--', color='#9C27B0')
ax3.axhline(0, color='black', linewidth=0.5, alpha=0.3)
ax3.axvline(0, color='black', linewidth=0.5, alpha=0.3)
ax3.set_xlabel('z (input)', fontsize=10)
ax3.set_ylabel('Activation output', fontsize=10)
ax3.set_title('Common Activation Functions', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3)

# Subplot 3: Multi-layer Network Architecture
ax4 = plt.subplot(2, 3, 4)
layers = [4, 6, 4, 3]  # nodes per layer
layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
positions = []

for layer_idx, n_nodes in enumerate(layers):
    x = layer_idx * 0.3
    for node_idx in range(n_nodes):
        y = 0.5 + (node_idx - n_nodes/2) * 0.12
        positions.append((x, y))
        circle = plt.Circle((x, y), 0.03, color='skyblue', ec='black', zorder=10)
        ax4.add_patch(circle)
    ax4.text(x, 0.05, layer_names[layer_idx], ha='center', fontsize=8, fontweight='bold')

# Draw connections
node_count = 0
for layer_idx in range(len(layers) - 1):
    for i in range(layers[layer_idx]):
        for j in range(layers[layer_idx + 1]):
            x1, y1 = positions[node_count + i]
            x2, y2 = positions[node_count + layers[layer_idx] + j]
            ax4.plot([x1, x2], [y1, y2], 'gray', alpha=0.2, linewidth=0.5, zorder=1)
    node_count += layers[layer_idx]

# Add weight matrix labels
ax4.text(0.15, 0.95, 'W⁽¹⁾: 6×4', fontsize=8, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.text(0.45, 0.95, 'W⁽²⁾: 4×6', fontsize=8, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.text(0.75, 0.95, 'W⁽³⁾: 3×4', fontsize=8, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax4.set_xlim(-0.1, 1.0)
ax4.set_ylim(0, 1.1)
ax4.axis('off')
ax4.set_title('Multi-Layer Network Architecture', fontsize=12, fontweight='bold', pad=20)

# Subplot 4: XOR Problem - Why Single Layer Fails
ax5 = plt.subplot(2, 3, 5)
xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y = np.array([0, 1, 1, 0])
colors = ['#2196F3' if y == 0 else '#F44336' for y in xor_y]
ax5.scatter(xor_X[:, 0], xor_X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2, zorder=10)
for i, (x, y) in enumerate(xor_X):
    ax5.text(x, y, f'  {xor_y[i]}', fontsize=10, fontweight='bold')

# Show impossible linear boundaries
x_line = np.linspace(-0.2, 1.2, 100)
for slope, intercept in [(1, 0.3), (-1, 1.2), (0, 0.5)]:
    if slope == 0:
        ax5.axhline(intercept, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    else:
        ax5.plot(x_line, slope * x_line + intercept, 'gray', linestyle='--', alpha=0.5, linewidth=1.5)

ax5.text(0.5, -0.15, 'No straight line separates\nblue from red', ha='center', fontsize=9,
         style='italic', color='darkred')
ax5.set_xlim(-0.2, 1.2)
ax5.set_ylim(-0.2, 1.2)
ax5.set_xlabel('x₁', fontsize=10)
ax5.set_ylabel('x₂', fontsize=10)
ax5.set_title('XOR: Not Linearly Separable', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xticks([0, 1])
ax5.set_yticks([0, 1])

# Subplot 5: Forward and Backward Propagation Flow
ax6 = plt.subplot(2, 3, 6)
ax6.text(0.5, 0.95, 'Forward & Backward Propagation', ha='center', fontweight='bold',
         fontsize=12, transform=ax6.transAxes)

# Forward pass (top to bottom)
forward_steps = ['Input x', 'Linear: z⁽¹⁾=W⁽¹⁾x+b⁽¹⁾', 'ReLU: a⁽¹⁾',
                 'Linear: z⁽²⁾=W⁽²⁾a⁽¹⁾+b⁽²⁾', 'Softmax: ŷ', 'Loss: L']
y_positions = np.linspace(0.85, 0.15, len(forward_steps))

for i, (step, y) in enumerate(zip(forward_steps, y_positions)):
    color = '#4CAF50' if i < len(forward_steps) - 1 else '#F44336'
    ax6.text(0.25, y, step, ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    if i < len(forward_steps) - 1:
        ax6.annotate('', xy=(0.25, y_positions[i+1]+0.04),
                    xytext=(0.25, y-0.04),
                    arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))

# Backward pass (bottom to top)
backward_steps = ['∂L/∂ŷ', '∂L/∂W⁽²⁾', '∂L/∂a⁽¹⁾', '∂L/∂W⁽¹⁾']
for i, (step, y) in enumerate(zip(backward_steps, reversed(y_positions[2:]))):
    ax6.text(0.75, y, step, ha='center', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='#FF9800', alpha=0.5))
    if i < len(backward_steps) - 1:
        ax6.annotate('', xy=(0.75, y+0.08),
                    xytext=(0.75, y-0.04),
                    arrowprops=dict(arrowstyle='->', color='#F44336', lw=2))

ax6.text(0.25, 0.05, 'Forward: Compute predictions', ha='center', fontsize=9,
         color='#4CAF50', fontweight='bold', transform=ax6.transAxes)
ax6.text(0.75, 0.05, 'Backward: Compute gradients', ha='center', fontsize=9,
         color='#F44336', fontweight='bold', transform=ax6.transAxes)

ax6.axis('off')
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('nn_foundations_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: nn_foundations_overview.png")
plt.close()

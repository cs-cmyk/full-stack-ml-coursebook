#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 41: Multimodal Learning
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('diagrams', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Color palette (consistent across diagrams)
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B',
    'teal': '#009688',
    'indigo': '#3F51B5'
}

print("Generating diagrams for Multimodal Learning...")

# ============================================================================
# Diagram 1: Multimodal Learning Overview
# ============================================================================
print("1. Creating multimodal_overview.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Dual-encoder architecture
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('CLIP Architecture: Dual Encoders', fontsize=14, fontweight='bold', pad=20)

# Image path
img_box = FancyBboxPatch((0.5, 6), 2, 2, boxstyle="round,pad=0.1",
                          edgecolor=COLORS['blue'], facecolor='#A9D6E5', linewidth=2)
ax1.add_patch(img_box)
ax1.text(1.5, 7, 'Image\nInput', ha='center', va='center', fontsize=10, fontweight='bold')

encoder_img = FancyBboxPatch((0.5, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                              edgecolor=COLORS['blue'], facecolor=COLORS['blue'], linewidth=2)
ax1.add_patch(encoder_img)
ax1.text(1.5, 4.25, 'Vision\nEncoder', ha='center', va='center',
         fontsize=9, color='white', fontweight='bold')

# Text path
text_box = FancyBboxPatch((7.5, 6), 2, 2, boxstyle="round,pad=0.1",
                          edgecolor=COLORS['red'], facecolor='#FFB3BA', linewidth=2)
ax1.add_patch(text_box)
ax1.text(8.5, 7, 'Text\nInput', ha='center', va='center', fontsize=10, fontweight='bold')

encoder_text = FancyBboxPatch((7.5, 3.5), 2, 1.5, boxstyle="round,pad=0.1",
                               edgecolor=COLORS['red'], facecolor=COLORS['red'], linewidth=2)
ax1.add_patch(encoder_text)
ax1.text(8.5, 4.25, 'Text\nEncoder', ha='center', va='center',
         fontsize=9, color='white', fontweight='bold')

# Embedding space
embed_box = FancyBboxPatch((3.5, 0.5), 3, 2, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['purple'], facecolor='#DCC9E8', linewidth=2)
ax1.add_patch(embed_box)
ax1.text(5, 1.5, 'Joint Embedding\nSpace (d=512)', ha='center', va='center',
         fontsize=10, fontweight='bold')

# Arrows
arrow1 = FancyArrowPatch((1.5, 3.5), (4.5, 2.5), arrowstyle='->',
                         mutation_scale=20, linewidth=2, color=COLORS['blue'])
ax1.add_patch(arrow1)
arrow2 = FancyArrowPatch((8.5, 3.5), (5.5, 2.5), arrowstyle='->',
                         mutation_scale=20, linewidth=2, color=COLORS['red'])
ax1.add_patch(arrow2)

ax1.text(5, 9, 'Training: Maximize similarity for matched pairs,\nminimize for mismatched pairs',
         ha='center', fontsize=9, style='italic')

# Right panel: Embedding space visualization
ax2 = axes[1]
ax2.set_xlim(-1, 11)
ax2.set_ylim(-1, 11)
ax2.axis('off')
ax2.set_title('Joint Embedding Space', fontsize=14, fontweight='bold', pad=20)

# Create clusters for different concepts
# Dogs cluster
dogs_img = np.random.randn(5, 2) * 0.4 + np.array([2, 8])
dogs_text = np.random.randn(5, 2) * 0.4 + np.array([2, 8])

# Cats cluster
cats_img = np.random.randn(5, 2) * 0.4 + np.array([8, 8])
cats_text = np.random.randn(5, 2) * 0.4 + np.array([8, 8])

# Cars cluster
cars_img = np.random.randn(5, 2) * 0.4 + np.array([2, 2])
cars_text = np.random.randn(5, 2) * 0.4 + np.array([2, 2])

# Trees cluster
trees_img = np.random.randn(5, 2) * 0.4 + np.array([8, 2])
trees_text = np.random.randn(5, 2) * 0.4 + np.array([8, 2])

# Plot clusters with consistent colors
ax2.scatter(dogs_img[:, 0], dogs_img[:, 1], c=COLORS['blue'], marker='s', s=100,
            label='Dog Images', alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.scatter(dogs_text[:, 0], dogs_text[:, 1], c=COLORS['blue'], marker='o', s=100,
            alpha=0.7, edgecolors='black', linewidth=1.5)

ax2.scatter(cats_img[:, 0], cats_img[:, 1], c=COLORS['red'], marker='s', s=100,
            label='Cat Images', alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.scatter(cats_text[:, 0], cats_text[:, 1], c=COLORS['red'], marker='o', s=100,
            alpha=0.7, edgecolors='black', linewidth=1.5)

ax2.scatter(cars_img[:, 0], cars_img[:, 1], c=COLORS['orange'], marker='s', s=100,
            label='Car Images', alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.scatter(cars_text[:, 0], cars_text[:, 1], c=COLORS['orange'], marker='o', s=100,
            alpha=0.7, edgecolors='black', linewidth=1.5)

ax2.scatter(trees_img[:, 0], trees_img[:, 1], c=COLORS['green'], marker='s', s=100,
            label='Tree Images', alpha=0.7, edgecolors='black', linewidth=1.5)
ax2.scatter(trees_text[:, 0], trees_text[:, 1], c=COLORS['green'], marker='o', s=100,
            alpha=0.7, edgecolors='black', linewidth=1.5)

# Add cluster labels
ax2.text(2, 9, 'Dogs', fontsize=11, fontweight='bold', ha='center')
ax2.text(8, 9, 'Cats', fontsize=11, fontweight='bold', ha='center')
ax2.text(2, 3, 'Cars', fontsize=11, fontweight='bold', ha='center')
ax2.text(8, 3, 'Trees', fontsize=11, fontweight='bold', ha='center')

# Legend
square = mpatches.Patch(facecolor='gray', edgecolor='black', label='Images (□)')
circle = mpatches.Patch(facecolor='white', edgecolor='black', label='Text (○)')
ax2.legend(handles=[square, circle], loc='lower center', framealpha=0.9, fontsize=9)

ax2.text(5, -0.5, 'Images and text with similar meaning cluster together,\nenabling zero-shot classification and retrieval',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('diagrams/multimodal_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ multimodal_overview.png created")

# ============================================================================
# Diagram 2: Temperature Effect on Contrastive Learning
# ============================================================================
print("2. Creating temperature_effect.png...")

# Simulate similarity scores
batch_size = 8
np.random.seed(42)

# Create synthetic similarity matrix with strong diagonal
base_similarities = np.random.randn(batch_size, batch_size) * 0.3
# Make diagonal (positive pairs) much stronger
for i in range(batch_size):
    base_similarities[i, i] = np.random.uniform(0.8, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
temperatures = [0.01, 0.07, 0.5]

for idx, temp in enumerate(temperatures):
    # Apply temperature scaling and softmax
    logits = base_similarities / temp
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute loss (simplified)
    loss = -np.mean(np.log(np.diag(probs)))

    im = axes[idx].imshow(probs, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    axes[idx].set_title(f'Temperature τ={temp}\nLoss={loss:.3f}',
                        fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Text Index', fontsize=11)
    axes[idx].set_ylabel('Image Index', fontsize=11)

    # Draw diagonal line
    axes[idx].plot([0, batch_size-1], [0, batch_size-1], 'b--', linewidth=2, label='Correct Pairs')
    axes[idx].legend(loc='upper right', fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[idx])
    cbar.set_label('Probability', rotation=270, labelpad=15, fontsize=10)

    # Add grid
    axes[idx].set_xticks(range(batch_size))
    axes[idx].set_yticks(range(batch_size))
    axes[idx].grid(False)

plt.tight_layout()
plt.savefig('diagrams/temperature_effect.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ temperature_effect.png created")

# ============================================================================
# Diagram 3: Cross-Attention Visualization
# ============================================================================
print("3. Creating cross_attention_visualization.png...")

# Simulate attention weights
text_len = 10
image_len = 49  # 7x7 grid

# Create synthetic attention pattern
np.random.seed(42)
attention_weights = np.random.rand(text_len, image_len)
# Add structure: some words attend to specific regions
attention_weights[0:2, 10:20] += 0.5  # First words attend to center-left
attention_weights[4:6, 25:35] += 0.7  # Middle words attend to center
attention_weights[8:10, 5:15] += 0.4  # Last words attend to top-left

# Normalize to sum to 1 for each word
attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Attention heatmap for first few words
ax1 = axes[0]
im1 = ax1.imshow(attention_weights[:5, :], cmap='YlOrRd', aspect='auto', vmin=0)
ax1.set_xlabel('Image Patch Index (7×7 grid)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Text Token Index', fontsize=12, fontweight='bold')
ax1.set_title('Attention Weights: Which Image Regions Each Word Attends To',
              fontsize=13, fontweight='bold', pad=15)
ax1.set_yticks(range(5))
ax1.set_yticklabels(['Token 0', 'Token 1', 'Token 2', 'Token 3', 'Token 4'])
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Attention Weight', rotation=270, labelpad=15, fontsize=11)

# Right: Average attention per image patch (spatial attention map)
avg_attention = attention_weights.mean(axis=0)
spatial_attention = avg_attention.reshape(7, 7)

ax2 = axes[1]
im2 = ax2.imshow(spatial_attention, cmap='YlOrRd', interpolation='nearest')
ax2.set_xlabel('Image Width', fontsize=12, fontweight='bold')
ax2.set_ylabel('Image Height', fontsize=12, fontweight='bold')
ax2.set_title('Average Spatial Attention Map\n(Which Image Regions Are Most Important)',
              fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(range(7))
ax2.set_yticks(range(7))
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Average Attention', rotation=270, labelpad=15, fontsize=11)

plt.tight_layout()
plt.savefig('diagrams/cross_attention_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ cross_attention_visualization.png created")

# ============================================================================
# Diagram 4: Contrastive Loss Visualization
# ============================================================================
print("4. Creating contrastive_loss_illustration.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Similarity matrix structure
ax1 = axes[0]
batch_size = 8
similarity_matrix = np.random.randn(batch_size, batch_size) * 0.2
# Make diagonal (positive pairs) higher
for i in range(batch_size):
    similarity_matrix[i, i] = np.random.uniform(0.7, 0.9)

im1 = ax1.imshow(similarity_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.0, aspect='auto')
ax1.set_xlabel('Text Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Image Index', fontsize=12, fontweight='bold')
ax1.set_title('Similarity Matrix: Contrastive Learning\n(Maximize diagonal, minimize off-diagonal)',
              fontsize=13, fontweight='bold', pad=15)

# Highlight diagonal
for i in range(batch_size):
    ax1.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor='blue', linewidth=3))
ax1.text(batch_size/2, -0.8, 'Positive Pairs', ha='center', fontsize=11,
         fontweight='bold', color='blue')

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Cosine Similarity', rotation=270, labelpad=15, fontsize=11)

ax1.set_xticks(range(batch_size))
ax1.set_yticks(range(batch_size))

# Right panel: Training dynamics
ax2 = axes[1]
epochs = np.arange(0, 100, 1)

# Simulate training curves
np.random.seed(42)
positive_similarity = 0.1 + 0.8 * (1 - np.exp(-epochs / 20)) + np.random.randn(len(epochs)) * 0.02
negative_similarity = 0.3 * np.exp(-epochs / 30) + np.random.randn(len(epochs)) * 0.02

ax2.plot(epochs, positive_similarity, linewidth=2.5, color=COLORS['green'],
         label='Positive Pairs (matched)', marker='o', markersize=3, markevery=10)
ax2.plot(epochs, negative_similarity, linewidth=2.5, color=COLORS['red'],
         label='Negative Pairs (mismatched)', marker='s', markersize=3, markevery=10)

ax2.set_xlabel('Training Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Average Similarity', fontsize=12, fontweight='bold')
ax2.set_title('Contrastive Learning Dynamics\n(Attraction of positives, repulsion of negatives)',
              fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(-0.1, 1.0)

# Add annotations
ax2.annotate('Positive pairs\nattract', xy=(50, 0.7), xytext=(60, 0.85),
            arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=2),
            fontsize=10, color=COLORS['green'], fontweight='bold')
ax2.annotate('Negative pairs\nrepel', xy=(50, 0.1), xytext=(65, 0.25),
            arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=2),
            fontsize=10, color=COLORS['red'], fontweight='bold')

plt.tight_layout()
plt.savefig('diagrams/contrastive_loss_illustration.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ contrastive_loss_illustration.png created")

# ============================================================================
# Diagram 5: Zero-Shot Classification Process
# ============================================================================
print("5. Creating zero_shot_classification.png...")

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Title
fig.suptitle('Zero-Shot Image Classification with CLIP', fontsize=16, fontweight='bold', y=0.98)

# Top row: Input image and text candidates
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
# Draw image box
img_box = FancyBboxPatch((2, 2), 6, 6, boxstyle="round,pad=0.2",
                         edgecolor=COLORS['blue'], facecolor='#E3F2FD', linewidth=3)
ax1.add_patch(img_box)
ax1.text(5, 5, '🐕\nQuery\nImage', ha='center', va='center',
         fontsize=14, fontweight='bold')
ax1.set_title('1. Input Image', fontsize=12, fontweight='bold', pad=10)

ax2 = fig.add_subplot(gs[0, 1:])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('2. Text Candidates', fontsize=12, fontweight='bold', pad=10)
classes = ['dog', 'cat', 'bird', 'car']
for i, cls in enumerate(classes):
    y_pos = 8 - i * 2.2
    text_box = FancyBboxPatch((1, y_pos-0.5), 8, 1.2, boxstyle="round,pad=0.1",
                             edgecolor=COLORS['gray'], facecolor='#FAFAFA', linewidth=2)
    ax2.add_patch(text_box)
    ax2.text(5, y_pos, f'"a photo of a {cls}"', ha='center', va='center',
            fontsize=11, fontweight='bold')

# Middle row: Embeddings
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
embed_box = FancyBboxPatch((1, 3), 8, 4, boxstyle="round,pad=0.2",
                          edgecolor=COLORS['blue'], facecolor='#E8EAF6', linewidth=2)
ax3.add_patch(embed_box)
ax3.text(5, 5, 'Image\nEmbedding\n[512-dim]', ha='center', va='center',
        fontsize=11, fontweight='bold')
ax3.set_title('3. Encode Image', fontsize=12, fontweight='bold', pad=10)

# Arrow down
ax3.annotate('', xy=(5, 2.5), xytext=(5, 8),
            arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['blue']))

ax4 = fig.add_subplot(gs[1, 1:])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis('off')
ax4.set_title('4. Encode Text Candidates', fontsize=12, fontweight='bold', pad=10)
for i in range(4):
    y_pos = 8 - i * 2.2
    embed_box = FancyBboxPatch((1, y_pos-0.5), 8, 1.2, boxstyle="round,pad=0.1",
                              edgecolor=COLORS['purple'], facecolor='#F3E5F5', linewidth=2)
    ax4.add_patch(embed_box)
    ax4.text(5, y_pos, f'Text Embedding {i+1} [512-dim]', ha='center', va='center',
            fontsize=10)

# Bottom row: Similarity computation and result
ax5 = fig.add_subplot(gs[2, :2])
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 10)
ax5.axis('off')
ax5.set_title('5. Compute Similarities', fontsize=12, fontweight='bold', pad=10)

similarities = [0.85, 0.23, 0.15, 0.12]
colors_bar = [COLORS['green'], COLORS['gray'], COLORS['gray'], COLORS['gray']]
y_positions = [7.5, 5.5, 3.5, 1.5]

for i, (sim, color, y_pos) in enumerate(zip(similarities, colors_bar, y_positions)):
    # Bar background
    bar_bg = FancyBboxPatch((2, y_pos-0.3), 6, 0.6, boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='#EEEEEE', linewidth=1)
    ax5.add_patch(bar_bg)

    # Bar fill
    bar_width = sim * 5.5
    bar_fill = FancyBboxPatch((2.2, y_pos-0.2), bar_width, 0.4, boxstyle="round,pad=0.02",
                             facecolor=color, linewidth=0, alpha=0.8)
    ax5.add_patch(bar_fill)

    # Label
    ax5.text(1, y_pos, classes[i], ha='right', va='center', fontsize=11, fontweight='bold')
    ax5.text(8.3, y_pos, f'{sim:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')

ax6 = fig.add_subplot(gs[2, 2])
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)
ax6.axis('off')
ax6.set_title('6. Prediction', fontsize=12, fontweight='bold', pad=10)

# Result box
result_box = FancyBboxPatch((1, 3), 8, 4, boxstyle="round,pad=0.3",
                           edgecolor=COLORS['green'], facecolor='#E8F5E9', linewidth=4)
ax6.add_patch(result_box)
ax6.text(5, 6, '✓', ha='center', va='center', fontsize=32, color=COLORS['green'])
ax6.text(5, 4, 'Dog\n(85% conf.)', ha='center', va='center',
        fontsize=13, fontweight='bold')

plt.savefig('diagrams/zero_shot_classification.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ zero_shot_classification.png created")

# ============================================================================
# Diagram 6: Multimodal Fusion Strategies
# ============================================================================
print("6. Creating fusion_strategies.png...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Multimodal Fusion Strategies', fontsize=16, fontweight='bold', y=0.98)

strategies = [
    ('Early Fusion', 0, 0),
    ('Late Fusion', 0, 1),
    ('Cross-Attention', 1, 0),
    ('Dual-Encoder', 1, 1)
]

for title, row, col in strategies:
    ax = axes[row, col]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    if title == 'Early Fusion':
        # Concatenate inputs before processing
        img_box = FancyBboxPatch((1, 7), 1.5, 1.5, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['blue'], facecolor='#BBDEFB', linewidth=2)
        ax.add_patch(img_box)
        ax.text(1.75, 7.75, 'Img', ha='center', va='center', fontsize=10, fontweight='bold')

        txt_box = FancyBboxPatch((3, 7), 1.5, 1.5, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['red'], facecolor='#FFCDD2', linewidth=2)
        ax.add_patch(txt_box)
        ax.text(3.75, 7.75, 'Text', ha='center', va='center', fontsize=10, fontweight='bold')

        # Concatenation
        concat_box = FancyBboxPatch((1.5, 4.5), 3, 1.5, boxstyle="round,pad=0.1",
                                   edgecolor=COLORS['purple'], facecolor='#E1BEE7', linewidth=2)
        ax.add_patch(concat_box)
        ax.text(3, 5.25, 'Concatenate', ha='center', va='center', fontsize=10, fontweight='bold')

        # Shared encoder
        encoder_box = FancyBboxPatch((1.5, 2), 3, 1.5, boxstyle="round,pad=0.1",
                                    edgecolor='black', facecolor='#CFD8DC', linewidth=2)
        ax.add_patch(encoder_box)
        ax.text(3, 2.75, 'Shared\nEncoder', ha='center', va='center', fontsize=10, fontweight='bold')

        # Output
        out_box = FancyBboxPatch((2.25, 0.3), 1.5, 0.8, boxstyle="round,pad=0.05",
                                edgecolor=COLORS['green'], facecolor='#C8E6C9', linewidth=2)
        ax.add_patch(out_box)
        ax.text(3, 0.7, 'Output', ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrows
        ax.annotate('', xy=(1.75, 4.5), xytext=(1.75, 7),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['blue']))
        ax.annotate('', xy=(3.75, 4.5), xytext=(3.75, 7),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red']))
        ax.annotate('', xy=(3, 2), xytext=(3, 4.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax.annotate('', xy=(3, 0.3), xytext=(3, 2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    elif title == 'Late Fusion':
        # Separate encoders, combine at end
        img_box = FancyBboxPatch((1, 7.5), 2, 1.2, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['blue'], facecolor='#BBDEFB', linewidth=2)
        ax.add_patch(img_box)
        ax.text(2, 8.1, 'Image', ha='center', va='center', fontsize=10, fontweight='bold')

        txt_box = FancyBboxPatch((6, 7.5), 2, 1.2, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['red'], facecolor='#FFCDD2', linewidth=2)
        ax.add_patch(txt_box)
        ax.text(7, 8.1, 'Text', ha='center', va='center', fontsize=10, fontweight='bold')

        # Separate encoders
        img_enc = FancyBboxPatch((1, 5), 2, 1.5, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['blue'], facecolor=COLORS['blue'], linewidth=2)
        ax.add_patch(img_enc)
        ax.text(2, 5.75, 'Image\nEncoder', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        txt_enc = FancyBboxPatch((6, 5), 2, 1.5, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['red'], facecolor=COLORS['red'], linewidth=2)
        ax.add_patch(txt_enc)
        ax.text(7, 5.75, 'Text\nEncoder', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Fusion
        fusion_box = FancyBboxPatch((3, 2.5), 3, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor=COLORS['purple'], facecolor='#E1BEE7', linewidth=2)
        ax.add_patch(fusion_box)
        ax.text(4.5, 3.1, 'Fusion Layer', ha='center', va='center', fontsize=10, fontweight='bold')

        # Output
        out_box = FancyBboxPatch((3.75, 0.5), 1.5, 0.8, boxstyle="round,pad=0.05",
                                edgecolor=COLORS['green'], facecolor='#C8E6C9', linewidth=2)
        ax.add_patch(out_box)
        ax.text(4.5, 0.9, 'Output', ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrows
        ax.annotate('', xy=(2, 5), xytext=(2, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['blue']))
        ax.annotate('', xy=(7, 5), xytext=(7, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red']))
        ax.annotate('', xy=(3.5, 3.1), xytext=(2, 5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['blue']))
        ax.annotate('', xy=(5.5, 3.1), xytext=(7, 5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red']))
        ax.annotate('', xy=(4.5, 0.5), xytext=(4.5, 2.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    elif title == 'Cross-Attention':
        # Cross-modal attention
        img_box = FancyBboxPatch((1, 7.5), 2, 1.2, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['blue'], facecolor='#BBDEFB', linewidth=2)
        ax.add_patch(img_box)
        ax.text(2, 8.1, 'Image', ha='center', va='center', fontsize=10, fontweight='bold')

        txt_box = FancyBboxPatch((6, 7.5), 2, 1.2, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['red'], facecolor='#FFCDD2', linewidth=2)
        ax.add_patch(txt_box)
        ax.text(7, 8.1, 'Text', ha='center', va='center', fontsize=10, fontweight='bold')

        # Features
        img_feat = FancyBboxPatch((1, 5.5), 2, 0.8, boxstyle="round,pad=0.05",
                                 edgecolor=COLORS['blue'], facecolor='#E3F2FD', linewidth=2)
        ax.add_patch(img_feat)
        ax.text(2, 5.9, 'K, V', ha='center', va='center', fontsize=9, fontweight='bold')

        txt_feat = FancyBboxPatch((6, 5.5), 2, 0.8, boxstyle="round,pad=0.05",
                                 edgecolor=COLORS['red'], facecolor='#FFEBEE', linewidth=2)
        ax.add_patch(txt_feat)
        ax.text(7, 5.9, 'Q', ha='center', va='center', fontsize=9, fontweight='bold')

        # Cross-attention
        attn_box = FancyBboxPatch((3, 3.5), 3, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor=COLORS['purple'], facecolor='#F3E5F5', linewidth=2)
        ax.add_patch(attn_box)
        ax.text(4.5, 4.25, 'Cross-\nAttention', ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Output
        out_box = FancyBboxPatch((3.75, 1), 1.5, 1, boxstyle="round,pad=0.05",
                                edgecolor=COLORS['green'], facecolor='#C8E6C9', linewidth=2)
        ax.add_patch(out_box)
        ax.text(4.5, 1.5, 'Attended\nFeatures', ha='center', va='center',
                fontsize=9, fontweight='bold')

        # Arrows
        ax.annotate('', xy=(2, 5.5), xytext=(2, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['blue']))
        ax.annotate('', xy=(7, 5.5), xytext=(7, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red']))
        ax.annotate('', xy=(3.5, 4.25), xytext=(2, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['blue']))
        ax.annotate('', xy=(5.5, 4.25), xytext=(7, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red']))
        ax.annotate('', xy=(4.5, 1), xytext=(4.5, 3.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    else:  # Dual-Encoder (CLIP-style)
        img_box = FancyBboxPatch((1, 7.5), 2, 1.2, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['blue'], facecolor='#BBDEFB', linewidth=2)
        ax.add_patch(img_box)
        ax.text(2, 8.1, 'Image', ha='center', va='center', fontsize=10, fontweight='bold')

        txt_box = FancyBboxPatch((6, 7.5), 2, 1.2, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['red'], facecolor='#FFCDD2', linewidth=2)
        ax.add_patch(txt_box)
        ax.text(7, 8.1, 'Text', ha='center', va='center', fontsize=10, fontweight='bold')

        # Separate encoders
        img_enc = FancyBboxPatch((1, 5), 2, 1.5, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['blue'], facecolor=COLORS['blue'], linewidth=2)
        ax.add_patch(img_enc)
        ax.text(2, 5.75, 'Vision\nEncoder', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        txt_enc = FancyBboxPatch((6, 5), 2, 1.5, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['red'], facecolor=COLORS['red'], linewidth=2)
        ax.add_patch(txt_enc)
        ax.text(7, 5.75, 'Text\nEncoder', ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')

        # Joint space
        joint_box = FancyBboxPatch((3, 2), 3, 2, boxstyle="round,pad=0.2",
                                  edgecolor=COLORS['purple'], facecolor='#F3E5F5', linewidth=3)
        ax.add_patch(joint_box)
        ax.text(4.5, 3, 'Joint\nEmbedding\nSpace', ha='center', va='center',
                fontsize=10, fontweight='bold')

        # Arrows
        ax.annotate('', xy=(2, 5), xytext=(2, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['blue']))
        ax.annotate('', xy=(7, 5), xytext=(7, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red']))
        ax.annotate('', xy=(3.5, 3.5), xytext=(2, 5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['blue']))
        ax.annotate('', xy=(5.5, 3.5), xytext=(7, 5),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['red']))

        # Similarity computation
        ax.text(4.5, 0.8, 'Cosine Similarity', ha='center', va='center',
               fontsize=10, fontweight='bold', style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.savefig('diagrams/fusion_strategies.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ fusion_strategies.png created")

print("\n" + "="*60)
print("All diagrams generated successfully!")
print("="*60)
print(f"\nDiagrams saved to: diagrams/")
print("\nFiles created:")
print("  1. multimodal_overview.png")
print("  2. temperature_effect.png")
print("  3. cross_attention_visualization.png")
print("  4. contrastive_loss_illustration.png")
print("  5. zero_shot_classification.png")
print("  6. fusion_strategies.png")

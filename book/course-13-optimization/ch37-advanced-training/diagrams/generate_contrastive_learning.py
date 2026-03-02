"""
Generate contrastive learning visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Create figure with two subplots
fig = plt.figure(figsize=(14, 6))

# ============= Left: Contrastive Learning Process =============
ax1 = plt.subplot(1, 2, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Contrastive Learning Process (SimCLR)', fontsize=13, fontweight='bold', pad=15)

# Original image
img_box = FancyBboxPatch((1, 7), 1.5, 1.5,
                        boxstyle="round,pad=0.05",
                        edgecolor=colors['blue'],
                        facecolor='#E3F2FD',
                        linewidth=2)
ax1.add_patch(img_box)
ax1.text(1.75, 7.75, 'Image\nx', fontsize=10, ha='center', va='center', fontweight='bold')

# Augmentation 1
aug1_arrow = FancyArrowPatch((2.5, 8), (3.5, 8.5),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=2, color=colors['gray'])
ax1.add_patch(aug1_arrow)
ax1.text(3, 8.7, 'Aug 1', fontsize=8, ha='center', style='italic')

aug1_box = FancyBboxPatch((3.5, 7.8), 1.3, 1.3,
                         boxstyle="round,pad=0.05",
                         edgecolor=colors['orange'],
                         facecolor='#FFF3E0',
                         linewidth=2)
ax1.add_patch(aug1_box)
ax1.text(4.15, 8.45, 'x_i', fontsize=10, ha='center', va='center', fontweight='bold')

# Augmentation 2
aug2_arrow = FancyArrowPatch((2.5, 7.5), (3.5, 6.9),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=2, color=colors['gray'])
ax1.add_patch(aug2_arrow)
ax1.text(3, 6.6, 'Aug 2', fontsize=8, ha='center', style='italic')

aug2_box = FancyBboxPatch((3.5, 6), 1.3, 1.3,
                         boxstyle="round,pad=0.05",
                         edgecolor=colors['orange'],
                         facecolor='#FFF3E0',
                         linewidth=2)
ax1.add_patch(aug2_box)
ax1.text(4.15, 6.65, 'x_j', fontsize=10, ha='center', va='center', fontweight='bold')

# Encoder
enc1_arrow = FancyArrowPatch((4.8, 8.45), (6, 8.45),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=2, color=colors['gray'])
ax1.add_patch(enc1_arrow)
ax1.text(5.4, 8.8, 'Encoder', fontsize=8, ha='center', style='italic')

enc2_arrow = FancyArrowPatch((4.8, 6.65), (6, 6.65),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=2, color=colors['gray'])
ax1.add_patch(enc2_arrow)
ax1.text(5.4, 6.3, 'Encoder', fontsize=8, ha='center', style='italic')

# Embeddings
emb1_circle = Circle((6.8, 8.45), 0.5,
                    edgecolor=colors['green'],
                    facecolor='#C8E6C9',
                    linewidth=2)
ax1.add_patch(emb1_circle)
ax1.text(6.8, 8.45, 'z_i', fontsize=10, ha='center', va='center', fontweight='bold')

emb2_circle = Circle((6.8, 6.65), 0.5,
                    edgecolor=colors['green'],
                    facecolor='#C8E6C9',
                    linewidth=2)
ax1.add_patch(emb2_circle)
ax1.text(6.8, 6.65, 'z_j', fontsize=10, ha='center', va='center', fontweight='bold')

# Pull together
pull_arrow1 = FancyArrowPatch((6.8, 7.95), (6.8, 7.45),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=3, color=colors['green'])
ax1.add_patch(pull_arrow1)
pull_arrow2 = FancyArrowPatch((6.8, 7.15), (6.8, 7.65),
                             arrowstyle='->', mutation_scale=20,
                             linewidth=3, color=colors['green'])
ax1.add_patch(pull_arrow2)
ax1.text(7.8, 7.55, 'Pull together\n(positive pair)',
        fontsize=9, ha='left', va='center', color=colors['green'], fontweight='bold')

# Negative examples
neg_y = 4.5
ax1.text(1.75, neg_y + 0.3, 'Different Images', fontsize=9, ha='center', fontweight='bold')
for i in range(3):
    neg_box = FancyBboxPatch((0.5 + i*1, neg_y - 0.8), 0.8, 0.8,
                            boxstyle="round,pad=0.03",
                            edgecolor=colors['gray'],
                            facecolor='#F5F5F5',
                            linewidth=1.5, alpha=0.6)
    ax1.add_patch(neg_box)

neg_emb_y = 3
for i in range(3):
    neg_circle = Circle((0.9 + i*1, neg_emb_y), 0.35,
                       edgecolor=colors['red'],
                       facecolor='#FFCDD2',
                       linewidth=1.5, alpha=0.6)
    ax1.add_patch(neg_circle)

# Push away
push_arrow = FancyArrowPatch((6.8, 6.15), (2, 3.3),
                            arrowstyle='->', mutation_scale=15,
                            linewidth=2.5, color=colors['red'],
                            linestyle='--')
ax1.add_patch(push_arrow)
ax1.text(4.5, 4.5, 'Push away\n(negative pairs)',
        fontsize=9, ha='center', va='center', color=colors['red'], fontweight='bold')

# NT-Xent Loss
loss_box = FancyBboxPatch((3, 1), 4, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['purple'],
                         facecolor='#F3E5F5',
                         linewidth=2)
ax1.add_patch(loss_box)
loss_formula = r'$\mathcal{L} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_k \exp(\mathrm{sim}(z_i, z_k)/\tau)}$'
ax1.text(5, 1.8, 'NT-Xent Loss:', fontsize=10, ha='center', va='top', fontweight='bold')
ax1.text(5, 1.3, loss_formula, fontsize=10, ha='center', va='center')

# ============= Right: Embedding Space Visualization =============
ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Learned Embedding Space', fontsize=13, fontweight='bold', pad=15)

# Generate synthetic embedding data for 3 classes
np.random.seed(42)
n_samples_per_class = 30
n_classes = 3

embeddings = []
labels = []

# Create clusters for each class
centers = [(0, 0), (3, 3), (-2, 4)]
cluster_colors = [colors['blue'], colors['orange'], colors['green']]

for i in range(n_classes):
    # Generate points around cluster center
    x = np.random.randn(n_samples_per_class) * 0.5 + centers[i][0]
    y = np.random.randn(n_samples_per_class) * 0.5 + centers[i][1]
    embeddings.extend(zip(x, y))
    labels.extend([i] * n_samples_per_class)

embeddings = np.array(embeddings)
labels = np.array(labels)

# Plot embeddings
for i in range(n_classes):
    mask = labels == i
    ax2.scatter(embeddings[mask, 0], embeddings[mask, 1],
               c=cluster_colors[i], label=f'Class {i}',
               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

# Draw some positive pair connections
for _ in range(8):
    class_idx = np.random.randint(0, n_classes)
    class_mask = labels == class_idx
    class_points = embeddings[class_mask]
    if len(class_points) >= 2:
        idx1, idx2 = np.random.choice(len(class_points), 2, replace=False)
        p1, p2 = class_points[idx1], class_points[idx2]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=cluster_colors[class_idx], alpha=0.3, linewidth=1.5, linestyle='-')

# Draw some negative pair repulsion (dashed lines between different classes)
for _ in range(5):
    c1, c2 = np.random.choice(n_classes, 2, replace=False)
    p1 = embeddings[labels == c1][np.random.randint(0, n_samples_per_class)]
    p2 = embeddings[labels == c2][np.random.randint(0, n_samples_per_class)]
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]],
            color=colors['red'], alpha=0.2, linewidth=1, linestyle='--')

ax2.set_xlabel('Embedding Dimension 1', fontsize=11)
ax2.set_ylabel('Embedding Dimension 2', fontsize=11)
ax2.legend(loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')

# Add annotation
ax2.text(0.5, 0.98, 'Similar samples cluster together\nDissimilar samples pushed apart',
        transform=ax2.transAxes, fontsize=9, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('contrastive_learning_process.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: contrastive_learning_process.png")
plt.close()

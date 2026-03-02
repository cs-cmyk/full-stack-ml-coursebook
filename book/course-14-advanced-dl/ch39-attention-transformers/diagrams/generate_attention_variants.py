import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Create figure comparing MHA, MQA, and GQA architectures
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Common parameters
num_heads = 8
head_height = 0.8
head_spacing = 1.0

# Color scheme - using consistent palette
color_q = '#2196F3'  # Blue for queries
color_k = '#F44336'  # Red for keys
color_v = '#4CAF50'  # Green for values

def draw_projections(ax, title, kv_config):
    """
    Draw Q, K, V projections for different attention variants.
    kv_config: list of (head_idx, kv_group_idx) tuples
    """
    ax.set_xlim(0, 4)
    ax.set_ylim(0, num_heads * head_spacing)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Draw query heads (always independent)
    for i in range(num_heads):
        y = i * head_spacing + 0.1
        rect = mpatches.Rectangle((0.2, y), 0.6, head_height,
                                   facecolor=color_q, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(0.5, y + head_height/2, f'Q{i}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Draw key and value projections based on config
    unique_kv = sorted(set(group_idx for _, group_idx in kv_config))
    kv_colors_k = {group_idx: color_k for group_idx in unique_kv}
    kv_colors_v = {group_idx: color_v for group_idx in unique_kv}

    for i, (head_idx, kv_group_idx) in enumerate(kv_config):
        y = head_idx * head_spacing + 0.1

        # Draw key
        rect_k = mpatches.Rectangle((1.5, y), 0.6, head_height,
                                     facecolor=kv_colors_k[kv_group_idx],
                                     edgecolor='black', linewidth=2,
                                     alpha=0.7)
        ax.add_patch(rect_k)
        ax.text(1.8, y + head_height/2, f'K{kv_group_idx}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # Draw value
        rect_v = mpatches.Rectangle((2.8, y), 0.6, head_height,
                                     facecolor=kv_colors_v[kv_group_idx],
                                     edgecolor='black', linewidth=2,
                                     alpha=0.7)
        ax.add_patch(rect_v)
        ax.text(3.1, y + head_height/2, f'V{kv_group_idx}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

        # Draw connection line
        ax.plot([0.8, 1.5], [y + head_height/2, y + head_height/2],
                'k-', linewidth=1, alpha=0.3)
        ax.plot([2.2, 2.8], [y + head_height/2, y + head_height/2],
                'k-', linewidth=1, alpha=0.3)

    # Add labels
    ax.text(0.5, -0.5, 'Queries', ha='center', fontsize=12, fontweight='bold')
    ax.text(1.8, -0.5, 'Keys', ha='center', fontsize=12, fontweight='bold')
    ax.text(3.1, -0.5, 'Values', ha='center', fontsize=12, fontweight='bold')

# Multi-Head Attention: Each head has its own K, V
mha_config = [(i, i) for i in range(num_heads)]
draw_projections(axes[0], 'Multi-Head Attention (MHA)', mha_config)

# Multi-Query Attention: All heads share single K, V
mqa_config = [(i, 0) for i in range(num_heads)]
draw_projections(axes[1], 'Multi-Query Attention (MQA)', mqa_config)

# Grouped-Query Attention: 8 heads, 2 groups
gqa_config = [(i, i // 4) for i in range(num_heads)]
draw_projections(axes[2], 'Grouped-Query Attention (GQA)\n8 heads, 2 groups', gqa_config)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch39/diagrams/attention_variants_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: attention_variants_comparison.png")
plt.close()

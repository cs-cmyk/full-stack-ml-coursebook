import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Simplified visual comparison
architectures = [
    ('GCN: Fixed Normalized Sum', 'All neighbors\nequal weight', 'Efficient\nTransductive'),
    ('GraphSAGE: Learnable Aggregation', 'Samples neighbors\nMean/Max/LSTM', 'Scalable\nInductive'),
    ('GAT: Attention-Weighted Sum', 'Learns neighbor\nimportance α_ij', 'Expressive\nInterpretable'),
    ('GIN: Sum + MLP', 'Sum aggregation\nInjective MLP', 'Maximally\nExpressive')
]

for idx, (ax, (title, agg, prop)) in enumerate(zip(axes.flat, architectures)):
    ax.text(0.5, 0.7, title, ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(0.5, 0.45, agg, ha='center', va='center', fontsize=10, style='italic')
    ax.text(0.5, 0.2, prop, ha='center', va='center', fontsize=9, color='darkblue')

    # Add visual representation
    if 'GCN' in title:
        circle = plt.Circle((0.5, 0.9), 0.08, color='#2196F3', ec='black', linewidth=2)
        ax.add_patch(circle)
        for angle in [0, 60, 120, 180, 240, 300]:
            x = 0.5 + 0.15 * np.cos(np.radians(angle))
            y = 0.9 + 0.15 * np.sin(np.radians(angle))
            small_circle = plt.Circle((x, y), 0.04, color='#4CAF50', ec='black')
            ax.add_patch(small_circle)
            ax.plot([0.5, x], [0.9, y], 'k-', linewidth=1)

    elif 'GraphSAGE' in title:
        circle = plt.Circle((0.5, 0.9), 0.08, color='#2196F3', ec='black', linewidth=2)
        ax.add_patch(circle)
        # Show sampling (only some neighbors)
        for i, angle in enumerate([0, 90, 180]):
            x = 0.5 + 0.15 * np.cos(np.radians(angle))
            y = 0.9 + 0.15 * np.sin(np.radians(angle))
            small_circle = plt.Circle((x, y), 0.04, color='#4CAF50', ec='black', linewidth=2)
            ax.add_patch(small_circle)
            ax.plot([0.5, x], [0.9, y], 'k-', linewidth=2)
        # Faded unsampled neighbors
        for angle in [270]:
            x = 0.5 + 0.15 * np.cos(np.radians(angle))
            y = 0.9 + 0.15 * np.sin(np.radians(angle))
            small_circle = plt.Circle((x, y), 0.04, color='#607D8B', ec='gray', alpha=0.3)
            ax.add_patch(small_circle)
            ax.plot([0.5, x], [0.9, y], 'gray', linewidth=0.5, linestyle='--', alpha=0.3)

    elif 'GAT' in title:
        circle = plt.Circle((0.5, 0.9), 0.08, color='#2196F3', ec='black', linewidth=2)
        ax.add_patch(circle)
        # Variable edge thickness (attention)
        attentions = [3, 1, 2]
        for i, angle in enumerate([0, 120, 240]):
            x = 0.5 + 0.15 * np.cos(np.radians(angle))
            y = 0.9 + 0.15 * np.sin(np.radians(angle))
            small_circle = plt.Circle((x, y), 0.04, color='#4CAF50', ec='black')
            ax.add_patch(small_circle)
            ax.plot([0.5, x], [0.9, y], 'k-', linewidth=attentions[i])

    elif 'GIN' in title:
        circle = plt.Circle((0.5, 0.9), 0.08, color='#2196F3', ec='black', linewidth=3)
        ax.add_patch(circle)
        for angle in [0, 90, 180, 270]:
            x = 0.5 + 0.15 * np.cos(np.radians(angle))
            y = 0.9 + 0.15 * np.sin(np.radians(angle))
            small_circle = plt.Circle((x, y), 0.04, color='#4CAF50', ec='black')
            ax.add_patch(small_circle)
            ax.plot([0.5, x], [0.9, y], 'k-', linewidth=1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.suptitle('GNN Architecture Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-20/ch57/diagrams/architecture_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Architecture comparison diagram created")

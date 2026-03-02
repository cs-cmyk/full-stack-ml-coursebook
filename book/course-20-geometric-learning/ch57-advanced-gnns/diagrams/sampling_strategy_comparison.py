import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Exponential growth without sampling
ax = axes[0]
layers = np.arange(0, 5)
avg_degree = 50
nodes_full = avg_degree ** layers
ax.plot(layers, nodes_full, 'o-', linewidth=3, markersize=12, color='#F44336', label='Full neighborhood')
ax.set_yscale('log')
ax.set_xlabel('Layer Depth', fontsize=13)
ax.set_ylabel('Nodes to Process (log scale)', fontsize=13)
ax.set_title('Full-Batch: Exponential Growth', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xticks(layers)

# Add annotations
for i, (x, y) in enumerate(zip(layers, nodes_full)):
    if i > 0:
        ax.annotate(f'{int(y):,}', xy=(x, y), xytext=(x-0.3, y*2),
                   fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Right: Controlled growth with sampling
ax = axes[1]
sample_size = 10
nodes_sampled = sample_size ** layers
ax.plot(layers, nodes_sampled, 's-', linewidth=3, markersize=12, color='#4CAF50', label='Sampled neighborhood (k=10)')
ax.set_yscale('log')
ax.set_xlabel('Layer Depth', fontsize=13)
ax.set_ylabel('Nodes to Process (log scale)', fontsize=13)
ax.set_title('Neighbor Sampling: Controlled Growth', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_xticks(layers)

# Add annotations
for i, (x, y) in enumerate(zip(layers, nodes_sampled)):
    if i > 0:
        ax.annotate(f'{int(y):,}', xy=(x, y), xytext=(x-0.3, y*2),
                   fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-20/ch57/diagrams/sampling_strategy_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Sampling strategy visualization saved")

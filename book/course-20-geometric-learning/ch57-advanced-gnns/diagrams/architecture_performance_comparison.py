import matplotlib.pyplot as plt
import numpy as np

# Simulated results (typical performance on Cora dataset)
architectures = ['GCN', 'GraphSAGE', 'GAT', 'GIN']
test_accs = [0.801, 0.814, 0.823, 0.818]

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']
bars = ax.bar(architectures, test_accs, color=colors, alpha=0.8)

ax.set_ylabel('Test Accuracy', fontsize=13)
ax.set_xlabel('Architecture', fontsize=13)
ax.set_title('GNN Architecture Comparison on Cora Dataset', fontsize=14, fontweight='bold')
ax.set_ylim(0.6, 0.85)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bar, acc in zip(bars, test_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-20/ch57/diagrams/architecture_performance_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Performance comparison saved")

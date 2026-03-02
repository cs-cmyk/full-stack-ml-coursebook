import matplotlib.pyplot as plt
import numpy as np

# Simulated benchmark results
strategies = ['Neighbor\nSampling', 'Cluster-GCN', 'GraphSAINT']
avg_times = [12.5, 18.3, 22.1]  # seconds per epoch
avg_memories = [2400, 3200, 2800]  # MB
test_accs = [0.718, 0.725, 0.731]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#2196F3', '#4CAF50', '#FF9800']

# Time vs Accuracy
ax = axes[0]
for i, (strategy, time_val, acc) in enumerate(zip(strategies, avg_times, test_accs)):
    ax.scatter(time_val, acc, s=300, c=colors[i], label=strategy.replace('\n', ' '),
              alpha=0.8, edgecolors='black', linewidth=2)
    ax.annotate(strategy, xy=(time_val, acc), xytext=(10, 10),
               textcoords='offset points', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

ax.set_xlabel('Training Time (s/epoch)', fontsize=13)
ax.set_ylabel('Test Accuracy', fontsize=13)
ax.set_title('Accuracy vs Training Time Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(10, 25)
ax.set_ylim(0.71, 0.74)

# Memory vs Accuracy
ax = axes[1]
for i, (strategy, mem, acc) in enumerate(zip(strategies, avg_memories, test_accs)):
    ax.scatter(mem, acc, s=300, c=colors[i], label=strategy.replace('\n', ' '),
              alpha=0.8, edgecolors='black', linewidth=2)
    ax.annotate(strategy, xy=(mem, acc), xytext=(10, 10),
               textcoords='offset points', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))

ax.set_xlabel('Peak Memory Usage (MB)', fontsize=13)
ax.set_ylabel('Test Accuracy', fontsize=13)
ax.set_title('Accuracy vs Memory Usage Trade-off', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(2200, 3400)
ax.set_ylim(0.71, 0.74)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-20/ch57/diagrams/sampling_tradeoffs.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Sampling tradeoffs visualization saved")

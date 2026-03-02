import matplotlib.pyplot as plt
import numpy as np

# Simulated results data (from Solution 3)
methods = ['Bi-encoder', 'Top-20', 'Top-50', 'Top-100']
latencies = [48.3, 412.5, 856.7, 1634.2]
ndcgs = [0.687, 0.761, 0.803, 0.821]

# Per-query precision data
test_queries = 8
np.random.seed(42)
precisions_bi = [0.6, 0.7, 0.8, 0.7, 0.7, 0.6, 0.8, 0.6]  # Simulated
precisions_50 = [0.8, 0.9, 0.9, 0.8, 0.8, 0.8, 0.9, 0.7]  # Simulated

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: nDCG vs Latency
ax1.plot(latencies, ndcgs, 'o-', linewidth=2, markersize=10, color='#2196F3')
for i, method in enumerate(methods):
    ax1.annotate(method, (latencies[i], ndcgs[i]), textcoords="offset points",
                xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
ax1.set_xlabel('Latency (ms)', fontweight='bold', fontsize=12)
ax1.set_ylabel('nDCG@10', fontweight='bold', fontsize=12)
ax1.set_title('Accuracy vs Latency Tradeoff', fontweight='bold', fontsize=13)
ax1.grid(True, alpha=0.3)

# Plot 2: Precision comparison
x = np.arange(test_queries)
width = 0.35

ax2.bar(x - width/2, precisions_bi, width, label='Bi-encoder', alpha=0.8, color='#4CAF50')
ax2.bar(x + width/2, precisions_50, width, label='Two-stage (top-50)', alpha=0.8, color='#FF9800')
ax2.set_xlabel('Query Index', fontweight='bold', fontsize=12)
ax2.set_ylabel('Precision@10', fontweight='bold', fontsize=12)
ax2.set_title('Per-Query Precision Comparison', fontweight='bold', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels([f'Q{i+1}' for i in range(test_queries)])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch49/diagrams/two_stage_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Diagram 4 saved: two_stage_analysis.png")

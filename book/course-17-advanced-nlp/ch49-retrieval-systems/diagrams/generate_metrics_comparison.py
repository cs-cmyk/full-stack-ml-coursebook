import matplotlib.pyplot as plt
import numpy as np

# Simulated results data (from Part 4)
systems = ['BM25', 'Bi-Encoder', 'Cross-Encoder', 'Hybrid-RRF']
results = {
    'BM25': {'Recall@10': 0.057, 'Recall@20': 0.105, 'Recall@50': 0.231, 'MRR': 0.714, 'nDCG@10': 0.623},
    'Bi-Encoder': {'Recall@10': 0.080, 'Recall@20': 0.148, 'Recall@50': 0.289, 'MRR': 0.833, 'nDCG@10': 0.724},
    'Cross-Encoder': {'Recall@10': 0.097, 'Recall@20': 0.165, 'Recall@50': 0.289, 'MRR': 0.909, 'nDCG@10': 0.821},
    'Hybrid-RRF': {'Recall@10': 0.091, 'Recall@20': 0.159, 'Recall@50': 0.308, 'MRR': 0.877, 'nDCG@10': 0.789}
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Recall@k
metrics_recall = ['Recall@10', 'Recall@20', 'Recall@50']
x_pos = np.arange(len(systems))
width = 0.25

colors_recall = ['#2196F3', '#4CAF50', '#FF9800']
for i, metric in enumerate(metrics_recall):
    values = [results[sys][metric] for sys in systems]
    axes[0].bar(x_pos + i*width, values, width,
               label=metric.replace('Recall@', 'k='), color=colors_recall[i], alpha=0.8)

axes[0].set_xlabel('System', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Recall', fontweight='bold', fontsize=12)
axes[0].set_title('Recall at Different k Values', fontweight='bold', fontsize=13)
axes[0].set_xticks(x_pos + width)
axes[0].set_xticklabels(systems, rotation=15, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: MRR
mrr_values = [results[sys]['MRR'] for sys in systems]
colors = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0']
axes[1].bar(systems, mrr_values, color=colors, alpha=0.7)
axes[1].set_ylabel('MRR', fontweight='bold', fontsize=12)
axes[1].set_title('Mean Reciprocal Rank', fontweight='bold', fontsize=13)
axes[1].set_xticklabels(systems, rotation=15, ha='right')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_ylim(0, 1)

# Plot 3: nDCG@10
ndcg_values = [results[sys]['nDCG@10'] for sys in systems]
axes[2].bar(systems, ndcg_values, color=colors, alpha=0.7)
axes[2].set_ylabel('nDCG@10', fontweight='bold', fontsize=12)
axes[2].set_title('Normalized Discounted Cumulative Gain', fontweight='bold', fontsize=13)
axes[2].set_xticklabels(systems, rotation=15, ha='right')
axes[2].grid(axis='y', alpha=0.3)
axes[2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch49/diagrams/metrics_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Diagram 3 saved: metrics_comparison.png")

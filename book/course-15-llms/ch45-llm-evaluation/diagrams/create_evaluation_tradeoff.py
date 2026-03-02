import matplotlib.pyplot as plt
import numpy as np

# Create evaluation cost-quality tradeoff visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Define evaluation methods with (cost, correlation) pairs
methods = {
    'ROUGE': (0.001, 0.40, 'o'),
    'BERTScore': (0.01, 0.60, 's'),
    'GPT-3.5 Judge': (0.05, 0.75, '^'),
    'GPT-4 Judge': (0.10, 0.85, 'D'),
    'GPT-4 Judge\n(debiased)': (0.20, 0.87, 'D'),
    'Human Eval': (5.00, 1.00, '*')
}

# Use consistent color palette
colors = ['#2196F3', '#9C27B0', '#FF9800', '#4CAF50', '#F44336', '#607D8B']

for idx, (method, (cost, corr, marker)) in enumerate(methods.items()):
    color = colors[idx % len(colors)]
    ax.scatter(cost, corr, s=200, marker=marker, alpha=0.7,
               edgecolors='black', linewidth=1.5, color=color)
    ax.annotate(method, (cost, corr), xytext=(10, 5), textcoords='offset points',
                fontsize=9, fontweight='bold')

ax.set_xscale('log')
ax.set_xlabel('Cost per Evaluation (USD, log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Human Agreement Correlation', fontsize=12, fontweight='bold')
ax.set_title('LLM Evaluation Methods: Cost vs. Quality Tradeoff', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0.0005, 10)
ax.set_ylim(0.3, 1.05)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-15/ch45/diagrams/evaluation_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: evaluation_tradeoff.png")

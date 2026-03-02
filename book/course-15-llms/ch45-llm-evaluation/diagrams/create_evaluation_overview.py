import matplotlib.pyplot as plt
import numpy as np

# Create evaluation taxonomy visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Evaluation dimensions
dimensions = ['Capability', 'Safety', 'Alignment']
aspects = [
    ['Knowledge\n(MMLU)', 'Reasoning\n(HellaSwag)', 'Code\n(HumanEval)', 'Math\n(GSM8K)'],
    ['Hallucination', 'Toxicity', 'Bias', 'Adversarial\nRobustness'],
    ['Helpfulness', 'Harmlessness', 'Honesty', 'Instruction\nFollowing']
]

colors_dim = ['#2196F3', '#F44336', '#4CAF50']  # Using consistent color palette
y_positions = [3, 2, 1]

for i, (dim, asp, color) in enumerate(zip(dimensions, aspects, colors_dim)):
    ax1.barh(y_positions[i], 1, left=0, height=0.5, color=color, alpha=0.7, label=dim)
    ax1.text(0.5, y_positions[i], dim, ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Add aspects as text
    for j, a in enumerate(asp):
        ax1.text(1.2 + j*0.4, y_positions[i], a, ha='left', va='center', fontsize=8)

ax1.set_ylim(0.5, 3.5)
ax1.set_xlim(0, 3)
ax1.axis('off')
ax1.set_title('LLM Evaluation Taxonomy', fontsize=14, fontweight='bold')

# Right: Cost vs Reliability tradeoff
methods = ['Automated\nMetrics', 'Benchmark\nSuites', 'LLM-as-Judge',
           'Crowdsourced\nHuman', 'Expert\nHuman']
costs = [1, 2, 3, 7, 9]  # Arbitrary scale
reliability = [4, 6, 7, 7.5, 9]  # Arbitrary scale

colors_scatter = ['#2196F3', '#9C27B0', '#FF9800', '#F39C12', '#F44336']  # Using consistent palette

for method, cost, rel, color in zip(methods, costs, reliability, colors_scatter):
    ax2.scatter(cost, rel, s=500, alpha=0.6, color=color)
    ax2.annotate(method, (cost, rel), ha='center', va='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Cost (Time, Money, Effort) →', fontsize=11)
ax2.set_ylabel('Reliability / Agreement with Human Judgment →', fontsize=11)
ax2.set_title('Evaluation Method Tradeoffs', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-15/ch45/diagrams/evaluation_overview.png', dpi=150, bbox_inches='tight')
plt.close()

print("Created: evaluation_overview.png")

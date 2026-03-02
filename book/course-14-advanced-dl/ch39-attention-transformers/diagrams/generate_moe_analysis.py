import matplotlib.pyplot as plt
import numpy as np

# Simulate MoE vs Dense training curves
np.random.seed(42)
epochs = np.arange(1, 51)

# Dense model: slower convergence, higher final loss
dense_initial = 2.5
dense_final = 0.45
dense_test_losses = dense_initial * np.exp(-0.06 * epochs) + dense_final + np.random.normal(0, 0.02, len(epochs))

# MoE model: faster convergence, lower final loss
moe_initial = 2.3
moe_final = 0.35
moe_test_losses = moe_initial * np.exp(-0.08 * epochs) + moe_final + np.random.normal(0, 0.015, len(epochs))

# Simulate expert usage (with some imbalance)
num_experts = 4
# Realistic usage pattern: not perfectly balanced
expert_usage = np.array([0.28, 0.24, 0.26, 0.22])

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color scheme - using consistent palette
color_moe = '#4CAF50'  # Green
color_dense = '#FF9800'  # Orange
color_expert = '#2196F3'  # Blue
color_ideal = '#F44336'  # Red

# Training curves
ax = axes[0]
ax.plot(epochs, moe_test_losses, label='MoE', linewidth=2.5, color=color_moe)
ax.plot(epochs, dense_test_losses, label='Dense', linewidth=2.5, color=color_dense)
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Test MSE', fontsize=13, fontweight='bold')
ax.set_title('Learning Curves: MoE vs Dense', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(alpha=0.3, linestyle='--')
ax.set_ylim(0, 2.8)

# Add annotation showing MoE advantage
ax.annotate('MoE converges faster',
            xy=(30, moe_test_losses[29]), xytext=(35, 1.2),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=11, bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='lightyellow', alpha=0.8))

# Expert usage
ax = axes[1]
bars = ax.bar(range(num_experts), expert_usage * 100,
              color=color_expert, edgecolor='black', linewidth=2, alpha=0.8)
ax.axhline(100 / num_experts, color=color_ideal, linestyle='--',
           linewidth=2.5, label='Ideal balance (25%)')

# Add percentage labels on bars
for i, (bar, usage) in enumerate(zip(bars, expert_usage)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{usage*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Expert Index', fontsize=13, fontweight='bold')
ax.set_ylabel('Usage (%)', fontsize=13, fontweight='bold')
ax.set_title('Expert Utilization', fontsize=14, fontweight='bold')
ax.set_xticks(range(num_experts))
ax.set_xticklabels([f'Expert {i}' for i in range(num_experts)], fontsize=11)
ax.legend(fontsize=12, loc='upper right')
ax.grid(alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(0, 35)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch39/diagrams/moe_analysis.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: moe_analysis.png")
plt.close()

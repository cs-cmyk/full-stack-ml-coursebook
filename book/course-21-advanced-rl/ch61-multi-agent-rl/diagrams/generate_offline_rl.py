"""
Generate Offline RL comparison visualization (CQL vs DQN)
"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate evaluation performance data
n_evaluations = 50

# DQN: Shows optimistic overestimation and instability
dqn_returns = []
for i in range(n_evaluations):
    # DQN starts well but becomes unstable
    if i < 20:
        base = 50 + i * 5
        noise = np.random.normal(0, 10)
    else:
        # Divergence due to overestimation
        base = 150 - (i - 20) * 3
        noise = np.random.normal(0, 20)
    dqn_returns.append(base + noise)

# CQL: More conservative but stable
cql_returns = []
for i in range(n_evaluations):
    # CQL gradually improves and stabilizes
    base = 40 + i * 3.5
    if i > 30:
        base = 145 + (i - 30) * 0.5
    noise = np.random.normal(0, 8)
    cql_returns.append(base + noise)

# Smooth the curves
def smooth(y, window=5):
    return np.convolve(y, np.ones(window)/window, mode='valid')

dqn_smooth = smooth(dqn_returns)
cql_smooth = smooth(cql_returns)

# Create the plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Raw performance
axes[0].plot(dqn_returns, alpha=0.3, color='#F44336', label='DQN (raw)')
axes[0].plot(cql_returns, alpha=0.3, color='#4CAF50', label='CQL (raw)')
axes[0].plot(range(2, len(dqn_returns)-2), dqn_smooth,
             color='#F44336', linewidth=2, label='DQN (smoothed)')
axes[0].plot(range(2, len(cql_returns)-2), cql_smooth,
             color='#4CAF50', linewidth=2, label='CQL (smoothed)')
axes[0].set_xlabel('Evaluation Episode', fontsize=12)
axes[0].set_ylabel('Average Return', fontsize=12)
axes[0].set_title('Offline RL: CQL vs DQN on Fixed Dataset', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Add annotation
axes[0].annotate('DQN diverges due to\nout-of-distribution overestimation',
                xy=(35, 80), xytext=(25, 30),
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5),
                fontsize=10, color='#F44336')

axes[0].annotate('CQL remains stable\nwith conservative estimates',
                xy=(40, 160), xytext=(15, 180),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5),
                fontsize=10, color='#4CAF50')

# Plot 2: Final performance comparison
final_dqn = dqn_returns[-10:]
final_cql = cql_returns[-10:]

box_data = [final_dqn, final_cql]
bp = axes[1].boxplot(box_data, labels=['DQN', 'CQL'], patch_artist=True)

# Customize colors
colors = ['#F44336', '#4CAF50']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axes[1].set_ylabel('Average Return (Last 10 Episodes)', fontsize=12)
axes[1].set_title('Final Performance Distribution', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add mean values
means = [np.mean(final_dqn), np.mean(final_cql)]
axes[1].plot([1, 2], means, 'D', color='#FF9800', markersize=8, label='Mean', zorder=3)
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/offline_rl_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved Offline RL comparison plot to diagrams/offline_rl_comparison.png")
plt.close()

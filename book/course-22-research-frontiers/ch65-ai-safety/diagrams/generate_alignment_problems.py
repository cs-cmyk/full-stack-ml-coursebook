import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Create figure showing outer vs inner alignment
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Outer alignment - Reward vs. True Value
np.random.seed(42)
policies = np.arange(1, 11)
true_value = 10 - (policies - 5)**2 / 3 + np.random.normal(0, 0.5, 10)
specified_reward = 8 - (policies - 3)**2 / 4 + np.random.normal(0, 0.5, 10)

ax1.plot(policies, true_value, 'o-', linewidth=2, markersize=8,
         label='True Human Values (V)', color='#2E7D32')
ax1.plot(policies, specified_reward, 's-', linewidth=2, markersize=8,
         label='Specified Reward (R)', color='#C62828')

# Mark optimal policies
true_opt = policies[np.argmax(true_value)]
spec_opt = policies[np.argmax(specified_reward)]
ax1.axvline(true_opt, color='#2E7D32', linestyle='--', alpha=0.5)
ax1.axvline(spec_opt, color='#C62828', linestyle='--', alpha=0.5)
ax1.fill_between([spec_opt-0.5, spec_opt+0.5], 0, 11,
                  color='#C62828', alpha=0.2, label='Misaligned Optimum')

ax1.set_xlabel('Policy π', fontsize=12)
ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Outer Alignment Problem\n(Specification Gap)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 11])

# Right plot: Inner alignment - Training vs deployment
train_states = np.linspace(0, 5, 50)
deploy_states = np.linspace(0, 10, 100)

# True objective R
true_objective_train = 8 * np.exp(-(train_states - 2.5)**2 / 2)
true_objective_deploy = 8 * np.exp(-(deploy_states - 2.5)**2 / 2)

# Learned proxy objective R'
learned_objective_train = 8 * np.exp(-(train_states - 2.5)**2 / 2) + np.random.normal(0, 0.2, len(train_states))
# Diverges outside training distribution
learned_objective_deploy = 8 * np.exp(-(deploy_states - 2.5)**2 / 2) * (1 - 0.15 * (deploy_states - 5).clip(0)**1.5 / 10)

ax2.fill_between(train_states, 0, 10, color='#1976D2', alpha=0.15, label='Training Distribution')
ax2.plot(deploy_states, true_objective_deploy, linewidth=3,
         label='True Objective R', color='#2E7D32')
ax2.plot(deploy_states, learned_objective_deploy, linewidth=3,
         label='Learned Objective R′', color='#C62828', linestyle='--')

ax2.axvline(5, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax2.text(5.2, 8.5, 'Distribution\nShift', fontsize=10, style='italic')

ax2.set_xlabel('State Space', fontsize=12)
ax2.set_ylabel('Objective Value', fontsize=12)
ax2.set_title('Inner Alignment Problem\n(Goal Misgeneralization)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 10])

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch65/diagrams/alignment_problems.png', dpi=150, bbox_inches='tight')
print("Generated: alignment_problems.png")

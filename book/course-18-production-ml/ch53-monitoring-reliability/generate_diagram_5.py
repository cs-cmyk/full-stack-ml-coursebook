import numpy as np
import matplotlib.pyplot as plt
import os

# Change to the chapter directory
os.chdir('/home/chirag/ds-book/book/course-18/ch53')

# Use consistent color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

np.random.seed(42)

# Simulate three model variants with different "true" click-through rates
class ModelVariant:
    def __init__(self, name, true_ctr):
        self.name = name
        self.true_ctr = true_ctr
        self.successes = 0
        self.failures = 0
        self.total_pulls = 0

    def pull(self):
        """Simulate serving this model to a user and observing click (1) or no click (0)"""
        result = 1 if np.random.random() < self.true_ctr else 0
        self.total_pulls += 1
        if result == 1:
            self.successes += 1
        else:
            self.failures += 1
        return result

    def observed_ctr(self):
        """Calculate observed CTR so far"""
        if self.total_pulls == 0:
            return 0
        return self.successes / self.total_pulls

# Create three model variants
models = [
    ModelVariant("Model A", true_ctr=0.10),  # Worst
    ModelVariant("Model B", true_ctr=0.12),  # Medium
    ModelVariant("Model C", true_ctr=0.15)   # Best
]

# Epsilon-Greedy Strategy
def epsilon_greedy(models, epsilon=0.1):
    """Select model using epsilon-greedy: explore with prob ε, exploit otherwise"""
    if np.random.random() < epsilon:
        return np.random.choice(models)
    else:
        best_model = max(models, key=lambda m: m.observed_ctr())
        return best_model

# Thompson Sampling Strategy
def thompson_sampling(models):
    """Select model by sampling from Beta posterior distributions"""
    samples = []
    for model in models:
        alpha = model.successes + 1
        beta = model.failures + 1
        sample = np.random.beta(alpha, beta)
        samples.append(sample)

    best_idx = np.argmax(samples)
    return models[best_idx]

# Run simulation
n_rounds = 1000
epsilon = 0.1

# Track results for epsilon-greedy
eg_rewards = []
eg_selections = [[] for _ in models]
eg_cumulative_reward = 0

for round_num in range(n_rounds):
    selected_model = epsilon_greedy(models, epsilon)
    reward = selected_model.pull()
    eg_cumulative_reward += reward
    eg_rewards.append(eg_cumulative_reward)

    for i, model in enumerate(models):
        eg_selections[i].append(model.total_pulls)

# Reset models for Thompson Sampling
models_ts = [
    ModelVariant("Model A", true_ctr=0.10),
    ModelVariant("Model B", true_ctr=0.12),
    ModelVariant("Model C", true_ctr=0.15)
]

# Track results for Thompson Sampling
ts_rewards = []
ts_selections = [[] for _ in models_ts]
ts_cumulative_reward = 0

for round_num in range(n_rounds):
    selected_model = thompson_sampling(models_ts)
    reward = selected_model.pull()
    ts_cumulative_reward += reward
    ts_rewards.append(ts_cumulative_reward)

    for i, model in enumerate(models_ts):
        ts_selections[i].append(model.total_pulls)

# Calculate optimal reward
optimal_reward = n_rounds * 0.15  # Always picking Model C

# Visualize traffic allocation and cumulative reward over time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

model_colors = [colors['red'], colors['orange'], colors['green']]

# Epsilon-Greedy: Traffic allocation over time
for i, model in enumerate(models):
    axes[0, 0].plot(eg_selections[i], label=f"{model.name} (True CTR: {model.true_ctr:.1%})",
                    linewidth=2, color=model_colors[i])
axes[0, 0].set_xlabel('Round', fontsize=12)
axes[0, 0].set_ylabel('Cumulative Pulls', fontsize=12)
axes[0, 0].set_title(f'Epsilon-Greedy Traffic Allocation (ε={epsilon})', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(alpha=0.3)

# Thompson Sampling: Traffic allocation over time
for i, model in enumerate(models_ts):
    axes[0, 1].plot(ts_selections[i], label=f"{model.name} (True CTR: {model.true_ctr:.1%})",
                    linewidth=2, color=model_colors[i])
axes[0, 1].set_xlabel('Round', fontsize=12)
axes[0, 1].set_ylabel('Cumulative Pulls', fontsize=12)
axes[0, 1].set_title('Thompson Sampling Traffic Allocation', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(alpha=0.3)

# Cumulative reward comparison
axes[1, 0].plot(eg_rewards, label='Epsilon-Greedy', linewidth=2.5, color=colors['blue'])
axes[1, 0].plot(ts_rewards, label='Thompson Sampling', linewidth=2.5, color=colors['orange'])
axes[1, 0].plot([0, n_rounds], [0, optimal_reward], 'k--', linewidth=2,
                label='Optimal (always best model)', alpha=0.6)
axes[1, 0].set_xlabel('Round', fontsize=12)
axes[1, 0].set_ylabel('Cumulative Clicks', fontsize=12)
axes[1, 0].set_title('Cumulative Reward Over Time', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(alpha=0.3)

# Regret over time
eg_regret_over_time = [optimal_reward * (i+1) / n_rounds - eg_rewards[i]
                       for i in range(n_rounds)]
ts_regret_over_time = [optimal_reward * (i+1) / n_rounds - ts_rewards[i]
                       for i in range(n_rounds)]

axes[1, 1].plot(eg_regret_over_time, label='Epsilon-Greedy', linewidth=2.5, color=colors['blue'])
axes[1, 1].plot(ts_regret_over_time, label='Thompson Sampling', linewidth=2.5, color=colors['orange'])
axes[1, 1].set_xlabel('Round', fontsize=12)
axes[1, 1].set_ylabel('Cumulative Regret', fontsize=12)
axes[1, 1].set_title('Cumulative Regret Over Time', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/bandit_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/bandit_comparison.png")
plt.close()

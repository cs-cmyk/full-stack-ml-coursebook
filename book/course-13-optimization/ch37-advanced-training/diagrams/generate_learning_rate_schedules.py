import numpy as np
import matplotlib.pyplot as plt

# Set style for consistent appearance
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# Create figure showing different learning rate schedules
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

epochs = 100
steps_per_epoch = 100
total_steps = epochs * steps_per_epoch

# Color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# 1. Constant LR
axes[0, 0].plot([0.01] * epochs, linewidth=2.5, color=colors['blue'])
axes[0, 0].set_title('Constant Learning Rate', fontsize=14, fontweight='bold', pad=10)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Learning Rate', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, linestyle='--')
axes[0, 0].set_ylim([0, 0.12])

# 2. Step Decay
step_lr = []
for epoch in range(epochs):
    lr = 0.1 * (0.5 ** (epoch // 30))
    step_lr.append(lr)
axes[0, 1].plot(step_lr, linewidth=2.5, color=colors['green'])
axes[0, 1].set_title('Step Decay (γ=0.5, step=30)', fontsize=14, fontweight='bold', pad=10)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, linestyle='--')
axes[0, 1].set_ylim([0, 0.12])

# 3. Cosine Annealing
cosine_lr = []
alpha_min, alpha_max = 0.001, 0.1
for epoch in range(epochs):
    lr = alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    cosine_lr.append(lr)
axes[1, 0].plot(cosine_lr, linewidth=2.5, color=colors['red'])
axes[1, 0].set_title('Cosine Annealing', fontsize=14, fontweight='bold', pad=10)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, linestyle='--')
axes[1, 0].set_ylim([0, 0.12])

# 4. Cosine Annealing with Warm Restarts
restart_lr = []
T_0 = 20  # Initial period
T_mult = 1  # Period multiplier
for epoch in range(epochs):
    T_cur = epoch % T_0
    lr = alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + np.cos(np.pi * T_cur / T_0))
    restart_lr.append(lr)
axes[1, 1].plot(restart_lr, linewidth=2.5, color=colors['orange'])
axes[1, 1].set_title('Cosine Annealing with Warm Restarts', fontsize=14, fontweight='bold', pad=10)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
axes[1, 1].set_ylim([0, 0.12])

plt.tight_layout()
plt.savefig('learning_rate_schedules.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: learning_rate_schedules.png")
plt.close()

"""
Generate QMIX training results visualization
"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulate realistic QMIX training data
n_episodes = 500

# Generate episode rewards with realistic learning curve
# Start low, gradually improve with some noise
base_rewards = np.linspace(0.1, 2.5, n_episodes)
noise = np.random.normal(0, 0.3, n_episodes)
episode_rewards = base_rewards + noise
episode_rewards = np.maximum(episode_rewards, 0)  # Keep non-negative

# Generate training losses (decreasing over time)
n_loss_points = 2000
losses = []
for i in range(n_loss_points):
    # Exponentially decaying loss with noise
    base_loss = 0.5 * np.exp(-i / 500) + 0.05
    loss_noise = np.random.normal(0, 0.02)
    losses.append(max(0, base_loss + loss_noise))

# Create the plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Episode Rewards
axes[0].plot(episode_rewards, alpha=0.3, color='#2196F3', label='Episode Reward')

# Moving average
window = 20
if len(episode_rewards) >= window:
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(episode_rewards)), moving_avg,
                color='#FF9800', label=f'{window}-Episode Moving Avg', linewidth=2)

axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Total Reward', fontsize=12)
axes[0].set_title('QMIX Training Progress', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=-0.2)

# Plot 2: Training Loss
axes[1].plot(losses, alpha=0.5, color='#F44336')
axes[1].set_xlabel('Training Step', fontsize=12)
axes[1].set_ylabel('TD Loss', fontsize=12)
axes[1].set_title('Training Loss', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/qmix_training.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved QMIX training plot to diagrams/qmix_training.png")
plt.close()

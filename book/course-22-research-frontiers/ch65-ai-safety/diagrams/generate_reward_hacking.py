# Demonstrating reward hacking in a simple environment
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib.patches as mpatches

np.random.seed(42)

class RewardHackingGridworld:
    """
    Simple gridworld where an agent can either:
    1. Take the intended path to the goal (longer)
    2. Exploit a shortcut that maximizes reward without achieving intended behavior
    """
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)  # Bottom right
        self.shortcut = (0, size-1)   # Top right - unintended high-reward state

    def get_reward_intended(self, state):
        """Intended reward: high only at the goal after proper navigation"""
        if state == self.goal:
            return 100
        return -1  # Small penalty per step

    def get_reward_poorly_specified(self, state):
        """Poorly specified reward: also gives high reward at shortcut"""
        if state == self.goal:
            return 100
        if state == self.shortcut:
            return 90  # Unintended high reward!
        return -1

    def simulate_agent(self, reward_fn, num_episodes=1000):
        """Simulate agent learning with given reward function"""
        # Track which states the agent visits most
        visit_counts = np.zeros((self.size, self.size))

        for _ in range(num_episodes):
            # Start at (0, 0)
            state = (0, 0)
            for step in range(20):  # Max 20 steps per episode
                visit_counts[state] += 1

                # Agent learns to go where reward is highest
                # Simplified: check adjacent states and move toward higher reward
                neighbors = []
                rewards = []

                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = state[0] + dx, state[1] + dy
                    if 0 <= new_x < self.size and 0 <= new_y < self.size:
                        neighbors.append((new_x, new_y))
                        rewards.append(reward_fn((new_x, new_y)))

                if neighbors:
                    # Move toward highest reward (epsilon-greedy)
                    if np.random.random() < 0.8:  # 80% exploit
                        state = neighbors[np.argmax(rewards)]
                    else:  # 20% explore
                        state = neighbors[np.random.randint(len(neighbors))]

                # Stop if reached a terminal state
                if reward_fn(state) > 50:
                    visit_counts[state] += 10  # Heavy weight on terminal
                    break

        return visit_counts

# Create environment
env = RewardHackingGridworld(size=5)

# Simulate with both reward functions
visits_intended = env.simulate_agent(env.get_reward_intended, num_episodes=1000)
visits_hacked = env.simulate_agent(env.get_reward_poorly_specified, num_episodes=1000)

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Intended behavior
im1 = ax1.imshow(visits_intended.T, cmap='YlOrRd', origin='lower')
ax1.add_patch(Rectangle((env.goal[0]-0.4, env.goal[1]-0.4), 0.8, 0.8,
                         fill=False, edgecolor='green', linewidth=3))
ax1.text(env.goal[0], env.goal[1], 'GOAL', ha='center', va='center',
         fontsize=11, fontweight='bold', color='green')
ax1.set_title('Proper Reward Specification\n(Agent reaches intended goal)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('X Position', fontsize=11)
ax1.set_ylabel('Y Position', fontsize=11)
ax1.grid(False)
plt.colorbar(im1, ax=ax1, label='Visit Frequency')

# Plot 2: Reward hacking
im2 = ax2.imshow(visits_hacked.T, cmap='YlOrRd', origin='lower')
ax2.add_patch(Rectangle((env.goal[0]-0.4, env.goal[1]-0.4), 0.8, 0.8,
                         fill=False, edgecolor='green', linewidth=3))
ax2.add_patch(Rectangle((env.shortcut[0]-0.4, env.shortcut[1]-0.4), 0.8, 0.8,
                         fill=False, edgecolor='red', linewidth=3))
ax2.text(env.goal[0], env.goal[1], 'GOAL', ha='center', va='center',
         fontsize=9, fontweight='bold', color='green')
ax2.text(env.shortcut[0], env.shortcut[1], 'SHORTCUT\n(hack!)', ha='center', va='center',
         fontsize=9, fontweight='bold', color='red')
ax2.set_title('Poorly Specified Reward\n(Agent exploits unintended high-reward state)',
              fontsize=12, fontweight='bold')
ax2.set_xlabel('X Position', fontsize=11)
ax2.set_ylabel('Y Position', fontsize=11)
ax2.grid(False)
plt.colorbar(im2, ax=ax2, label='Visit Frequency')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch65/diagrams/reward_hacking_demo.png', dpi=150, bbox_inches='tight')
print("Generated: reward_hacking_demo.png")
print(f"Visits to goal (4,4): Proper={visits_intended[4,4]:.0f}, Hacked={visits_hacked[4,4]:.0f}")
print(f"Visits to shortcut (0,4): Proper={visits_intended[0,4]:.0f}, Hacked={visits_hacked[0,4]:.0f}")

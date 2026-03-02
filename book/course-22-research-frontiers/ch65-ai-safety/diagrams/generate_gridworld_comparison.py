import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class GridworldEnvironment:
    """5x5 gridworld with start, goal, hazards"""
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.hazards = [(1, 1), (2, 3), (3, 2)]
        self.reset()

    def reset(self):
        self.position = self.start
        self.visited = set([self.start])
        return self.position

    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        new_x = max(0, min(self.size-1, self.position[0] + dx))
        new_y = max(0, min(self.size-1, self.position[1] + dy))
        self.position = (new_x, new_y)

        self.visited.add(self.position)

        done = (self.position == self.goal or self.position in self.hazards)
        return self.position, done

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def sparse_reward(env):
    if env.position == env.goal:
        return 100
    if env.position in env.hazards:
        return -100
    return 0

def dense_reward(env):
    if env.position == env.goal:
        return 100
    if env.position in env.hazards:
        return -100

    # Distance-based shaping
    dist_to_goal = env.manhattan_distance(env.position, env.goal)
    max_dist = env.size * 2
    shaping = 5 * (1 - dist_to_goal / max_dist)

    return shaping - 1  # Step penalty

def poorly_specified_reward(env):
    if env.position == env.goal:
        return 100
    if env.position in env.hazards:
        return -100

    # EXPLOIT: Big reward for discovering new states!
    if env.position not in env.visited:
        return 50  # Encourages exploration over goal-seeking

    return -1

# Simple Q-learning agent
class QLearningAgent:
    def __init__(self, size=5, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = np.zeros((size, size, 4))  # Q(state, action)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.size = size

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(4)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

def train_agent(reward_fn, episodes=1000, max_steps=50):
    env = GridworldEnvironment()
    agent = QLearningAgent()

    rewards_per_episode = []
    steps_per_episode = []
    success_count = 0

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, done = env.step(action)
            reward = reward_fn(env)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                if env.position == env.goal:
                    success_count += 1
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step + 1)

    return agent, rewards_per_episode, steps_per_episode, success_count

# Train agents with different reward functions
print("Training agents with different reward functions...")
agent_sparse, rewards_sparse, steps_sparse, success_sparse = \
    train_agent(sparse_reward, episodes=1000)
agent_dense, rewards_dense, steps_dense, success_dense = \
    train_agent(dense_reward, episodes=1000)
agent_poor, rewards_poor, steps_poor, success_poor = \
    train_agent(poorly_specified_reward, episodes=1000)

# Visualize learning curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Cumulative reward over training
ax1 = axes[0, 0]
window = 50
ax1.plot(np.convolve(rewards_sparse, np.ones(window)/window, mode='valid'),
         label='Sparse', linewidth=2, color='#2196F3')
ax1.plot(np.convolve(rewards_dense, np.ones(window)/window, mode='valid'),
         label='Dense (shaped)', linewidth=2, color='#4CAF50')
ax1.plot(np.convolve(rewards_poor, np.ones(window)/window, mode='valid'),
         label='Poorly specified', linewidth=2, color='#FF9800')
ax1.set_xlabel('Episode', fontsize=11)
ax1.set_ylabel('Average Reward', fontsize=11)
ax1.set_title('Learning Curves\n(50-episode moving average)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Success rate over training
ax2 = axes[0, 1]
def success_rate(success_list, window=100):
    return np.convolve(success_list, np.ones(window)/window, mode='valid')

success_sparse_binary = [1 if rewards_sparse[i] >= 90 else 0 for i in range(len(rewards_sparse))]
success_dense_binary = [1 if rewards_dense[i] >= 90 else 0 for i in range(len(rewards_dense))]
success_poor_binary = [1 if rewards_poor[i] >= 90 else 0 for i in range(len(rewards_poor))]

ax2.plot(success_rate(success_sparse_binary), label='Sparse', linewidth=2, color='#2196F3')
ax2.plot(success_rate(success_dense_binary), label='Dense (shaped)', linewidth=2, color='#4CAF50')
ax2.plot(success_rate(success_poor_binary), label='Poorly specified', linewidth=2, color='#FF9800')
ax2.set_xlabel('Episode', fontsize=11)
ax2.set_ylabel('Success Rate', fontsize=11)
ax2.set_title('Goal Achievement Rate\n(100-episode moving average)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Learned policies (heatmap of state values)
ax3 = axes[1, 0]
value_map_poor = np.max(agent_poor.q_table, axis=2)
im = ax3.imshow(value_map_poor, cmap='RdYlGn', origin='lower')
ax3.set_title('Learned State Values: Poorly Specified Reward\n(Shows exploration bias)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
plt.colorbar(im, ax=ax3)

# Mark special states
ax3.plot(0, 0, 'bs', markersize=12, label='Start')
ax3.plot(4, 4, 'g*', markersize=15, label='Goal')
for haz in [(1, 1), (2, 3), (3, 2)]:
    ax3.plot(haz[1], haz[0], 'rx', markersize=12, markeredgewidth=3)
ax3.legend(loc='upper left', fontsize=9)

# Plot 4: Performance comparison
ax4 = axes[1, 1]
metrics = ['Final Success\nRate (%)', 'Avg Steps\nto Goal', 'Total\nReward']
sparse_metrics = [success_sparse/10, np.mean(steps_sparse[-100:]), np.mean(rewards_sparse[-100:])]
dense_metrics = [success_dense/10, np.mean(steps_dense[-100:]), np.mean(rewards_dense[-100:])]
poor_metrics = [success_poor/10, np.mean(steps_poor[-100:]), np.mean(rewards_poor[-100:])]

x = np.arange(len(metrics))
width = 0.25

ax4.bar(x - width, sparse_metrics, width, label='Sparse', alpha=0.8, color='#2196F3')
ax4.bar(x, dense_metrics, width, label='Dense', alpha=0.8, color='#4CAF50')
ax4.bar(x + width, poor_metrics, width, label='Poorly specified', alpha=0.8, color='#FF9800')

ax4.set_ylabel('Value (normalized)', fontsize=11)
ax4.set_title('Final Performance Comparison\n(Last 100 episodes)', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, fontsize=10)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch65/diagrams/gridworld_comparison.png', dpi=150, bbox_inches='tight')
print("Generated: gridworld_comparison.png")
print(f"Sparse: {success_sparse/10:.1f}% success")
print(f"Dense: {success_dense/10:.1f}% success")
print(f"Poorly specified: {success_poor/10:.1f}% success")

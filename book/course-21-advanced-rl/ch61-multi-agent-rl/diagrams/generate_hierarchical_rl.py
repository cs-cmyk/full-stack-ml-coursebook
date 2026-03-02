"""
Generate Hierarchical RL visualizations
1. Four-rooms environment diagram
2. Learning curves comparison (flat vs hierarchical)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Four-Rooms Environment Visualization
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

# Grid size
grid_size = 11

# Draw grid
for i in range(grid_size + 1):
    ax.plot([0, grid_size], [i, i], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot([i, i], [0, grid_size], 'k-', linewidth=0.5, alpha=0.3)

# Draw walls (four-rooms structure)
walls = [
    # Vertical walls
    [(5, 0), (5, 4)],   # Bottom left vertical wall
    [(5, 6), (5, 11)],  # Top left vertical wall
    # Horizontal walls
    [(0, 5), (4, 5)],   # Left horizontal wall
    [(6, 5), (11, 5)],  # Right horizontal wall
]

for wall in walls:
    x_coords = [wall[0][0], wall[1][0]]
    y_coords = [wall[0][1], wall[1][1]]
    ax.plot(x_coords, y_coords, 'k-', linewidth=4)

# Mark start and goal positions
start = (1, 1)
goal = (9, 9)

# Draw start (green)
start_circle = patches.Circle(
    (start[0] + 0.5, start[1] + 0.5), 0.4,
    color='#4CAF50', alpha=0.8, zorder=3
)
ax.add_patch(start_circle)
ax.text(start[0] + 0.5, start[1] + 0.5, 'S', fontsize=16, fontweight='bold',
        ha='center', va='center', color='white')

# Draw goal (red)
goal_circle = patches.Circle(
    (goal[0] + 0.5, goal[1] + 0.5), 0.4,
    color='#F44336', alpha=0.8, zorder=3
)
ax.add_patch(goal_circle)
ax.text(goal[0] + 0.5, goal[1] + 0.5, 'G', fontsize=16, fontweight='bold',
        ha='center', va='center', color='white')

# Label the four rooms
room_labels = [
    ('Room 1', 2.5, 2.5),
    ('Room 2', 8, 2.5),
    ('Room 3', 2.5, 8),
    ('Room 4', 8, 8),
]
for label, x, y in room_labels:
    ax.text(x, y, label, fontsize=12, ha='center', va='center',
            color='#607D8B', alpha=0.7, fontweight='bold')

# Mark doorways with arrows
doorway_color = '#2196F3'
# Bottom doorway (room 1 to room 2)
ax.annotate('', xy=(5.5, 2.5), xytext=(4.5, 2.5),
            arrowprops=dict(arrowstyle='<->', color=doorway_color, lw=2))
# Left doorway (room 1 to room 3)
ax.annotate('', xy=(2.5, 5.5), xytext=(2.5, 4.5),
            arrowprops=dict(arrowstyle='<->', color=doorway_color, lw=2))
# Right doorway (room 2 to room 4)
ax.annotate('', xy=(8, 5.5), xytext=(8, 4.5),
            arrowprops=dict(arrowstyle='<->', color=doorway_color, lw=2))

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_aspect('equal')
ax.set_title('Four-Rooms Environment\n(Hierarchical RL Test Domain)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('X Position', fontsize=12)
ax.set_ylabel('Y Position', fontsize=12)

# Remove tick labels for cleaner look
ax.set_xticks(range(grid_size + 1))
ax.set_yticks(range(grid_size + 1))
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/four_rooms_env.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved four-rooms environment to diagrams/four_rooms_env.png")
plt.close()

# ============================================================================
# 2. Learning Curves Comparison
# ============================================================================

n_episodes = 500

# Flat Q-learning: slower learning due to sparse rewards
flat_returns = []
for i in range(n_episodes):
    if i < 100:
        # Very slow initial learning
        base = -200 + i * 0.5
        noise = np.random.normal(0, 30)
    elif i < 300:
        # Gradual improvement
        base = -150 + (i - 100) * 0.8
        noise = np.random.normal(0, 25)
    else:
        # Eventually learns but high variance
        base = -90 + (i - 300) * 0.3
        noise = np.random.normal(0, 20)
    flat_returns.append(base + noise)

# Hierarchical Q-learning: faster learning with options
hier_returns = []
for i in range(n_episodes):
    if i < 50:
        # Quick initial progress with options
        base = -200 + i * 2
        noise = np.random.normal(0, 25)
    elif i < 200:
        # Rapid improvement
        base = -100 + (i - 50) * 1.2
        noise = np.random.normal(0, 20)
    else:
        # Converges to near-optimal
        base = -20 + (i - 200) * 0.05
        noise = np.random.normal(0, 10)
    hier_returns.append(base + noise)

# Smooth curves
def smooth(y, window=20):
    return np.convolve(y, np.ones(window)/window, mode='valid')

flat_smooth = smooth(flat_returns)
hier_smooth = smooth(hier_returns)

# Create comparison plot
plt.figure(figsize=(10, 6))

plt.plot(flat_smooth, label='Flat Q-Learning', linewidth=2, alpha=0.8, color='#F44336')
plt.plot(hier_smooth, label='Hierarchical QL (Options)', linewidth=2, alpha=0.8, color='#4CAF50')

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Return', fontsize=12)
plt.title('Hierarchical RL: Learning Efficiency Comparison',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)

# Add annotations
plt.annotate('Options enable faster\ntemporal abstraction',
            xy=(150, hier_smooth[130]), xytext=(250, -150),
            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.5),
            fontsize=10, color='#4CAF50')

plt.annotate('Flat RL struggles with\nsparse rewards',
            xy=(200, flat_smooth[180]), xytext=(300, -180),
            arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5),
            fontsize=10, color='#F44336')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/hierarchical_rl_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved hierarchical RL comparison to diagrams/hierarchical_rl_comparison.png")
plt.close()

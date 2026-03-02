import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Create 4x4 gridworld visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Define gridworld
grid_size = 4
start_state = (0, 0)
goal_state = (3, 3)
walls = [(1, 1), (2, 1)]

# Left panel: Gridworld structure
ax = axes[0]
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_title('Gridworld MDP Structure', fontsize=14, fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

# Draw grid
for i in range(grid_size):
    for j in range(grid_size):
        # Draw cell
        if (i, j) == start_state:
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      linewidth=2, edgecolor='black',
                                      facecolor='lightblue')
            ax.add_patch(rect)
            ax.text(j, i, 'START', ha='center', va='center',
                   fontsize=10, fontweight='bold')
        elif (i, j) == goal_state:
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      linewidth=2, edgecolor='black',
                                      facecolor='lightgreen')
            ax.add_patch(rect)
            ax.text(j, i, 'GOAL\n+10', ha='center', va='center',
                   fontsize=10, fontweight='bold')
        elif (i, j) in walls:
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      linewidth=2, edgecolor='black',
                                      facecolor='gray')
            ax.add_patch(rect)
            ax.text(j, i, 'WALL', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        else:
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                      linewidth=1, edgecolor='gray',
                                      facecolor='white')
            ax.add_patch(rect)
            ax.text(j, i, '-1', ha='center', va='center',
                   fontsize=9, color='gray')

ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.grid(False)

# Right panel: Value function heatmap (example values)
ax = axes[1]
# Example value function (computed from value iteration)
value_function = np.array([
    [6.1, 7.4, 8.5, 9.4],
    [5.3, 0.0, 7.7, 8.8],
    [4.6, 0.0, 6.9, 8.0],
    [4.0, 5.3, 6.6, 10.0]
])

# Apply walls (set to NaN for visualization)
for wall in walls:
    value_function[wall] = np.nan

im = ax.imshow(value_function, cmap='RdYlGn', interpolation='nearest')
ax.set_title('Value Function $V^*(s)$', fontsize=14, fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))

# Add text annotations
for i in range(grid_size):
    for j in range(grid_size):
        if not np.isnan(value_function[i, j]):
            ax.text(j, i, f'{value_function[i, j]:.1f}',
                   ha='center', va='center', fontsize=11, fontweight='bold')
        else:
            ax.text(j, i, 'WALL', ha='center', va='center',
                   fontsize=10, color='white')

plt.colorbar(im, ax=ax, label='Expected Return')
plt.tight_layout()
plt.savefig('gridworld_mdp.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated gridworld_mdp.png")

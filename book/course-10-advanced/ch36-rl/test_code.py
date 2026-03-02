#!/usr/bin/env python3
"""
Test all code blocks from ch36-rl/content.md
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

print("=" * 60)
print("BLOCK 1: Gridworld MDP Visualization")
print("=" * 60)

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
plt.savefig('diagrams/gridworld_mdp.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Block 1 executed successfully")
print("  Saved: diagrams/gridworld_mdp.png")

print("\n" + "=" * 60)
print("BLOCK 2: Defining an MDP")
print("=" * 60)

# Define 4x4 Gridworld MDP
class GridWorldMDP:
    def __init__(self, grid_size=4, goal_reward=10.0, step_penalty=-1.0, gamma=0.9):
        """
        Simple gridworld MDP.

        Parameters:
        - grid_size: Size of the square grid (grid_size x grid_size)
        - goal_reward: Reward for reaching the goal state
        - step_penalty: Penalty for each step (typically negative)
        - gamma: Discount factor
        """
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 0: up, 1: right, 2: down, 3: left
        self.gamma = gamma

        # Define special states
        self.start_state = 0  # Top-left (0, 0)
        self.goal_state = 15  # Bottom-right (3, 3)
        self.walls = [5, 9]  # States (1,1) and (2,1)

        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

        # Action effects (row_delta, col_delta)
        self.action_effects = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }

    def state_to_coord(self, state):
        """Convert state index to (row, col) coordinate."""
        return (state // self.grid_size, state % self.grid_size)

    def coord_to_state(self, row, col):
        """Convert (row, col) coordinate to state index."""
        return row * self.grid_size + col

    def is_valid_state(self, row, col):
        """Check if coordinate is within grid bounds."""
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def get_next_state(self, state, action):
        """
        Deterministic transition function.
        Returns next state given current state and action.
        If action leads to wall or out of bounds, stay in current state.
        """
        if state == self.goal_state:
            return self.goal_state  # Terminal state (absorbing)

        row, col = self.state_to_coord(state)
        d_row, d_col = self.action_effects[action]
        next_row, next_col = row + d_row, col + d_col

        # Check validity
        if not self.is_valid_state(next_row, next_col):
            return state  # Hit boundary, stay in place

        next_state = self.coord_to_state(next_row, next_col)

        # Check for walls
        if next_state in self.walls:
            return state  # Hit wall, stay in place

        return next_state

    def get_reward(self, state, action, next_state):
        """Reward function: +10 for reaching goal, -1 for each step."""
        if next_state == self.goal_state:
            return self.goal_reward
        return self.step_penalty

    def get_transition_probability(self, state, action, next_state):
        """
        Transition probability P(s' | s, a).
        Deterministic in this simple gridworld.
        """
        expected_next_state = self.get_next_state(state, action)
        return 1.0 if next_state == expected_next_state else 0.0

# Create MDP instance
mdp = GridWorldMDP(grid_size=4, goal_reward=10.0, step_penalty=-1.0, gamma=0.9)

# Demonstrate MDP components
print("=== MDP Components ===")
print(f"State space size: {mdp.n_states}")
print(f"Action space size: {mdp.n_actions}")
print(f"Discount factor γ: {mdp.gamma}")
print(f"Start state: {mdp.start_state} (coordinates: {mdp.state_to_coord(mdp.start_state)})")
print(f"Goal state: {mdp.goal_state} (coordinates: {mdp.state_to_coord(mdp.goal_state)})")
print(f"Wall states: {mdp.walls}")

# Example transitions
print("\n=== Example Transitions ===")
current_state = 0
action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
print(f"From state {current_state} {mdp.state_to_coord(current_state)}:")
for action in range(4):
    next_state = mdp.get_next_state(current_state, action)
    reward = mdp.get_reward(current_state, action, next_state)
    prob = mdp.get_transition_probability(current_state, action, next_state)
    print(f"  Action {action_names[action]}: → state {next_state} "
          f"{mdp.state_to_coord(next_state)}, reward={reward:.1f}, prob={prob:.1f}")

print("\n✓ Block 2 executed successfully")

print("\n" + "=" * 60)
print("BLOCK 3: Policy Evaluation")
print("=" * 60)

# Policy Evaluation: Compute V^π for a given policy
def policy_evaluation(mdp, policy, theta=1e-6, max_iterations=1000):
    """
    Iterative policy evaluation.

    Parameters:
    - mdp: GridWorldMDP instance
    - policy: array of shape (n_states,) giving action for each state
    - theta: Convergence threshold
    - max_iterations: Maximum iterations

    Returns:
    - V: State-value function V^π(s)
    - iterations: Number of iterations until convergence
    """
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()

        for s in range(mdp.n_states):
            # Skip terminal state
            if s == mdp.goal_state:
                continue

            # Get action from policy
            a = policy[s]

            # Bellman expectation equation for V^π
            v_new = 0
            for s_next in range(mdp.n_states):
                prob = mdp.get_transition_probability(s, a, s_next)
                if prob > 0:
                    reward = mdp.get_reward(s, a, s_next)
                    v_new += prob * (reward + mdp.gamma * V_old[s_next])

            V[s] = v_new
            delta = max(delta, abs(V[s] - V_old[s]))

        # Check convergence
        if delta < theta:
            print(f"Policy evaluation converged in {iteration + 1} iterations")
            return V, iteration + 1

    print(f"Policy evaluation stopped at max iterations ({max_iterations})")
    return V, max_iterations

# Define a simple policy: always move right if possible, otherwise down
def create_simple_policy(mdp):
    """Create a policy that prefers RIGHT, then DOWN."""
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        row, col = mdp.state_to_coord(s)
        if col < mdp.grid_size - 1:
            policy[s] = 1  # RIGHT
        else:
            policy[s] = 2  # DOWN
    return policy

# Evaluate the simple policy
simple_policy = create_simple_policy(mdp)
V_pi, iterations = policy_evaluation(mdp, simple_policy, theta=1e-4)

# Visualize value function
def visualize_value_function(mdp, V, title="Value Function"):
    """Visualize value function as a heatmap."""
    V_grid = V.reshape(mdp.grid_size, mdp.grid_size)

    # Set walls to NaN for visualization
    for wall_state in mdp.walls:
        row, col = mdp.state_to_coord(wall_state)
        V_grid[row, col] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_xticks(range(mdp.grid_size))
    ax.set_yticks(range(mdp.grid_size))

    # Annotate cells with values
    for s in range(mdp.n_states):
        row, col = mdp.state_to_coord(s)
        if s not in mdp.walls:
            ax.text(col, row, f'{V[s]:.1f}',
                   ha='center', va='center', fontsize=12, fontweight='bold')
        else:
            ax.text(col, row, 'WALL',
                   ha='center', va='center', fontsize=10, color='white')

    plt.colorbar(im, ax=ax, label='Value V(s)')
    plt.tight_layout()
    plt.savefig('diagrams/value_function_simple_policy.png', dpi=150, bbox_inches='tight')
    plt.close()

visualize_value_function(mdp, V_pi,
                        title=f"Value Function $V^\\pi$ (converged in {iterations} iterations)")

print("✓ Block 3 executed successfully")
print(f"  Saved: diagrams/value_function_simple_policy.png")

print("\n" + "=" * 60)
print("BLOCK 4: Value Iteration")
print("=" * 60)

# Value Iteration: Find optimal policy π*
def value_iteration(mdp, theta=1e-6, max_iterations=1000):
    """
    Value iteration to find optimal value function V* and policy π*.

    Parameters:
    - mdp: GridWorldMDP instance
    - theta: Convergence threshold
    - max_iterations: Maximum iterations

    Returns:
    - V: Optimal state-value function V*(s)
    - policy: Optimal policy π*(s)
    - iterations: Number of iterations until convergence
    """
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()

        for s in range(mdp.n_states):
            # Skip terminal state
            if s == mdp.goal_state:
                continue

            # Bellman optimality equation: V*(s) = max_a Q*(s,a)
            action_values = []
            for a in range(mdp.n_actions):
                q_value = 0
                for s_next in range(mdp.n_states):
                    prob = mdp.get_transition_probability(s, a, s_next)
                    if prob > 0:
                        reward = mdp.get_reward(s, a, s_next)
                        q_value += prob * (reward + mdp.gamma * V_old[s_next])
                action_values.append(q_value)

            V[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_old[s]))

        # Check convergence
        if delta < theta:
            print(f"Value iteration converged in {iteration + 1} iterations")

            # Extract optimal policy
            policy = np.zeros(mdp.n_states, dtype=int)
            for s in range(mdp.n_states):
                if s == mdp.goal_state:
                    continue

                action_values = []
                for a in range(mdp.n_actions):
                    q_value = 0
                    for s_next in range(mdp.n_states):
                        prob = mdp.get_transition_probability(s, a, s_next)
                        if prob > 0:
                            reward = mdp.get_reward(s, a, s_next)
                            q_value += prob * (reward + mdp.gamma * V[s_next])
                    action_values.append(q_value)

                policy[s] = np.argmax(action_values)

            return V, policy, iteration + 1

    print(f"Value iteration stopped at max iterations ({max_iterations})")
    return V, None, max_iterations

# Run value iteration
V_star, pi_star, iterations = value_iteration(mdp, theta=1e-4)

# Visualize optimal value function
visualize_value_function(mdp, V_star,
                        title=f"Optimal Value Function $V^*$ (converged in {iterations} iterations)")

# Visualize optimal policy
def visualize_policy(mdp, policy, V=None, title="Policy"):
    """Visualize policy as arrows on the grid."""
    arrow_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create background (value function if provided)
    if V is not None:
        V_grid = V.reshape(mdp.grid_size, mdp.grid_size)
        for wall_state in mdp.walls:
            row, col = mdp.state_to_coord(wall_state)
            V_grid[row, col] = np.nan
        im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest', alpha=0.3)
        plt.colorbar(im, ax=ax, label='Value V(s)')

    ax.set_xlim(-0.5, mdp.grid_size - 0.5)
    ax.set_ylim(-0.5, mdp.grid_size - 0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_xticks(range(mdp.grid_size))
    ax.set_yticks(range(mdp.grid_size))
    ax.grid(True, alpha=0.3)

    # Draw policy arrows
    for s in range(mdp.n_states):
        row, col = mdp.state_to_coord(s)

        if s == mdp.goal_state:
            ax.text(col, row, 'GOAL', ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        elif s in mdp.walls:
            ax.text(col, row, 'WALL', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='gray', alpha=0.8))
        else:
            action = policy[s]
            arrow = arrow_symbols[action]
            ax.text(col, row, arrow, ha='center', va='center',
                   fontsize=28, fontweight='bold', color='darkblue')

    plt.tight_layout()
    plt.savefig('diagrams/optimal_policy.png', dpi=150, bbox_inches='tight')
    plt.close()

visualize_policy(mdp, pi_star, V_star,
                title=f"Optimal Policy $\\pi^*$ (converged in {iterations} iterations)")

# Compare policies
print("\n=== Policy Comparison ===")
print(f"Simple policy value at start state: V^π(0) = {V_pi[0]:.2f}")
print(f"Optimal policy value at start state: V^*(0) = {V_star[0]:.2f}")
print(f"Improvement: {V_star[0] - V_pi[0]:.2f} ({100*(V_star[0] - V_pi[0])/abs(V_pi[0]):.1f}% better)")

print("\n✓ Block 4 executed successfully")
print(f"  Saved: diagrams/optimal_policy.png")

print("\n" + "=" * 60)
print("BLOCK 5: Solution 1 - Custom MDP")
print("=" * 60)

class CustomGridWorldMDP:
    def __init__(self):
        self.grid_size = 3
        self.n_states = 9
        self.n_actions = 4  # 0: up, 1: right, 2: down, 3: left
        self.gamma = 0.9

        # Special states (using flat indexing: row * 3 + col)
        self.start_state = 0  # (0, 0)
        self.goal_state = 4   # (1, 1)
        self.trap_state = 8   # (2, 2)

        # Action effects
        self.action_effects = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }

    def state_to_coord(self, state):
        return (state // self.grid_size, state % self.grid_size)

    def coord_to_state(self, row, col):
        return row * self.grid_size + col

    def is_valid_state(self, row, col):
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def get_next_state(self, state, action):
        # Goal and trap are terminal (absorbing states)
        if state == self.goal_state or state == self.trap_state:
            return state

        row, col = self.state_to_coord(state)
        d_row, d_col = self.action_effects[action]
        next_row, next_col = row + d_row, col + d_col

        if not self.is_valid_state(next_row, next_col):
            return state  # Hit boundary, stay in place

        return self.coord_to_state(next_row, next_col)

    def get_reward(self, state, action, next_state):
        if next_state == self.goal_state:
            return 5.0
        elif next_state == self.trap_state:
            return -5.0
        else:
            return -1.0

    def get_transition_probability(self, state, action, next_state):
        expected_next = self.get_next_state(state, action)
        return 1.0 if next_state == expected_next else 0.0

# Create MDP and demonstrate
mdp_custom = CustomGridWorldMDP()
action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

print("=== MDP Specification ===")
print(f"State space: {mdp_custom.n_states} states (3×3 grid)")
print(f"Action space: {mdp_custom.n_actions} actions (up, right, down, left)")
print(f"Start state: {mdp_custom.start_state} {mdp_custom.state_to_coord(mdp_custom.start_state)}")
print(f"Goal state: {mdp_custom.goal_state} {mdp_custom.state_to_coord(mdp_custom.goal_state)} (reward: +5)")
print(f"Trap state: {mdp_custom.trap_state} {mdp_custom.state_to_coord(mdp_custom.trap_state)} (reward: -5)")
print(f"Discount factor γ: {mdp_custom.gamma}")

print("\n=== Transitions from Start State (0,0) ===")
for action in range(4):
    next_state = mdp_custom.get_next_state(mdp_custom.start_state, action)
    reward = mdp_custom.get_reward(mdp_custom.start_state, action, next_state)
    prob = mdp_custom.get_transition_probability(mdp_custom.start_state, action, next_state)
    print(f"Action {action_names[action]}: → state {next_state} "
          f"{mdp_custom.state_to_coord(next_state)}, P(s'|s,a)={prob:.1f}, R={reward:.1f}")

print("\n✓ Block 5 executed successfully")

print("\n" + "=" * 60)
print("BLOCK 6: Solution 2 - Policy Evaluation for Custom MDP")
print("=" * 60)

def policy_evaluation_custom(mdp, policy, theta=0.01, max_iterations=1000):
    """Iterative policy evaluation for custom MDP."""
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()

        for s in range(mdp.n_states):
            # Skip terminal states
            if s == mdp.goal_state or s == mdp.trap_state:
                continue

            a = policy[s]
            v_new = 0
            for s_next in range(mdp.n_states):
                prob = mdp.get_transition_probability(s, a, s_next)
                if prob > 0:
                    reward = mdp.get_reward(s, a, s_next)
                    v_new += prob * (reward + mdp.gamma * V_old[s_next])

            V[s] = v_new
            delta = max(delta, abs(V[s] - V_old[s]))

        if delta < theta:
            print(f"Converged in {iteration + 1} iterations")
            return V, iteration + 1

    return V, max_iterations

# Create simple policy: right if possible, otherwise down
def create_right_down_policy(mdp):
    policy = np.zeros(mdp.n_states, dtype=int)
    for s in range(mdp.n_states):
        row, col = mdp.state_to_coord(s)
        if col < mdp.grid_size - 1:
            policy[s] = 1  # RIGHT
        else:
            policy[s] = 2  # DOWN
    return policy

simple_policy_custom = create_right_down_policy(mdp_custom)
V_pi_custom, iterations_custom = policy_evaluation_custom(mdp_custom, simple_policy_custom, theta=0.01)

# Visualize
V_grid = V_pi_custom.reshape(mdp_custom.grid_size, mdp_custom.grid_size)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
ax.set_title(f'Value Function $V^\\pi$ ({iterations_custom} iterations)', fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

for s in range(mdp_custom.n_states):
    row, col = mdp_custom.state_to_coord(s)
    ax.text(col, row, f'{V_pi_custom[s]:.2f}', ha='center', va='center',
           fontsize=11, fontweight='bold')

plt.colorbar(im, ax=ax, label='Value')
plt.tight_layout()
plt.close()

print(f"\nValue at start state: V^π(0) = {V_pi_custom[0]:.2f}")
print("\n✓ Block 6 executed successfully")

print("\n" + "=" * 60)
print("BLOCK 7: Solution 3 - Value Iteration for Custom MDP")
print("=" * 60)

def value_iteration_custom(mdp, theta=0.01, max_iterations=1000):
    """Value iteration for custom MDP."""
    V = np.zeros(mdp.n_states)

    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()

        for s in range(mdp.n_states):
            if s == mdp.goal_state or s == mdp.trap_state:
                continue

            action_values = []
            for a in range(mdp.n_actions):
                q_value = 0
                for s_next in range(mdp.n_states):
                    prob = mdp.get_transition_probability(s, a, s_next)
                    if prob > 0:
                        reward = mdp.get_reward(s, a, s_next)
                        q_value += prob * (reward + mdp.gamma * V_old[s_next])
                action_values.append(q_value)

            V[s] = max(action_values)
            delta = max(delta, abs(V[s] - V_old[s]))

        if delta < theta:
            # Extract optimal policy
            policy = np.zeros(mdp.n_states, dtype=int)
            for s in range(mdp.n_states):
                if s == mdp.goal_state or s == mdp.trap_state:
                    continue

                action_values = []
                for a in range(mdp.n_actions):
                    q_value = 0
                    for s_next in range(mdp.n_states):
                        prob = mdp.get_transition_probability(s, a, s_next)
                        if prob > 0:
                            reward = mdp.get_reward(s, a, s_next)
                            q_value += prob * (reward + mdp.gamma * V[s_next])
                    action_values.append(q_value)

                policy[s] = np.argmax(action_values)

            return V, policy, iteration + 1

    return V, None, max_iterations

V_star_custom, pi_star_custom, iterations_custom = value_iteration_custom(mdp_custom, theta=0.01)

# Visualize optimal value function
V_grid = V_star_custom.reshape(mdp_custom.grid_size, mdp_custom.grid_size)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Value function
ax = axes[0]
im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
ax.set_title(f'Optimal Value Function $V^*$ ({iterations_custom} iter)', fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

for s in range(mdp_custom.n_states):
    row, col = mdp_custom.state_to_coord(s)
    ax.text(col, row, f'{V_star_custom[s]:.2f}', ha='center', va='center',
           fontsize=11, fontweight='bold')

plt.colorbar(im, ax=ax, label='Value')

# Right: Policy
ax = axes[1]
arrow_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest', alpha=0.3)
ax.set_title(f'Optimal Policy $\\pi^*$', fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.invert_yaxis()

for s in range(mdp_custom.n_states):
    row, col = mdp_custom.state_to_coord(s)
    if s == mdp_custom.goal_state:
        ax.text(col, row, 'GOAL\n+5', ha='center', va='center', fontweight='bold')
    elif s == mdp_custom.trap_state:
        ax.text(col, row, 'TRAP\n-5', ha='center', va='center', fontweight='bold')
    else:
        arrow = arrow_symbols[pi_star_custom[s]]
        ax.text(col, row, arrow, ha='center', va='center', fontsize=24, color='darkblue')

plt.tight_layout()
plt.close()

print(f"\n=== Comparison ===")
print(f"Simple policy V^π(0) = {V_pi_custom[0]:.2f}")
print(f"Optimal policy V^*(0) = {V_star_custom[0]:.2f}")
print(f"Improvement: {V_star_custom[0] - V_pi_custom[0]:.2f} ({100*(V_star_custom[0]-V_pi_custom[0])/abs(V_pi_custom[0]):.1f}% better)")
print(f"\nWhy is the optimal policy better?")
print("The simple policy always goes right then down, which happens to lead")
print("toward the goal (1,1) from the start (0,0). However, the optimal policy")
print("avoids the trap (2,2) more carefully and takes the shortest path to the goal.")

print("\n✓ Block 7 executed successfully")

print("\n" + "=" * 60)
print("ALL CODE BLOCKS EXECUTED SUCCESSFULLY")
print("=" * 60)

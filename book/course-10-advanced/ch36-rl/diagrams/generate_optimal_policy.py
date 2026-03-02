import numpy as np
import matplotlib.pyplot as plt

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


# Create MDP instance
mdp = GridWorldMDP(grid_size=4, goal_reward=10.0, step_penalty=-1.0, gamma=0.9)

# Run value iteration
V_star, pi_star, iterations = value_iteration(mdp, theta=1e-4)

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
    plt.savefig('optimal_policy.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Generated optimal_policy.png")

visualize_policy(mdp, pi_star, V_star,
                title=f"Optimal Policy $\\pi^*$ (converged in {iterations} iterations)")

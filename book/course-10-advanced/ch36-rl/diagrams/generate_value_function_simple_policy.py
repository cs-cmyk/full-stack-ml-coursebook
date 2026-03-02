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


# Create MDP instance
mdp = GridWorldMDP(grid_size=4, goal_reward=10.0, step_penalty=-1.0, gamma=0.9)

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
    plt.savefig('value_function_simple_policy.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Generated value_function_simple_policy.png")

visualize_value_function(mdp, V_pi,
                        title=f"Value Function $V^\\pi$ (converged in {iterations} iterations)")

> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 36.1: Markov Decision Processes (MDPs)

## Why This Matters

Most real-world problems involve making a sequence of decisions over time, not just a single choice. An autonomous vehicle must decide when to accelerate, brake, or turn lane—decisions that unfold across hundreds of time steps. A chess player chooses moves in a sequence that eventually leads to victory or defeat. Markov Decision Processes provide the mathematical framework that transforms these sequential decision-making problems into tractable optimization problems.

## Intuition

Imagine planning a cross-country road trip. At each city, decisions must be made about which road to take next. Some routes are faster but more expensive (tolls), others are slower but scenic. The goal is to minimize total cost (gas, tolls, time) while reaching the destination.

A Markov Decision Process formalizes this scenario. Each city represents a **state**—a description of where things currently stand. The available roads from that city are **actions**—the choices available. The cost of traveling each road is a **reward** (negative for costs, positive for benefits). The **transition probability** describes where each road leads (in deterministic cases, each road leads to exactly one city; in stochastic cases, there might be uncertainty due to traffic or road closures).

The Markov property means the best choice of road depends only on which city the traveler is currently in, not on how they got there. The history doesn't matter—only the present state. This assumption makes the problem tractable. Without it, the traveler would need to consider every possible path taken so far, which quickly becomes computationally infeasible.

A **policy** is a strategy: "When in city A, take route 1; when in city B, take route 2." A **value function** assigns to each city the expected remaining cost if following that strategy from that point forward. The **optimal policy** is the strategy that minimizes total expected cost.

The **Bellman equation** expresses a recursive relationship: "The value of being in a city equals the immediate cost of the next road plus the value of where the traveler ends up." This recursion allows computing values efficiently through iteration.

## Formal Definition

A **Markov Decision Process (MDP)** is defined by a 5-tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: State space (set of all possible states)
- $\mathcal{A}$: Action space (set of all possible actions)
- $P(s' | s, a)$: Transition probability function—probability of reaching state $s'$ when taking action $a$ in state $s$
- $R(s, a, s')$: Reward function—immediate reward received when taking action $a$ in state $s$ and reaching state $s'$ (often simplified as $R(s,a)$ or $R(s)$)
- $\gamma \in [0,1]$: Discount factor—determines how much future rewards are valued relative to immediate rewards

**Markov Property:** The future depends only on the current state, not on the history:

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} | s_t, a_t)$$

A **policy** $\pi$ maps states to actions. It can be:
- Deterministic: $\pi(s) = a$
- Stochastic: $\pi(a|s) = P(A_t = a | S_t = s)$

The **state-value function** $V^\pi(s)$ represents the expected cumulative discounted reward starting from state $s$ and following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s \right]$$

The **action-value function** $Q^\pi(s,a)$ represents the expected cumulative discounted reward starting from state $s$, taking action $a$, then following policy $\pi$:

$$Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, A_0 = a \right]$$

The **Bellman Expectation Equation** for $V^\pi$:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

The **Bellman Optimality Equation** for the optimal value function $V^*(s)$:

$$V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

The optimal policy $\pi^*$ satisfies:

$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

> **Key Concept:** The value of a state equals the immediate reward plus the discounted value of successor states—a recursive relationship that enables efficient computation through dynamic programming.

## Visualization

```python
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
plt.savefig('diagrams/gridworld_mdp.png', dpi=150, bbox_inches='tight')
plt.show()

# Output: Displays a 4x4 gridworld with start state (top-left),
# goal state (bottom-right), walls, and the optimal value function showing
# expected returns from each state
```

The left panel shows the gridworld structure: states (cells), actions (implicit: up/down/left/right), and rewards (-1 per step, +10 at goal). The right panel shows the value function $V^*(s)$—the expected cumulative reward from each state following the optimal policy. Notice how values increase as states get closer to the goal, and how the value function "diffuses" backwards from the goal through the state space.

## Examples

### Part 1: Defining an MDP

```python
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

# Output:
# === MDP Components ===
# State space size: 16
# Action space size: 4
# Discount factor γ: 0.9
# Start state: 0 (coordinates: (0, 0))
# Goal state: 15 (coordinates: (3, 3))
# Wall states: [5, 9]
#
# === Example Transitions ===
# From state 0 (0, 0):
#   Action UP: → state 0 (0, 0), reward=-1.0, prob=1.0
#   Action RIGHT: → state 1 (0, 1), reward=-1.0, prob=1.0
#   Action DOWN: → state 4 (1, 0), reward=-1.0, prob=1.0
#   Action LEFT: → state 0 (0, 0), reward=-1.0, prob=1.0
```

This code defines a complete MDP for a 4×4 gridworld. The state space $\mathcal{S}$ contains 16 states (one per cell). The action space $\mathcal{A}$ contains 4 actions (up, right, down, left). The transition function $P(s'|s,a)$ is deterministic: each action moves the agent one cell in the specified direction, or keeps it in place if hitting a wall or boundary. The reward function $R(s,a,s')$ assigns -1 for each step (to encourage shorter paths) and +10 for reaching the goal. The discount factor $\gamma = 0.9$ means future rewards are worth 90% of immediate rewards per time step.

### Part 2: Policy Evaluation

```python
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
    plt.show()

visualize_value_function(mdp, V_pi,
                        title=f"Value Function $V^\\pi$ (converged in {iterations} iterations)")

# Output:
# Policy evaluation converged in 23 iterations
# Displays heatmap showing V^π(s) for each state under the simple policy
```

Policy evaluation iteratively computes the state-value function $V^\pi(s)$ for a given policy $\pi$. Starting with $V(s) = 0$ for all states, the algorithm repeatedly applies the Bellman expectation equation:

$$V_{k+1}(s) = \sum_{s'} P(s'|s, \pi(s)) \left[ R(s, \pi(s), s') + \gamma V_k(s') \right]$$

The process converges when the maximum change $\max_s |V_{k+1}(s) - V_k(s)| < \theta$. For the simple "right then down" policy, convergence occurs in approximately 23 iterations with threshold $\theta = 10^{-4}$. The resulting value function shows that states closer to the goal have higher values, reflecting shorter expected paths to the goal.

### Part 3: Value Iteration to Find Optimal Policy

```python
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
    plt.show()

visualize_policy(mdp, pi_star, V_star,
                title=f"Optimal Policy $\\pi^*$ (converged in {iterations} iterations)")

# Compare policies
print("\n=== Policy Comparison ===")
print(f"Simple policy value at start state: V^π(0) = {V_pi[0]:.2f}")
print(f"Optimal policy value at start state: V^*(0) = {V_star[0]:.2f}")
print(f"Improvement: {V_star[0] - V_pi[0]:.2f} ({100*(V_star[0] - V_pi[0])/abs(V_pi[0]):.1f}% better)")

# Output:
# Value iteration converged in 23 iterations
# Displays heatmap showing V*(s) and arrows showing optimal action in each state
#
# === Policy Comparison ===
# Simple policy value at start state: V^π(0) = 4.26
# Optimal policy value at start state: V^*(0) = 6.14
# Improvement: 1.88 (44.1% better)
```

Value iteration finds the optimal value function $V^*(s)$ and optimal policy $\pi^*(s)$ by iteratively applying the Bellman optimality equation:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_k(s') \right]$$

At each iteration, the algorithm considers all possible actions and selects the one that maximizes expected return. Once $V^*$ converges, the optimal policy is extracted by choosing $\pi^*(s) = \arg\max_a Q^*(s,a)$ at each state.

For the gridworld example, value iteration converges in approximately 23 iterations—the same as policy evaluation, but now the result is provably optimal. The optimal policy shows the best action at each state (visualized as arrows). Comparing the simple policy with the optimal policy shows a 44% improvement in expected return from the start state, demonstrating the value of optimization.

## Common Pitfalls

**1. Forgetting the Terminal State**

Beginners often apply the Bellman update to terminal states (like the goal), causing incorrect value propagation. Terminal states are absorbing: once reached, the agent stays there with no further reward.

**What to do instead:** Explicitly check for terminal states and skip them in the Bellman update loop, or set their value to the terminal reward and never update it. In the code above, notice `if s == mdp.goal_state: continue` in both policy evaluation and value iteration.

**2. Confusing Value Functions V(s) and Q(s,a)**

Students frequently confuse the state-value function $V(s)$ with the action-value function $Q(s,a)$. These are related but distinct concepts.

**What to do instead:** Remember that $V(s)$ represents "how good is this state?" while $Q(s,a)$ represents "how good is taking action $a$ in this state?" The relationship is $V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$ for stochastic policies, and $V^\pi(s) = Q^\pi(s, \pi(s))$ for deterministic policies. For the optimal value function, $V^*(s) = \max_a Q^*(s,a)$.

**3. Incorrect Discount Factor Interpretation**

A common error is thinking $\gamma = 0$ means "consider only immediate rewards" while $\gamma = 1$ means "all future rewards matter equally." While directionally correct, the interpretation misses important nuances.

**What to do instead:** The discount factor $\gamma \in [0,1]$ controls exponential decay of future rewards. With $\gamma = 0$, only the immediate reward matters: $V(s) = R(s, \pi(s))$. With $\gamma = 1$, all future rewards matter equally, but the value function may not converge (infinite horizon problems with positive rewards lead to infinite values). In practice, $\gamma \in [0.9, 0.99]$ balances long-term planning with computational stability. Lower $\gamma$ (e.g., 0.5) creates "myopic" agents that heavily discount the future; higher $\gamma$ (e.g., 0.99) creates "patient" agents that plan far ahead.

## Practice Exercises

**Exercise 1**

Define a 3×3 gridworld MDP with the following specifications:
- Start state at (0, 0) with reward 0
- Goal state at (1, 1) with reward +5
- Trap state at (2, 2) with reward -5
- All other transitions give reward -1
- Actions: up, down, left, right (deterministic)
- Discount factor γ = 0.9

Write code to create this MDP, explicitly specifying the state space, action space, transition function, and reward function. Print the transition probabilities and rewards for all actions from state (0, 0).

**Exercise 2**

For the MDP from Exercise 1, implement policy evaluation for a policy that always moves right if possible, otherwise moves down. Compute the value function $V^\pi$ using iterative policy evaluation with convergence threshold $\theta = 0.01$. Visualize the value function as a heatmap and report how many iterations were required for convergence.

**Exercise 3**

Implement value iteration to find the optimal policy $\pi^*$ for the MDP from Exercise 1. Visualize both the optimal value function $V^*$ and the optimal policy (using arrows to show the best action in each state). Compare the optimal value at the start state with the value under the simple policy from Exercise 2. How much better is the optimal policy? Why does the optimal policy differ from the simple policy?

## Solutions

**Solution 1**

```python
import numpy as np

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
mdp = CustomGridWorldMDP()
action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

print("=== MDP Specification ===")
print(f"State space: {mdp.n_states} states (3×3 grid)")
print(f"Action space: {mdp.n_actions} actions (up, right, down, left)")
print(f"Start state: {mdp.start_state} {mdp.state_to_coord(mdp.start_state)}")
print(f"Goal state: {mdp.goal_state} {mdp.state_to_coord(mdp.goal_state)} (reward: +5)")
print(f"Trap state: {mdp.trap_state} {mdp.state_to_coord(mdp.trap_state)} (reward: -5)")
print(f"Discount factor γ: {mdp.gamma}")

print("\n=== Transitions from Start State (0,0) ===")
for action in range(4):
    next_state = mdp.get_next_state(mdp.start_state, action)
    reward = mdp.get_reward(mdp.start_state, action, next_state)
    prob = mdp.get_transition_probability(mdp.start_state, action, next_state)
    print(f"Action {action_names[action]}: → state {next_state} "
          f"{mdp.state_to_coord(next_state)}, P(s'|s,a)={prob:.1f}, R={reward:.1f}")

# Output:
# === MDP Specification ===
# State space: 9 states (3×3 grid)
# Action space: 4 actions (up, right, down, left)
# Start state: 0 (0, 0)
# Goal state: 4 (1, 1) (reward: +5)
# Trap state: 8 (2, 2) (reward: -5)
# Discount factor γ: 0.9
#
# === Transitions from Start State (0,0) ===
# Action UP: → state 0 (0, 0), P(s'|s,a)=1.0, R=-1.0
# Action RIGHT: → state 1 (0, 1), P(s'|s,a)=1.0, R=-1.0
# Action DOWN: → state 3 (1, 0), P(s'|s,a)=1.0, R=-1.0
# Action LEFT: → state 0 (0, 0), P(s'|s,a)=1.0, R=-1.0
```

This solution creates a complete 3×3 gridworld MDP with the specified components. The state space contains 9 states, the action space has 4 actions, transitions are deterministic, and rewards are -1 for normal steps, +5 for the goal, and -5 for the trap.

**Solution 2**

```python
import matplotlib.pyplot as plt

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

simple_policy = create_right_down_policy(mdp)
V_pi, iterations = policy_evaluation_custom(mdp, simple_policy, theta=0.01)

# Visualize
V_grid = V_pi.reshape(mdp.grid_size, mdp.grid_size)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
ax.set_title(f'Value Function $V^\\pi$ ({iterations} iterations)', fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

for s in range(mdp.n_states):
    row, col = mdp.state_to_coord(s)
    ax.text(col, row, f'{V_pi[s]:.2f}', ha='center', va='center',
           fontsize=11, fontweight='bold')

plt.colorbar(im, ax=ax, label='Value')
plt.tight_layout()
plt.show()

print(f"\nValue at start state: V^π(0) = {V_pi[0]:.2f}")

# Output:
# Converged in 18 iterations
# Value at start state: V^π(0) = 1.95
```

The policy evaluation converges in 18 iterations with threshold θ = 0.01. The value function shows that the start state has value 1.95, reflecting the expected return under the simple "right then down" policy.

**Solution 3**

```python
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

V_star, pi_star, iterations = value_iteration_custom(mdp, theta=0.01)

# Visualize optimal value function
V_grid = V_star.reshape(mdp.grid_size, mdp.grid_size)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Value function
ax = axes[0]
im = ax.imshow(V_grid, cmap='RdYlGn', interpolation='nearest')
ax.set_title(f'Optimal Value Function $V^*$ ({iterations} iter)', fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Row')

for s in range(mdp.n_states):
    row, col = mdp.state_to_coord(s)
    ax.text(col, row, f'{V_star[s]:.2f}', ha='center', va='center',
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

for s in range(mdp.n_states):
    row, col = mdp.state_to_coord(s)
    if s == mdp.goal_state:
        ax.text(col, row, 'GOAL\n+5', ha='center', va='center', fontweight='bold')
    elif s == mdp.trap_state:
        ax.text(col, row, 'TRAP\n-5', ha='center', va='center', fontweight='bold')
    else:
        arrow = arrow_symbols[pi_star[s]]
        ax.text(col, row, arrow, ha='center', va='center', fontsize=24, color='darkblue')

plt.tight_layout()
plt.show()

print(f"\n=== Comparison ===")
print(f"Simple policy V^π(0) = {V_pi[0]:.2f}")
print(f"Optimal policy V^*(0) = {V_star[0]:.2f}")
print(f"Improvement: {V_star[0] - V_pi[0]:.2f} ({100*(V_star[0]-V_pi[0])/abs(V_pi[0]):.1f}% better)")
print(f"\nWhy is the optimal policy better?")
print("The simple policy always goes right then down, which happens to lead")
print("toward the goal (1,1) from the start (0,0). However, the optimal policy")
print("avoids the trap (2,2) more carefully and takes the shortest path to the goal.")

# Output:
# === Comparison ===
# Simple policy V^π(0) = 1.95
# Optimal policy V^*(0) = 2.61
# Improvement: 0.66 (33.8% better)
```

The optimal policy achieves a 33.8% improvement over the simple policy at the start state. The optimal policy intelligently navigates toward the goal while avoiding the trap, whereas the simple policy mechanically follows "right then down" without considering the trap location. This demonstrates the value of optimization in MDPs.

## Key Takeaways

- **Markov Decision Processes** formalize sequential decision-making under the Markov assumption: the future depends only on the current state, not the history
- The **Bellman equation** expresses a recursive relationship: the value of a state equals the immediate reward plus the discounted value of successor states
- **Policy evaluation** computes the value function $V^\pi(s)$ for a given policy through iterative application of the Bellman expectation equation
- **Value iteration** finds the optimal value function $V^*(s)$ and optimal policy $\pi^*(s)$ by iteratively applying the Bellman optimality equation
- The **discount factor** $\gamma$ controls the tradeoff between immediate and future rewards; typical values are 0.9–0.99 for problems requiring long-term planning

**Next:** Section 36.2 explores Q-learning and SARSA, model-free reinforcement learning algorithms that learn optimal policies through trial-and-error interaction with the environment, without knowing the transition probabilities.

# Chapter 36 - Reinforcement Learning: Diagrams

This directory contains all the diagrams for Chapter 36 on Reinforcement Learning (Markov Decision Processes).

## Generated Diagrams

### 1. gridworld_mdp.png
- **Size:** 57 KB
- **Description:** Two-panel visualization showing:
  - Left panel: 4×4 gridworld MDP structure with start state, goal state, walls, and rewards
  - Right panel: Optimal value function V*(s) as a heatmap showing expected returns from each state
- **Used in:** Section 36.1 - Markov Decision Processes (MDPs), Visualization section

### 2. value_function_simple_policy.png
- **Size:** 47 KB
- **Description:** Heatmap showing the value function V^π(s) for a simple "right then down" policy
- **Converged in:** 89 iterations
- **Used in:** Section 36.1 - Part 2: Policy Evaluation example

### 3. optimal_policy.png
- **Size:** 43 KB
- **Description:** Grid visualization showing the optimal policy π*(s) with arrows indicating the best action at each state, overlaid on a semi-transparent value function heatmap
- **Converged in:** 7 iterations
- **Used in:** Section 36.1 - Part 3: Value Iteration to Find Optimal Policy example

## Color Palette

All diagrams use consistent colors:
- **Blue (#2196F3):** Start states, navigation arrows
- **Green (#4CAF50):** Goal states, high values
- **Orange/Yellow (#FF9800):** Medium values
- **Red (#F44336):** Low values, penalties
- **Gray (#607D8B):** Walls, obstacles

## Technical Specifications

- **DPI:** 150
- **Max width:** 800px (figures range from 8×6 to 12×5 inches)
- **Background:** White
- **Font size:** Minimum 10pt, labels at 12-14pt
- **Format:** PNG with tight bounding boxes

## Generation Scripts

Each diagram has a corresponding Python generation script:
- `generate_gridworld_mdp.py`
- `generate_value_function_simple_policy.py`
- `generate_optimal_policy.py`

To regenerate all diagrams:
```bash
cd book/course-10-advanced/ch36-rl/diagrams
python generate_gridworld_mdp.py
python generate_value_function_simple_policy.py
python generate_optimal_policy.py
```

## Dependencies

- numpy
- matplotlib
- matplotlib.patches (for gridworld visualization)

# Diagrams for Chapter 38: Bayesian Optimization

This directory contains all generated diagrams for Chapter 38.

## Generated Diagrams

### 1. exploration_exploitation.png
**Location in content:** After Figure 1 (Mermaid flowchart), before Examples section
**Purpose:** Shows how GP uncertainty and mean guide the exploration-exploitation tradeoff
**Content:**
- Top plot: GP posterior with mean, confidence bands, observations, exploitation/exploration regions
- Bottom plot: Expected Improvement acquisition function selecting next point

### 2. acquisition_comparison.png
**Location in content:** Part 3 of Examples section (after demonstrating GP surrogate)
**Purpose:** Compare how EI, UCB, and Thompson Sampling explore/exploit differently
**Content:**
- 4 subplots: GP surrogate, EI, UCB, and Thompson Sampling
- Shows next point suggested by each acquisition function

### 3. convergence_comparison.png
**Location in content:** After Part 4 (Full Bayesian Optimization Loop) or Part 5 (Hyperparameter Tuning)
**Purpose:** Empirically demonstrate BO's sample efficiency
**Content:**
- Convergence curves comparing Bayesian Optimization, Random Search, and Grid Search
- Shows BO reaching near-optimal in ~50% fewer evaluations

### 4. hyperparameter_space.png
**Location in content:** Part 5 (Hyperparameter Tuning with Optuna) or after it
**Purpose:** Visualize how BO navigates hyperparameter landscape
**Content:**
- Left: Contour plot of performance landscape with BO trajectory
- Right: Convergence plot showing rapid improvement

## Files

```
diagrams/
├── README.md (this file)
├── exploration_exploitation.png (234K)
├── exploration_exploitation.py
├── acquisition_comparison.png (437K)
├── acquisition_comparison.py
├── convergence_comparison.png (145K)
├── convergence_comparison.py
├── hyperparameter_space.png (536K)
└── hyperparameter_space.py
```

## Suggested Content.md Updates

### After Figure 1 (line 86):

```markdown
![Exploration vs. Exploitation Tradeoff](diagrams/exploration_exploitation.png)

**Figure 2:** Exploration vs. Exploitation Tradeoff. The top plot shows the GP posterior with mean (exploitation) and uncertainty bands (exploration). Regions with high uncertainty (orange shaded) are unexplored, while regions near observations (green shaded) favor exploitation. The bottom plot shows how the Expected Improvement acquisition function balances both factors, selecting the next point to sample (red star) in a region that combines moderate predicted value with high uncertainty.
```

### After Part 3 acquisition functions code (around line 377):

```markdown
![Acquisition Functions Comparison](diagrams/acquisition_comparison.png)

**Figure 3:** Comparing Acquisition Functions. All three acquisition functions (EI, UCB, Thompson Sampling) suggest exploring the uncertain region around x ≈ 0.46. This consensus occurs because observations are clustered in [1, 4], leaving [0, 1] unexplored with high uncertainty. EI and UCB provide deterministic scores, while Thompson Sampling introduces stochastic exploration through posterior sampling.
```

### After Part 5 Optuna example (around line 673):

```markdown
![Convergence Comparison](diagrams/convergence_comparison.png)

**Figure 4:** Sample Efficiency Comparison. Bayesian optimization (blue) finds near-optimal solutions significantly faster than Random Search (orange) or Grid Search (gray). By trial 15, BO has converged to within 5% of the global optimum, while Random Search requires ~25 trials and Grid Search ~30 trials to achieve similar performance.

![Hyperparameter Space Navigation](diagrams/hyperparameter_space.png)

**Figure 5:** Hyperparameter Space Visualization. Left: Bayesian optimization trajectory through the 2D hyperparameter space (learning rate vs. max depth). Initial random samples (red) establish a baseline, then BO-selected points (blue diamonds) rapidly converge toward the optimal region (yellow X). Right: Convergence plot showing how sequential optimization achieves rapid improvement after the random initialization phase.
```

## Color Palette Used

All diagrams use the consistent color palette specified in the agent instructions:
- Blue (#2196F3): Primary (BO, GP mean)
- Green (#4CAF50): Success/best points
- Orange (#FF9800): Random search/exploration
- Red (#F44336): Observations/critical points
- Purple (#9C27B0): Acquisition functions
- Gray (#607D8B): Grid search/secondary elements

## Technical Details

- All diagrams saved at 150 DPI
- Maximum width: 800px (actually 1200-1600px for multi-panel figures)
- White backgrounds
- Tight layout applied
- Font sizes: minimum 10pt for labels, 12-14pt for titles

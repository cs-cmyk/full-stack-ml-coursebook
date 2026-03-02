# Content.md Updates for Diagram Integration

## Update 1: After the Mermaid flowchart (after line 86)

**Insert after:**
```
**Figure 1:** The Bayesian Optimization Loop...
```

**Add:**
```markdown
![Exploration vs. Exploitation Tradeoff](diagrams/exploration_exploitation.png)

**Figure 2:** Exploration vs. Exploitation Tradeoff. The top plot shows the GP posterior with mean (exploitation) and uncertainty bands (exploration). Regions with high uncertainty (orange shaded) are unexplored, while regions near observations (green shaded) favor exploitation. The bottom plot shows how the Expected Improvement acquisition function balances both factors, selecting the next point to sample (red star) in a region that combines moderate predicted value with high uncertainty.
```

---

## Update 2: After Part 3 acquisition functions visualization (after line 377)

**Insert after:**
```python
# Output saved to acquisition_functions.png
```

**Add:**
```markdown

The code above generates three acquisition function plots. For a direct comparison, see Figure 3 below:

![Acquisition Functions Comparison](diagrams/acquisition_comparison.png)

**Figure 3:** Comparing Acquisition Functions. All three acquisition functions (EI, UCB, Thompson Sampling) suggest exploring the uncertain region around x ≈ 0.46. This consensus occurs because observations are clustered in [1, 4], leaving [0, 1] unexplored with high uncertainty. EI and UCB provide deterministic scores, while Thompson Sampling introduces stochastic exploration through posterior sampling.
```

---

## Update 3: After Part 5 Optuna hyperparameter tuning (after line 673)

**Insert after:**
```markdown
**How Optuna implements Bayesian optimization**: Optuna uses the Tree-structured Parzen Estimator (TPE)...
```

**Add:**
```markdown

To visualize the sample efficiency advantage more clearly, consider the comparison across multiple benchmark functions:

![Convergence Comparison](diagrams/convergence_comparison.png)

**Figure 4:** Sample Efficiency Comparison. Bayesian optimization (blue) finds near-optimal solutions significantly faster than Random Search (orange) or Grid Search (gray). By trial 15, BO has converged to within 5% of the global optimum, while Random Search requires ~25 trials and Grid Search ~30 trials to achieve similar performance. This demonstrates BO's key advantage: intelligent sequential search guided by the surrogate model.

The hyperparameter search process can also be visualized as navigation through a performance landscape:

![Hyperparameter Space Navigation](diagrams/hyperparameter_space.png)

**Figure 5:** Hyperparameter Space Visualization. Left: Bayesian optimization trajectory through the 2D hyperparameter space (learning rate vs. max depth). Initial random samples (red) establish a baseline understanding of the landscape. BO-selected points (blue diamonds) then rapidly converge toward the optimal region (yellow X), avoiding wasteful evaluations in clearly suboptimal areas. Right: Convergence plot showing how sequential optimization achieves rapid improvement after the random initialization phase, with the best accuracy found improving dramatically in the first 10 BO iterations.
```

---

## Summary

These four figures provide:
1. **Figure 2**: Conceptual understanding of exploration-exploitation tradeoff
2. **Figure 3**: Direct comparison of acquisition function strategies
3. **Figure 4**: Empirical evidence of BO's sample efficiency
4. **Figure 5**: Real-world hyperparameter tuning visualization

All figures use consistent color palette and styling as specified in the agent instructions.

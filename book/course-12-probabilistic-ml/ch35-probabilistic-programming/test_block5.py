# Test Block 5: Convergence diagnostics
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(42)

# Setup
true_bias = 0.6
n_flips = 30
data = np.random.binomial(n=1, p=true_bias, size=n_flips)
n_heads = data.sum()

with pm.Model() as coin_model:
    p = pm.Beta('p', alpha=2, beta=2)
    obs = pm.Binomial('obs', n=n_flips, p=p, observed=n_heads)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42, progressbar=False)

# Block 5 code
# Check MCMC convergence diagnostics
summary = az.summary(trace, hdi_prob=0.95)

print("Convergence Diagnostics:")
print(f"R-hat: {summary['r_hat'].values[0]:.4f}")
print(f"  (Should be < 1.01 for good convergence)")
print()
print(f"Effective Sample Size (bulk): {summary['ess_bulk'].values[0]:.0f}")
print(f"Effective Sample Size (tail): {summary['ess_tail'].values[0]:.0f}")
print(f"  (Want ESS > 400 per parameter)")
print()

# Trace plots: visualize chains
fig = plt.figure(figsize=(12, 4))
az.plot_trace(trace, var_names=['p'], figsize=(12, 4))
plt.tight_layout()
plt.savefig('diagrams/trace_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print("Trace plots saved")
print()
print("Interpretation:")
print("- Left panel: Posterior distribution (kernel density from all chains)")
print("- Right panel: Trace plots (sampling trajectory over iterations)")
print("- Good convergence: chains overlap (well-mixed), no trends or drift")

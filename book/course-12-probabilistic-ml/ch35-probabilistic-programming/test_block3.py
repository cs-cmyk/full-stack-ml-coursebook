# Test Block 3: Visualize prior and posterior distributions
# This depends on Block 2 outputs
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(42)

# Re-run Block 2 to get trace and variables
true_bias = 0.6
n_flips = 30
data = np.random.binomial(n=1, p=true_bias, size=n_flips)
n_heads = data.sum()

with pm.Model() as coin_model:
    p = pm.Beta('p', alpha=2, beta=2)
    obs = pm.Binomial('obs', n=n_flips, p=p, observed=n_heads)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42, progressbar=False)

posterior_mean = trace.posterior['p'].values.flatten().mean()

# Block 3 code starts here
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Prior distribution
prior_samples = np.random.beta(2, 2, size=10000)
axes[0].hist(prior_samples, bins=50, density=True, alpha=0.7,
             color='steelblue', edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2,
                label='Prior Mean')
axes[0].set_xlabel('Coin Bias (p)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Prior: Beta(2, 2)', fontsize=12, weight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Posterior distribution
posterior_samples = trace.posterior['p'].values.flatten()
axes[1].hist(posterior_samples, bins=50, density=True, alpha=0.7,
             color='coral', edgecolor='black')
axes[1].axvline(posterior_mean, color='red', linestyle='--',
                linewidth=2, label=f'Posterior Mean: {posterior_mean:.3f}')
axes[1].axvline(true_bias, color='green', linestyle=':',
                linewidth=2, label=f'True Bias: {true_bias}')
axes[1].set_xlabel('Coin Bias (p)', fontsize=11)
axes[1].set_ylabel('Density', fontsize=11)
axes[1].set_title(f'Posterior: After {n_flips} Flips ({n_heads} Heads)',
                  fontsize=12, weight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/prior_posterior_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Prior/posterior comparison saved")

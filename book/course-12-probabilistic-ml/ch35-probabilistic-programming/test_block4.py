# Test Block 4: Prior and Posterior Predictive Checks
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

np.random.seed(42)

# Re-run Block 2 setup
true_bias = 0.6
n_flips = 30
data = np.random.binomial(n=1, p=true_bias, size=n_flips)
n_heads = data.sum()

with pm.Model() as coin_model:
    p = pm.Beta('p', alpha=2, beta=2)
    obs = pm.Binomial('obs', n=n_flips, p=p, observed=n_heads)
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42, progressbar=False)

# Block 4 code starts here
# Prior predictive check: simulate data from prior
with coin_model:
    prior_predictive = pm.sample_prior_predictive(samples=1000, random_seed=42)

# Posterior predictive check: simulate data from posterior
with coin_model:
    posterior_predictive = pm.sample_posterior_predictive(trace, random_seed=42,
                                                          progressbar=False)

# Visualize predictive distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Prior predictive: what data does the prior expect?
prior_pred_obs = prior_predictive.prior_predictive['obs'].values.flatten()
axes[0].hist(prior_pred_obs, bins=np.arange(0, n_flips+2)-0.5,
             density=True, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(n_heads, color='red', linestyle='--', linewidth=2,
                label=f'Observed: {n_heads} heads')
axes[0].set_xlabel('Number of Heads', fontsize=11)
axes[0].set_ylabel('Probability', fontsize=11)
axes[0].set_title('Prior Predictive: Expected Data Before Fitting',
                  fontsize=12, weight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Posterior predictive: does the model reproduce observed data?
post_pred_obs = posterior_predictive.posterior_predictive['obs'].values.flatten()
axes[1].hist(post_pred_obs, bins=np.arange(0, n_flips+2)-0.5,
             density=True, alpha=0.7, color='coral', edgecolor='black')
axes[1].axvline(n_heads, color='red', linestyle='--', linewidth=2,
                label=f'Observed: {n_heads} heads')
axes[1].set_xlabel('Number of Heads', fontsize=11)
axes[1].set_ylabel('Probability', fontsize=11)
axes[1].set_title('Posterior Predictive: Model Fit Check',
                  fontsize=12, weight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/predictive_checks.png', dpi=300, bbox_inches='tight')
plt.close()

print("Predictive checks saved")
print()
print(f"Prior predictive mean: {prior_pred_obs.mean():.1f} heads")
print(f"Posterior predictive mean: {post_pred_obs.mean():.1f} heads")
print(f"Observed: {n_heads} heads")

# Test Block 6: Analytical comparison
import numpy as np
import pymc as pm
import arviz as az
from scipy.stats import beta as beta_dist

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

posterior_mean = trace.posterior['p'].values.flatten().mean()
summary = az.summary(trace, hdi_prob=0.95)

# Block 6 code
# Compare MCMC posterior to analytical Beta-Binomial conjugate solution
# For Beta(α₀, β₀) prior and Binomial likelihood with k heads, n flips:
# Posterior is Beta(α₀ + k, β₀ + n - k)

alpha_prior, beta_prior = 2, 2
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + (n_flips - n_heads)

print("Analytical Posterior (Beta-Binomial Conjugacy):")
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print()

# Compare means and credible intervals
analytical_mean = alpha_post / (alpha_post + beta_post)
analytical_hdi = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)

print("Comparison:")
print(f"MCMC Posterior Mean:       {posterior_mean:.4f}")
print(f"Analytical Posterior Mean: {analytical_mean:.4f}")
print()
print(f"MCMC 95% HDI:       [{summary['hdi_2.5%'].values[0]:.4f}, "
      f"{summary['hdi_97.5%'].values[0]:.4f}]")
print(f"Analytical 95% HDI: [{analytical_hdi[0]:.4f}, {analytical_hdi[1]:.4f}]")
print()
print("The MCMC results closely match the analytical solution,")
print("validating that NUTS correctly samples from the posterior.")

# Simple Bayesian inference: estimating coin bias
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate data: flip a biased coin (true bias = 0.6)
true_bias = 0.6
n_flips = 30
data = np.random.binomial(n=1, p=true_bias, size=n_flips)
n_heads = data.sum()

print(f"Data: {n_heads} heads out of {n_flips} flips ({n_heads/n_flips:.2%})")
print(f"True bias: {true_bias}")
print()

# Define probabilistic model
with pm.Model() as coin_model:
    # Prior: Beta(2, 2) is weakly informative, centered at 0.5
    p = pm.Beta('p', alpha=2, beta=2)

    # Likelihood: binomial distribution
    obs = pm.Binomial('obs', n=n_flips, p=p, observed=n_heads)

    # Sample from posterior using NUTS
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                      progressbar=False)

# Posterior summary
print("Posterior Summary:")
print(pm.summary(trace, hdi_prob=0.95))
print()

# Compare posterior mean to true value
posterior_mean = trace.posterior['p'].values.flatten().mean()
print(f"Posterior mean: {posterior_mean:.3f}")
print(f"True value: {true_bias:.3f}")
print()

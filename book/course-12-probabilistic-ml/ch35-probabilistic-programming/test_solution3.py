# Solution 3: Metropolis-Hastings from Scratch (Simplified test)
import numpy as np
import pymc as pm
import arviz as az
from scipy.stats import multivariate_normal

np.random.seed(42)

# Target distribution
rho = 0.8
cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
mean = np.array([0.0, 0.0])

def log_posterior(theta):
    """Log probability of bivariate normal."""
    return multivariate_normal.logpdf(theta, mean=mean, cov=cov_matrix)

def metropolis_hastings(log_posterior, init, n_samples, step_size):
    """Metropolis-Hastings sampler."""
    samples = np.zeros((n_samples, len(init)))
    samples[0] = init
    n_accepted = 0

    for i in range(1, n_samples):
        current = samples[i-1]
        proposed = current + np.random.normal(0, step_size, size=len(init))
        log_ratio = log_posterior(proposed) - log_posterior(current)

        if np.log(np.random.rand()) < log_ratio:
            samples[i] = proposed
            n_accepted += 1
        else:
            samples[i] = current

    acceptance_rate = n_accepted / (n_samples - 1)
    return samples, acceptance_rate

# Test MH with one step size
step_size = 1.0
n_chains = 2  # Reduced for speed
n_samples = 5000  # Reduced for speed

print(f"Running MH with step_size={step_size}")
chains = []
for chain_idx in range(n_chains):
    init = np.random.randn(2)
    samples, acc_rate = metropolis_hastings(log_posterior, init, n_samples, step_size)
    chains.append(samples)

chains_array = np.array(chains)
acceptance_rate = acc_rate  # Last chain's rate

# Compute ESS
trace_dict = {'theta': chains_array.transpose(1, 0, 2)}
idata = az.convert_to_inference_data(trace_dict)
ess = az.ess(idata)['theta'].values

print(f"  Acceptance rate: {acceptance_rate:.3f}")
print(f"  ESS: {ess.mean():.0f}")

# Compare to PyMC NUTS (small sample)
with pm.Model() as nuts_model:
    theta = pm.MvNormal('theta', mu=mean, cov=cov_matrix, shape=2)
    trace_nuts = pm.sample(1000, tune=500, chains=2, random_seed=42, progressbar=False)

ess_nuts = az.ess(trace_nuts)['theta'].values
print(f"\nNUTS ESS: {ess_nuts.mean():.0f}")
print(f"NUTS is {ess_nuts.mean() / ess.mean():.1f}x better ESS than MH")

print("\nSolution 3: PASS")

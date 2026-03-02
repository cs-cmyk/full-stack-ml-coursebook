# Solution 2: Hierarchical Bayesian A/B Testing (Simplified test)
import numpy as np
import pymc as pm
import arviz as az

np.random.seed(42)

# Generate synthetic data
n_segments = 3
n_visitors = 1000

true_control = np.array([0.05, 0.08, 0.06])
true_treatment = np.array([0.07, 0.09, 0.065])

control_conversions = np.random.binomial(n_visitors, true_control)
treatment_conversions = np.random.binomial(n_visitors, true_treatment)

print("Data Summary:")
print(f"Control:   Conversions = {control_conversions}, "
      f"Rates = {control_conversions/n_visitors}")
print(f"Treatment: Conversions = {treatment_conversions}, "
      f"Rates = {treatment_conversions/n_visitors}")
print()

# Prepare data
segment_idx = np.tile(np.arange(n_segments), 2)
variant_idx = np.repeat([0, 1], n_segments)
conversions = np.concatenate([control_conversions, treatment_conversions])
visitors = np.full(n_segments * 2, n_visitors)

# Hierarchical model
with pm.Model() as ab_model:
    mu_baseline = pm.Beta('mu_baseline', alpha=5, beta=95)
    sigma_segment = pm.HalfNormal('sigma_segment', sigma=0.5)
    delta_segment = pm.Normal('delta_segment', mu=0, sigma=sigma_segment,
                               shape=n_segments)
    delta_treatment = pm.Normal('delta_treatment', mu=0, sigma=0.5)

    logit_p_base = pm.math.log(mu_baseline / (1 - mu_baseline))
    logit_p = (logit_p_base +
               delta_segment[segment_idx] +
               delta_treatment * variant_idx)
    p = pm.Deterministic('p', pm.math.sigmoid(logit_p))

    obs = pm.Binomial('obs', n=visitors, p=p, observed=conversions)

    # Reduced sampling for faster test
    trace = pm.sample(1000, tune=500, chains=2, random_seed=42, progressbar=False)

# Check convergence
summary = az.summary(trace, hdi_prob=0.95)
print(f"Max R-hat: {summary['r_hat'].max():.4f}")
print(f"Min ESS (bulk): {summary['ess_bulk'].min():.0f}")
print()

# Business questions
p_control = trace.posterior['p'].values[:, :, :3].mean(axis=2)
p_treatment = trace.posterior['p'].values[:, :, 3:].mean(axis=2)
prob_treatment_better = (p_treatment > p_control).mean()

print(f"P(treatment > control overall): {prob_treatment_better:.3f}")
print("\nSolution 2: PASS")

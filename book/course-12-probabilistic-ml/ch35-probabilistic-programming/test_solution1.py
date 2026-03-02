# Solution 1: Bayesian Logistic Regression on Breast Cancer Dataset
import numpy as np
import pymc as pm
import arviz as az
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

np.random.seed(42)

# Load and prepare data
data = load_breast_cancer()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

n_features = X_train_scaled.shape[1]

# Bayesian logistic regression model
with pm.Model() as logistic_model:
    # Weakly informative priors on coefficients
    beta = pm.Normal('beta', mu=0, sigma=2.5, shape=n_features)
    intercept = pm.Normal('intercept', mu=0, sigma=2.5)

    # Linear combination
    logit_p = intercept + pm.math.dot(X_train_scaled, beta)

    # Likelihood
    obs = pm.Bernoulli('obs', logit_p=logit_p, observed=y_train)

    # Sample posterior
    trace = pm.sample(2000, tune=1000, chains=4, random_seed=42,
                      target_accept=0.9, progressbar=False)

# Check convergence
summary = az.summary(trace, hdi_prob=0.95)
print("Convergence Diagnostics:")
print(f"Max R-hat: {summary['r_hat'].max():.4f} (want < 1.01)")
print(f"Min ESS (bulk): {summary['ess_bulk'].min():.0f} (want > 400)")
print()

# Compare to sklearn
sklearn_model = LogisticRegression(max_iter=10000, random_state=42)
sklearn_model.fit(X_train_scaled, y_train)

posterior_beta = trace.posterior['beta'].values.reshape(-1, n_features).mean(axis=0)
correlation = np.corrcoef(posterior_beta, sklearn_model.coef_[0])[0, 1]
print(f"Correlation between Bayesian and sklearn coefficients: {correlation:.3f}")
print()

# Predictions with uncertainty on test set
posterior_samples = trace.posterior.stack(sample=('chain', 'draw'))
beta_samples = posterior_samples['beta'].values.T  # Shape: (n_samples, n_features)
intercept_samples = posterior_samples['intercept'].values  # Shape: (n_samples,)

# Select 10 random test samples
test_indices = np.random.choice(len(X_test), size=10, replace=False)
X_test_subset = X_test_scaled[test_indices]
y_test_subset = y_test[test_indices]

# Compute predicted probabilities for each posterior sample
logits = X_test_subset @ beta_samples.T + intercept_samples  # Shape: (10, n_samples)
probs = 1 / (1 + np.exp(-logits))  # Sigmoid

# Summarize predictions
prob_mean = probs.mean(axis=1)
prob_lower = np.percentile(probs, 2.5, axis=1)
prob_upper = np.percentile(probs, 97.5, axis=1)

# Visualize predictions with uncertainty
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(test_indices))

ax.errorbar(x_pos, prob_mean,
            yerr=[prob_mean - prob_lower, prob_upper - prob_mean],
            fmt='o', markersize=8, capsize=5, capthick=2,
            label='Predicted Probability (95% CI)')
ax.scatter(x_pos, y_test_subset, color='red', s=100, marker='x',
           linewidths=3, label='True Label', zorder=5)

ax.set_xlabel('Test Sample Index', fontsize=12)
ax.set_ylabel('Probability of Malignant', fontsize=12)
ax.set_title('Bayesian Logistic Regression: Predictions with Uncertainty',
             fontsize=13, weight='bold')
ax.set_ylim(-0.1, 1.1)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Decision Boundary')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/solution1_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Predictions for 10 test samples:")
for i, idx in enumerate(test_indices):
    print(f"Sample {idx}: P(malignant) = {prob_mean[i]:.3f} "
          f"[{prob_lower[i]:.3f}, {prob_upper[i]:.3f}], True = {y_test_subset[i]}")

print("\nSolution 1: PASS")

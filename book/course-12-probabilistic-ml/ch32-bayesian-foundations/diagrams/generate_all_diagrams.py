#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 32: Bayesian Foundations
"""

# All imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom, norm
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

# Set style and random seed for reproducibility
sns.set_style('whitegrid')
np.random.seed(42)

print("Generating diagrams for Chapter 32...")

# =============================================================================
# DIAGRAM 1: Beta-Binomial Inference
# =============================================================================
print("\n1. Generating beta_binomial_inference.png...")

# Simulate A/B test data
n_visitors = 100
true_conversion_rate = 0.25
conversions = np.random.binomial(n=n_visitors, p=true_conversion_rate)

# Prior specification
alpha_prior = 2
beta_prior = 8
prior = beta(alpha_prior, beta_prior)

# Posterior computation
alpha_post = alpha_prior + conversions
beta_post = beta_prior + (n_visitors - conversions)
posterior = beta(alpha_post, beta_post)

# Visualization
p_values = np.linspace(0, 1, 500)
prior_density = prior.pdf(p_values)
likelihood = p_values**conversions * (1 - p_values)**(n_visitors - conversions)
likelihood = likelihood / np.max(likelihood) * np.max(prior_density)
posterior_density = posterior.pdf(p_values)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(p_values, prior_density, 'b-', linewidth=2, label='Prior: Beta(2, 8)', alpha=0.7)
ax.plot(p_values, likelihood, 'g--', linewidth=2, label=f'Likelihood (scaled): Binomial({conversions}/{n_visitors})', alpha=0.7)
ax.plot(p_values, posterior_density, color='#9C27B0', linewidth=3, label=f'Posterior: Beta({alpha_post}, {beta_post})')
ax.axvline(true_conversion_rate, color='red', linestyle=':', linewidth=2, label='True p = 0.25')
ax.axvline(posterior.mean(), color='#9C27B0', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Posterior mean = {posterior.mean():.3f}')
ax.fill_between(p_values, 0, posterior_density,
                where=(p_values >= posterior.ppf(0.025)) & (p_values <= posterior.ppf(0.975)),
                alpha=0.2, color='#9C27B0', label='95% Credible Interval')
ax.set_xlabel('Conversion Rate p', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Bayesian Inference: Prior + Likelihood → Posterior', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-12/ch32/diagrams/beta_binomial_inference.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ beta_binomial_inference.png created")

# =============================================================================
# DIAGRAM 2: Sequential Updating
# =============================================================================
print("\n2. Generating sequential_updating.png...")

batch_sizes = [10, 40, 50, 100]
np.random.seed(42)
all_conversions = np.random.binomial(n=200, p=0.25)

current_alpha = alpha_prior
current_beta = beta_prior

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, n in enumerate(batch_sizes):
    k = int(all_conversions * n / 200)
    current_alpha += k
    current_beta += (n - k)

    p_values = np.linspace(0, 1, 500)
    current_posterior = beta(current_alpha, current_beta)
    density = current_posterior.pdf(p_values)

    axes[idx].plot(p_values, density, color='#9C27B0', linewidth=3)
    axes[idx].fill_between(p_values, 0, density,
                           where=(p_values >= current_posterior.ppf(0.025)) & (p_values <= current_posterior.ppf(0.975)),
                           alpha=0.3, color='#9C27B0')
    axes[idx].axvline(true_conversion_rate, color='red', linestyle=':', linewidth=2, label='True p = 0.25')
    axes[idx].axvline(current_posterior.mean(), color='#9C27B0', linestyle='--', linewidth=1.5, alpha=0.7)

    axes[idx].set_title(f'After {n} visitors: Beta({current_alpha}, {current_beta})', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Conversion Rate p', fontsize=11)
    axes[idx].set_ylabel('Density', fontsize=11)
    axes[idx].text(0.98, 0.95, f'Mean: {current_posterior.mean():.3f}\n95% CI: [{current_posterior.ppf(0.025):.3f}, {current_posterior.ppf(0.975):.3f}]',
                  transform=axes[idx].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[idx].legend(loc='upper left', fontsize=9)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-12/ch32/diagrams/sequential_updating.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ sequential_updating.png created")

# =============================================================================
# DIAGRAM 3: Gaussian-Gaussian Inference
# =============================================================================
print("\n3. Generating gaussian_gaussian_inference.png...")

# Load California Housing dataset
housing = fetch_california_housing()
median_income = housing.data[:, 0]

np.random.seed(42)
sample_small = np.random.choice(median_income, size=50, replace=False)
sample_large = np.random.choice(median_income, size=500, replace=False)

sigma_known = median_income.std()
sigma2_known = sigma_known**2

mu_prior = 3.0
sigma2_prior = 2.0

def gaussian_posterior(sample, mu_prior, sigma2_prior, sigma2_known):
    n = len(sample)
    x_bar = sample.mean()
    tau_prior = 1 / sigma2_prior
    tau_data = 1 / sigma2_known
    tau_post = tau_prior + n * tau_data
    sigma2_post = 1 / tau_post
    mu_post = (tau_prior * mu_prior + n * tau_data * x_bar) / tau_post
    return mu_post, sigma2_post, x_bar

mu_post_small, sigma2_post_small, xbar_small = gaussian_posterior(sample_small, mu_prior, sigma2_prior, sigma2_known)
mu_post_large, sigma2_post_large, xbar_large = gaussian_posterior(sample_large, mu_prior, sigma2_prior, sigma2_known)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (n, sample, mu_post, sigma2_post, xbar) in enumerate([
    (50, sample_small, mu_post_small, sigma2_post_small, xbar_small),
    (500, sample_large, mu_post_large, sigma2_post_large, xbar_large)
]):
    mu_values = np.linspace(1, 5, 500)
    prior_density = norm.pdf(mu_values, mu_prior, np.sqrt(sigma2_prior))
    likelihood_density = norm.pdf(mu_values, xbar, sigma_known/np.sqrt(n))
    posterior_density = norm.pdf(mu_values, mu_post, np.sqrt(sigma2_post))

    axes[idx].plot(mu_values, prior_density, 'b-', linewidth=2, label=f'Prior: N({mu_prior}, {sigma2_prior})', alpha=0.7)
    axes[idx].plot(mu_values, likelihood_density, 'g--', linewidth=2, label=f'Likelihood: N({xbar:.2f}, {(sigma_known/np.sqrt(n)):.3f})', alpha=0.7)
    axes[idx].plot(mu_values, posterior_density, color='#9C27B0', linewidth=3, label=f'Posterior: N({mu_post:.3f}, {sigma2_post:.4f})')
    axes[idx].axvline(mu_prior, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    axes[idx].axvline(xbar, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    axes[idx].axvline(mu_post, color='#9C27B0', linestyle='--', alpha=0.7, linewidth=2)
    axes[idx].fill_between(mu_values, 0, posterior_density,
                           where=(mu_values >= mu_post - 1.96*np.sqrt(sigma2_post)) &
                                 (mu_values <= mu_post + 1.96*np.sqrt(sigma2_post)),
                           alpha=0.2, color='#9C27B0', label='95% Credible Interval')
    axes[idx].set_xlabel('Mean μ (Median Income, $10k)', fontsize=11)
    axes[idx].set_ylabel('Density', fontsize=11)
    axes[idx].set_title(f'n={n}: Posterior Mean = Precision-Weighted Average', fontsize=12, fontweight='bold')
    axes[idx].legend(loc='upper right', fontsize=9)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-12/ch32/diagrams/gaussian_gaussian_inference.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ gaussian_gaussian_inference.png created")

# =============================================================================
# DIAGRAM 4: MAP vs MLE (Ridge Connection)
# =============================================================================
print("\n4. Generating map_vs_mle.png...")

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_with_intercept = np.column_stack([np.ones(len(X_scaled)), X_scaled])

# MLE
theta_mle = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

# MAP for different alphas
alphas = [0.1, 1.0, 10.0, 100.0]
map_solutions = []

for alpha in alphas:
    penalty_matrix = alpha * np.eye(X_with_intercept.shape[1])
    penalty_matrix[0, 0] = 0
    theta_map = np.linalg.solve(X_with_intercept.T @ X_with_intercept + penalty_matrix,
                                 X_with_intercept.T @ y)
    map_solutions.append(theta_map)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Coefficient values
feature_names = ['bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'sex', 'age']
feature_idx = np.arange(1, 11)

axes[0].plot(feature_idx, theta_mle[1:], 'o-', linewidth=2, markersize=8, label='MLE (α=0)', color='black')
for alpha, theta_map in zip(alphas, map_solutions):
    axes[0].plot(feature_idx, theta_map[1:], 'o-', linewidth=2, markersize=6, label=f'MAP (α={alpha})', alpha=0.7)

axes[0].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_xlabel('Feature Index', fontsize=11)
axes[0].set_ylabel('Coefficient Value θⱼ', fontsize=11)
axes[0].set_title('MAP Shrinks Coefficients Toward Zero (Prior Mean)', fontsize=12, fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].set_xticks(feature_idx)
axes[0].set_xticklabels(feature_names, rotation=45, ha='right')

# Right: L2 norm vs alpha
alphas_fine = np.logspace(-2, 2, 50)
norms = []

for alpha in alphas_fine:
    penalty_matrix = alpha * np.eye(X_with_intercept.shape[1])
    penalty_matrix[0, 0] = 0
    theta_map = np.linalg.solve(X_with_intercept.T @ X_with_intercept + penalty_matrix,
                                 X_with_intercept.T @ y)
    norms.append(np.sum(theta_map[1:]**2))

axes[1].plot(alphas_fine, norms, color='#9C27B0', linewidth=3)
axes[1].axhline(np.sum(theta_mle[1:]**2), color='black', linestyle='--', linewidth=2, label='MLE (α=0)', alpha=0.7)
for alpha, theta_map in zip(alphas, map_solutions):
    axes[1].plot(alpha, np.sum(theta_map[1:]**2), 'ro', markersize=10, alpha=0.7)

axes[1].set_xlabel('Prior Strength α (Regularization)', fontsize=11)
axes[1].set_ylabel('||θ||² (L2 Norm Squared)', fontsize=11)
axes[1].set_title('Prior Strength Controls Shrinkage', fontsize=12, fontweight='bold')
axes[1].set_xscale('log')
axes[1].legend(loc='upper right', fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-12/ch32/diagrams/map_vs_mle.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ map_vs_mle.png created")

# =============================================================================
# DIAGRAM 5: Bayesian Regression Uncertainty
# =============================================================================
print("\n5. Generating bayesian_regression_uncertainty.png...")

np.random.seed(42)
n_train = 20
X_train = np.sort(np.random.uniform(0, 10, n_train))
y_true = 2 * X_train + 1
y_train = y_true + np.random.normal(0, 1, n_train)

X_train_design = np.column_stack([np.ones(n_train), X_train])

alpha = 0.01
sigma2_noise = 1.0

Sigma_post = np.linalg.inv(alpha * np.eye(2) + (1/sigma2_noise) * X_train_design.T @ X_train_design)
mu_post = (1/sigma2_noise) * Sigma_post @ X_train_design.T @ y_train

n_samples = 100
theta_samples = np.random.multivariate_normal(mu_post, Sigma_post, size=n_samples)

X_test = np.linspace(-1, 11, 200)
X_test_design = np.column_stack([np.ones(len(X_test)), X_test])

y_pred_mean = X_test_design @ mu_post
y_pred_var = sigma2_noise + np.sum(X_test_design @ Sigma_post * X_test_design, axis=1)
y_pred_std = np.sqrt(y_pred_var)

epistemic_var = np.sum(X_test_design @ Sigma_post * X_test_design, axis=1)
epistemic_std = np.sqrt(epistemic_var)

lr = LinearRegression()
lr.fit(X_train.reshape(-1, 1), y_train)
y_pred_ols = lr.predict(X_test.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(12, 7))

for theta_sample in theta_samples[:100]:
    y_sample = X_test_design @ theta_sample
    ax.plot(X_test, y_sample, 'gray', alpha=0.05, linewidth=1)

ax.scatter(X_train, y_train, color='black', s=50, zorder=5, label='Training data', edgecolors='white', linewidth=1.5)

y_true_line = 2 * X_test + 1
ax.plot(X_test, y_true_line, 'r--', linewidth=2, label='True: y = 2x + 1', alpha=0.7)
ax.plot(X_test, y_pred_mean, color='#9C27B0', linewidth=3, label=f'Posterior mean: y = {mu_post[1]:.2f}x + {mu_post[0]:.2f}')

ax.fill_between(X_test, y_pred_mean - 1.96*epistemic_std, y_pred_mean + 1.96*epistemic_std,
                alpha=0.3, color='blue', label='95% Epistemic uncertainty (parameter)')

ax.fill_between(X_test, y_pred_mean - 1.96*y_pred_std, y_pred_mean + 1.96*y_pred_std,
                alpha=0.2, color='#9C27B0', label='95% Total uncertainty (param + noise)')

ax.plot(X_test, y_pred_ols, color='#FF9800', linewidth=2, linestyle=':', label='OLS (no uncertainty)', alpha=0.8)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Bayesian Linear Regression: Quantifying Prediction Uncertainty', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(-1, 11)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-12/ch32/diagrams/bayesian_regression_uncertainty.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ bayesian_regression_uncertainty.png created")

print("\n" + "="*70)
print("All diagrams generated successfully!")
print("="*70)

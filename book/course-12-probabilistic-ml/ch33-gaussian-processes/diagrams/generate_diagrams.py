import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# ============================================================================
# DIAGRAM 1: GP Prior Samples
# ============================================================================

def rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0):
    """Compute RBF kernel matrix between X1 and X2."""
    dists = cdist(X1, X2, metric='sqeuclidean')
    return variance * np.exp(-dists / (2 * lengthscale**2))

# Create input points
X = np.linspace(0, 10, 200).reshape(-1, 1)

# Compute covariance matrix
K = rbf_kernel(X, X, lengthscale=1.0, variance=1.0)

# Add small jitter for numerical stability
K_stable = K + 1e-6 * np.eye(len(X))

# Sample 8 functions from the GP prior using Cholesky decomposition
L = np.linalg.cholesky(K_stable)
n_samples = 8
f_samples = L @ np.random.randn(len(X), n_samples)

# Plot samples
plt.figure(figsize=(12, 5))
colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red'],
          COLORS['purple'], COLORS['gray'], '#FF5722', '#00BCD4']
for i in range(n_samples):
    plt.plot(X, f_samples[:, i], alpha=0.7, linewidth=1.5, color=colors[i])
plt.xlabel('Input x', fontsize=12)
plt.ylabel('Function value f(x)', fontsize=12)
plt.title('Samples from GP Prior with RBF Kernel (lengthscale=1.0)', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/gp_prior_samples.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created gp_prior_samples.png")

# ============================================================================
# DIAGRAM 2: GP Posterior Update
# ============================================================================

def gp_posterior(X_train, y_train, X_test, kernel_func, noise_var=1e-6):
    """Compute GP posterior mean and covariance."""
    K = kernel_func(X_train, X_train) + noise_var * np.eye(len(X_train))
    K_s = kernel_func(X_train, X_test)
    K_ss = kernel_func(X_test, X_test)

    K_inv = np.linalg.inv(K)
    mu = K_s.T @ K_inv @ y_train
    cov = K_ss - K_s.T @ K_inv @ K_s

    return mu, cov

X_test = np.linspace(0, 10, 200).reshape(-1, 1)
kernel = lambda X1, X2: rbf_kernel(X1, X2, lengthscale=1.0, variance=1.0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

true_func = lambda x: np.sin(x) + 0.5 * np.sin(3 * x)
X_all = np.array([1.0, 3.0, 4.5, 5.0, 6.5, 7.5, 8.0, 9.0]).reshape(-1, 1)
y_all = true_func(X_all).flatten()

n_obs_stages = [0, 1, 3, 8]

for idx, n_obs in enumerate(n_obs_stages):
    ax = axes[idx]

    if n_obs == 0:
        mu = np.zeros(len(X_test))
        cov = kernel(X_test, X_test)
    else:
        X_train = X_all[:n_obs]
        y_train = y_all[:n_obs]
        mu, cov = gp_posterior(X_train, y_train, X_test, kernel, noise_var=0.01)
        mu = mu.flatten()

    std = np.sqrt(np.diag(cov))

    ax.plot(X_test, true_func(X_test).flatten(), 'k--',
            label='True function', alpha=0.4, linewidth=2)
    ax.plot(X_test, mu, color=COLORS['blue'], linewidth=2, label='Posterior mean')
    ax.fill_between(X_test.flatten(), mu - 1.96*std, mu + 1.96*std,
                     alpha=0.3, color=COLORS['blue'], label='95% confidence')

    if n_obs > 0:
        ax.scatter(X_all[:n_obs], y_all[:n_obs], c=COLORS['red'], s=100,
                   zorder=5, label='Observations', edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Input x', fontsize=11)
    ax.set_ylabel('Output f(x)', fontsize=11)
    ax.set_title(f'{"Prior" if n_obs == 0 else f"After {n_obs} observation(s)"}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/gp_posterior_update.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created gp_posterior_update.png")

# ============================================================================
# DIAGRAM 3: Lengthscale Comparison
# ============================================================================

def sample_gp_prior(X, lengthscale, n_samples=5):
    """Sample functions from GP prior with given lengthscale."""
    K = rbf_kernel(X, X, lengthscale=lengthscale, variance=1.0)
    K_stable = K + 1e-8 * np.eye(len(X))
    L = np.linalg.cholesky(K_stable)
    return L @ np.random.randn(len(X), n_samples)

X = np.linspace(0, 10, 300).reshape(-1, 1)
lengthscales = [0.2, 1.0, 3.0]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

np.random.seed(42)
for idx, ℓ in enumerate(lengthscales):
    samples = sample_gp_prior(X, lengthscale=ℓ, n_samples=6)

    colors_cycle = [COLORS['blue'], COLORS['green'], COLORS['orange'],
                    COLORS['red'], COLORS['purple'], COLORS['gray']]
    for i in range(6):
        axes[idx].plot(X, samples[:, i], alpha=0.7, linewidth=1.5, color=colors_cycle[i])

    axes[idx].set_xlabel('Input x', fontsize=12)
    axes[idx].set_ylabel('Function value f(x)', fontsize=12)
    axes[idx].set_title(f'Lengthscale ℓ = {ℓ}', fontsize=13, fontweight='bold')
    axes[idx].grid(alpha=0.3)
    axes[idx].set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/lengthscale_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created lengthscale_comparison.png")

# ============================================================================
# DIAGRAM 4: GP Regression from Scratch
# ============================================================================

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X_full = housing.data
y_full = housing.target

feature_idx = 0
X_full_1d = X_full[:, feature_idx].reshape(-1, 1)

n_train = 200
indices = np.random.permutation(len(X_full_1d))
train_idx = indices[:n_train]

X_train = X_full_1d[train_idx]
y_train = y_full[train_idx]

def gp_regression(X_train, y_train, X_test, lengthscale=1.0,
                  signal_var=1.0, noise_var=0.1):
    """Gaussian Process Regression from scratch."""
    n_train = len(X_train)

    K = rbf_kernel(X_train, X_train, lengthscale, signal_var)
    K_s = rbf_kernel(X_train, X_test, lengthscale, signal_var)
    K_ss = rbf_kernel(X_test, X_test, lengthscale, signal_var)

    K_y = K + noise_var * np.eye(n_train)

    L = np.linalg.cholesky(K_y)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu_pred = K_s.T @ alpha

    v = np.linalg.solve(L, K_s)
    cov_pred = K_ss - v.T @ v
    std_pred = np.sqrt(np.diag(cov_pred))

    return mu_pred, std_pred

X_grid = np.linspace(X_train.min(), X_train.max(), 300).reshape(-1, 1)
mu_grid, std_grid = gp_regression(X_train, y_train, X_grid,
                                  lengthscale=0.5, signal_var=2.0, noise_var=0.5)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.4, s=20, c=COLORS['gray'], label='Training data')
plt.plot(X_grid, mu_grid, color=COLORS['blue'], linewidth=2, label='GP prediction (mean)')
plt.fill_between(X_grid.flatten(), mu_grid - 1.96*std_grid, mu_grid + 1.96*std_grid,
                 alpha=0.3, color=COLORS['blue'], label='95% confidence interval')
plt.xlabel('Median Income (scaled)', fontsize=12)
plt.ylabel('House Price (100k USD)', fontsize=12)
plt.title('GP Regression from Scratch: California Housing', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/gp_regression_from_scratch.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created gp_regression_from_scratch.png")

# ============================================================================
# DIAGRAM 5: GP with scikit-learn
# ============================================================================

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(2.0, (0.1, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.1, 2.0)) + \
         WhiteKernel(noise_level=0.5, noise_level_bounds=(0.01, 2.0))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp.fit(X_train, y_train)

mu_grid_sklearn, std_grid_sklearn = gp.predict(X_grid, return_std=True)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, alpha=0.4, s=20, c=COLORS['gray'], label='Training data')
plt.plot(X_grid, mu_grid_sklearn, color=COLORS['blue'], linewidth=2, label='GP prediction (mean)')
plt.fill_between(X_grid.flatten(),
                 mu_grid_sklearn - 1.96*std_grid_sklearn,
                 mu_grid_sklearn + 1.96*std_grid_sklearn,
                 alpha=0.3, color=COLORS['blue'], label='95% confidence interval')
plt.xlabel('Median Income (scaled)', fontsize=12)
plt.ylabel('House Price (100k USD)', fontsize=12)
plt.title('GP Regression with scikit-learn (Optimized Hyperparameters)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/gp_sklearn.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created gp_sklearn.png")

# ============================================================================
# DIAGRAM 6: Kernel Composition for Time Series
# ============================================================================

from sklearn.gaussian_process.kernels import ExpSineSquared

# Create synthetic seasonal data with trend
X_air = np.arange(144).reshape(-1, 1)
trend = 100 + 2 * X_air.flatten()
seasonality = 20 * np.sin(2 * np.pi * X_air.flatten() / 12)
noise = np.random.randn(144) * 5
y_air = trend + seasonality + noise

n_train = int(0.8 * len(X_air))
X_train_ts = X_air[:n_train]
y_train_ts = y_air[:n_train]
X_test_ts = X_air[n_train:]
y_test_ts = y_air[n_train:]

# Define three different kernels
kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=10.0)
kernel_periodic = ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0,
                                                       periodicity=12.0,
                                                       periodicity_bounds=(11.0, 13.0))
kernel_combined = ConstantKernel(1.0) * RBF(length_scale=50.0) + \
                  ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=12.0) + \
                  WhiteKernel(noise_level=1.0)

kernels = [kernel_rbf, kernel_periodic, kernel_combined]
kernel_names = ['RBF only', 'Periodic only', 'RBF + Periodic']

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for idx, (kernel, name) in enumerate(zip(kernels, kernel_names)):
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   random_state=42, alpha=1e-6)
    gp.fit(X_train_ts, y_train_ts)

    mu_pred, std_pred = gp.predict(X_air, return_std=True)
    mu_test = gp.predict(X_test_ts)
    rmse = np.sqrt(np.mean((y_test_ts - mu_test)**2))
    log_ml = gp.log_marginal_likelihood(gp.kernel_.theta)

    ax = axes[idx]
    ax.plot(X_air, y_air, 'o', color=COLORS['gray'], alpha=0.5, markersize=4, label='Observed data')
    ax.plot(X_air, mu_pred, color=COLORS['blue'], linewidth=2, label='GP mean')
    ax.fill_between(X_air.flatten(), mu_pred - 1.96*std_pred, mu_pred + 1.96*std_pred,
                     alpha=0.3, color=COLORS['blue'], label='95% confidence')
    ax.axvline(x=n_train, color=COLORS['red'], linestyle='--', linewidth=2,
               label='Train/Test split')
    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Passengers', fontsize=11)
    ax.set_title(f'{name} | Test RMSE: {rmse:.2f} | Log-ML: {log_ml:.2f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/kernel_composition_timeseries.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created kernel_composition_timeseries.png")

# ============================================================================
# DIAGRAM 7: Hyperparameter Optimization
# ============================================================================

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()
X_full = diabetes.data
y_full = diabetes.target

feature_idx = 2
X_1d = X_full[:, feature_idx].reshape(-1, 1)

n_samples = 100
indices = np.random.permutation(len(X_1d))[:n_samples]
X_subset = X_1d[indices]
y_subset = y_full[indices]

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X_subset)
y_scaled = scaler_y.fit_transform(y_subset.reshape(-1, 1)).flatten()

lengthscales = np.logspace(-1, 1, 30)
log_marginal_likelihoods = []

for ℓ in lengthscales:
    kernel = ConstantKernel(1.0, constant_value_bounds='fixed') * \
             RBF(length_scale=ℓ, length_scale_bounds='fixed') + \
             WhiteKernel(noise_level=0.1, noise_level_bounds='fixed')

    gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gp.fit(X_scaled, y_scaled)
    log_ml = gp.log_marginal_likelihood(gp.kernel_.theta)
    log_marginal_likelihoods.append(log_ml)

log_marginal_likelihoods = np.array(log_marginal_likelihoods)
optimal_idx = np.argmax(log_marginal_likelihoods)
optimal_lengthscale = lengthscales[optimal_idx]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lengthscales, log_marginal_likelihoods, color=COLORS['blue'], linewidth=2)
plt.axvline(optimal_lengthscale, color=COLORS['red'], linestyle='--', linewidth=2,
            label=f'Optimal ℓ = {optimal_lengthscale:.3f}')
plt.xlabel('Lengthscale ℓ', fontsize=12)
plt.ylabel('Log Marginal Likelihood', fontsize=12)
plt.title('Marginal Likelihood vs. Lengthscale', fontsize=13, fontweight='bold')
plt.xscale('log')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

X_grid = np.linspace(X_scaled.min(), X_scaled.max(), 200).reshape(-1, 1)

kernel_small = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
gp_small = GaussianProcessRegressor(kernel=kernel_small, random_state=42)
gp_small.fit(X_scaled, y_scaled)
mu_small, std_small = gp_small.predict(X_grid, return_std=True)

kernel_opt = ConstantKernel(1.0) * RBF(length_scale=optimal_lengthscale) + WhiteKernel(noise_level=0.1)
gp_opt = GaussianProcessRegressor(kernel=kernel_opt, random_state=42)
gp_opt.fit(X_scaled, y_scaled)
mu_opt, std_opt = gp_opt.predict(X_grid, return_std=True)

plt.subplot(1, 2, 2)
plt.scatter(X_scaled, y_scaled, alpha=0.6, s=40, c=COLORS['gray'], label='Data')
plt.plot(X_grid, mu_small, color=COLORS['red'], linewidth=2, alpha=0.7, label=f'ℓ = 0.1 (suboptimal)')
plt.plot(X_grid, mu_opt, color=COLORS['blue'], linewidth=2, label=f'ℓ = {optimal_lengthscale:.3f} (optimal)')
plt.xlabel('BMI (standardized)', fontsize=12)
plt.ylabel('Disease Progression (standardized)', fontsize=12)
plt.title('Prediction Comparison', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/hyperparameter_optimization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created hyperparameter_optimization.png")

# ============================================================================
# DIAGRAM 8: GP Classification
# ============================================================================

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_full = cancer.data
y_full = cancer.target

feature_indices = [0, 1]
X_2d = X_full[:, feature_indices]

X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y_full, test_size=0.3, random_state=42, stratify=y_full
)

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42, n_restarts_optimizer=5)
gpc.fit(X_train, y_train)

h = 0.2
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = gpc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
contourf = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.7)
plt.colorbar(contourf, label='P(Benign)')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
            edgecolors='black', s=50, alpha=0.6, label='Training')
plt.xlabel('Mean Radius', fontsize=11)
plt.ylabel('Mean Texture', fontsize=11)
plt.title('GP Classification: Decision Boundary', fontsize=13, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
epsilon = 1e-10
entropy = -(Z * np.log(Z + epsilon) + (1 - Z) * np.log(1 - Z + epsilon))
contourf = plt.contourf(xx, yy, entropy, levels=20, cmap='viridis', alpha=0.7)
plt.colorbar(contourf, label='Uncertainty (Entropy)')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu',
            edgecolors='black', s=50, alpha=0.6)
plt.xlabel('Mean Radius', fontsize=11)
plt.ylabel('Mean Texture', fontsize=11)
plt.title('GP Classification: Predictive Uncertainty', fontsize=13, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/gp_classification.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created gp_classification.png")

# ============================================================================
# DIAGRAM 9: Bayesian Optimization
# ============================================================================

def black_box_function(x):
    """Expensive function: (x-3)^2 * sin(5x) + noise"""
    return -((x - 3)**2 * np.sin(5 * x)).flatten()

X_space = np.linspace(0, 10, 1000).reshape(-1, 1)
y_space = black_box_function(X_space)

def expected_improvement(X, X_sample, y_sample, gp, xi=0.01):
    """Compute Expected Improvement at points X."""
    mu, sigma = gp.predict(X, return_std=True)
    mu_sample = gp.predict(X_sample)

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

n_init = 3
np.random.seed(42)
X_sample = np.random.uniform(0, 10, n_init).reshape(-1, 1)
y_sample = black_box_function(X_sample)

n_iterations = 10
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

history = []

for i in range(n_iterations):
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   random_state=42, alpha=1e-6)
    gp.fit(X_sample, y_sample)

    ei = expected_improvement(X_space, X_sample, y_sample, gp, xi=0.01)

    next_idx = np.argmax(ei)
    X_next = X_space[next_idx].reshape(-1, 1)
    y_next = black_box_function(X_next)

    mu, std = gp.predict(X_space, return_std=True)
    history.append({
        'iteration': i,
        'X_sample': X_sample.copy(),
        'y_sample': y_sample.copy(),
        'X_next': X_next,
        'y_next': y_next,
        'mu': mu,
        'std': std,
        'ei': ei
    })

    X_sample = np.vstack((X_sample, X_next))
    y_sample = np.append(y_sample, y_next)

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
axes = axes.flatten()

plot_iterations = [0, 1, 2, 4, 6, 9]

for plot_idx, iter_idx in enumerate(plot_iterations):
    ax = axes[plot_idx]
    h = history[iter_idx]

    ax.plot(X_space, y_space, 'k--', alpha=0.3, linewidth=1.5, label='True function')
    ax.plot(X_space, h['mu'], color=COLORS['blue'], linewidth=2, label='GP mean')
    ax.fill_between(X_space.flatten(),
                     h['mu'] - 1.96*h['std'],
                     h['mu'] + 1.96*h['std'],
                     alpha=0.3, color=COLORS['blue'], label='95% confidence')

    ax.scatter(h['X_sample'], h['y_sample'], c=COLORS['red'], s=100, zorder=5,
               edgecolors='black', linewidth=1.5, label='Samples')
    ax.scatter(h['X_next'], h['y_next'], c=COLORS['green'], s=200, marker='*',
               zorder=5, edgecolors='black', linewidth=2, label='Next sample')

    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.set_title(f'Iteration {h["iteration"]+1} ({len(h["X_sample"])+1} samples)',
                 fontsize=11, fontweight='bold')
    if plot_idx == 0:
        ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim(-50, 15)

# Hide unused subplots
for idx in range(len(plot_iterations), 9):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('book/course-12/ch33/diagrams/bayesian_optimization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Created bayesian_optimization.png")

print("\n✅ All diagrams created successfully!")

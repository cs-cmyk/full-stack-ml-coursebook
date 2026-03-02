"""
Diagram 3: Acquisition Functions Compared
Shows how EI, UCB, and Thompson Sampling explore/exploit differently
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm

np.random.seed(42)

# Define 1D test function
def test_function(x):
    return np.sin(3*x) + 0.3*np.cos(9*x) + 0.5*x

# Domain
x_domain = np.linspace(0, 5, 500).reshape(-1, 1)
y_true = test_function(x_domain)

# Initial observations (5 points)
X_obs = np.array([[0.8], [1.5], [2.5], [3.5], [4.2]])
y_obs = test_function(X_obs).ravel()

# Fit GP
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=42)
gp.fit(X_obs, y_obs)

# Predict
y_mean, y_std = gp.predict(x_domain, return_std=True)

# Acquisition functions
y_best = y_obs.max()

# Expected Improvement
Z = (y_mean - y_best) / (y_std + 1e-9)
ei = (y_mean - y_best) * norm.cdf(Z) + y_std * norm.pdf(Z)
ei[y_std < 1e-9] = 0.0

# Upper Confidence Bound (β=2.0)
beta = 2.0
ucb = y_mean + beta * y_std

# Thompson Sampling (one sample)
ts_sample = gp.sample_y(x_domain, n_samples=1, random_state=42).ravel()

# Next points suggested
next_x_ei = x_domain[np.argmax(ei)]
next_x_ucb = x_domain[np.argmax(ucb)]
next_x_ts = x_domain[np.argmax(ts_sample)]

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(12, 14))

# Subplot 1: GP surrogate
ax0 = axes[0]
ax0.plot(x_domain, y_true, 'k-', linewidth=2, label='True function', alpha=0.4)
ax0.plot(x_domain, y_mean, color='#2196F3', linewidth=2.5, label='GP mean (μ)')
ax0.fill_between(x_domain.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                 alpha=0.3, color='#2196F3', label='95% confidence')
ax0.scatter(X_obs, y_obs, c='#F44336', s=120, zorder=5,
           marker='o', edgecolors='black', linewidths=1.5, label='Observations')
ax0.axhline(y_best, color='#F44336', linestyle='--', linewidth=1.5,
           alpha=0.7, label=f'Best observed = {y_best:.3f}')
ax0.set_ylabel('f(x)', fontsize=12)
ax0.set_title('Gaussian Process Surrogate Model', fontsize=14, fontweight='bold')
ax0.legend(fontsize=10, loc='upper left')
ax0.grid(True, alpha=0.3)
ax0.set_xlim(0, 5)

# Subplot 2: Expected Improvement
ax1 = axes[1]
ax1.plot(x_domain, ei, color='#9C27B0', linewidth=2.5)
ax1.fill_between(x_domain.ravel(), 0, ei, alpha=0.3, color='#9C27B0')
ax1.axvline(next_x_ei, color='#9C27B0', linestyle='--', linewidth=2,
           label=f'Next point: x={next_x_ei[0]:.3f}')
ax1.scatter([next_x_ei], [ei.max()], c='#9C27B0', s=250,
           marker='*', edgecolors='black', linewidths=2, zorder=5)
ax1.set_ylabel('EI(x)', fontsize=12)
ax1.set_title('Expected Improvement: E[max(0, f(x) - f_best)] — Balances mean and uncertainty',
             fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 5)
ax1.set_ylim(bottom=0)

# Subplot 3: Upper Confidence Bound
ax2 = axes[2]
ax2.plot(x_domain, ucb, color='#FF9800', linewidth=2.5, label='UCB = μ + 2σ')
ax2.plot(x_domain, y_mean, color='#2196F3', linewidth=1.5, linestyle=':',
        alpha=0.6, label='μ (mean)')
ax2.plot(x_domain, y_mean + beta*y_std, color='#607D8B', linewidth=1.5,
        linestyle=':', alpha=0.6, label='μ + 2σ (upper bound)')
ax2.axvline(next_x_ucb, color='#FF9800', linestyle='--', linewidth=2,
           label=f'Next point: x={next_x_ucb[0]:.3f}')
ax2.scatter([next_x_ucb], [ucb.max()], c='#FF9800', s=250,
           marker='*', edgecolors='black', linewidths=2, zorder=5)
ax2.set_ylabel('UCB(x)', fontsize=12)
ax2.set_title('Upper Confidence Bound (β=2.0): μ(x) + β·σ(x) — Optimistic estimate',
             fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 5)

# Subplot 4: Thompson Sampling
ax3 = axes[3]
ax3.plot(x_domain, ts_sample, color='#4CAF50', linewidth=2.5, label='Sampled function')
ax3.plot(x_domain, y_mean, color='#2196F3', linewidth=1.5, linestyle=':',
        alpha=0.6, label='GP mean (μ)')
ax3.fill_between(x_domain.ravel(), y_mean - 2*y_std, y_mean + 2*y_std,
                alpha=0.15, color='#2196F3', label='95% confidence')
ax3.axvline(next_x_ts, color='#4CAF50', linestyle='--', linewidth=2,
           label=f'Next point: x={next_x_ts[0]:.3f}')
ax3.scatter([next_x_ts], [ts_sample.max()], c='#4CAF50', s=250,
           marker='*', edgecolors='black', linewidths=2, zorder=5)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('Sampled f(x)', fontsize=12)
ax3.set_title('Thompson Sampling: Sample from posterior, maximize sample — Stochastic exploration',
             fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 5)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-13/ch38/diagrams/acquisition_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: acquisition_comparison.png")

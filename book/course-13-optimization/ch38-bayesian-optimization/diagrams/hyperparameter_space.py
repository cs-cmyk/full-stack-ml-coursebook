"""
Diagram 5: Hyperparameter Space Visualization
Shows how Bayesian Optimization navigates the hyperparameter landscape
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize

np.random.seed(42)

# Simulated hyperparameter optimization landscape
# 2D: learning_rate (log scale) vs max_depth (discrete but visualized as continuous)
def performance_landscape(lr, depth):
    """
    Simulated model performance as a function of learning_rate and max_depth.
    Peak performance around lr=0.1, depth=7-10
    """
    # Convert lr to log scale for smoother function
    log_lr = np.log10(lr)

    # Optimal region: lr around 0.1 (log10(0.1) = -1), depth around 8
    lr_term = -((log_lr + 1)**2) / 0.5
    depth_term = -((depth - 8)**2) / 20

    # Add some complex structure
    interaction = 0.3 * np.sin(log_lr * 3) * np.cos(depth * 0.5)

    # Base performance with noise
    performance = 0.85 + 0.1 * np.exp(lr_term + depth_term) + interaction

    # Clip to [0, 1] range (accuracy)
    return np.clip(performance, 0, 1)

# Create grid for contour plot
lr_range = np.logspace(-3, 0, 100)  # 0.001 to 1.0
depth_range = np.linspace(3, 15, 100)
LR, DEPTH = np.meshgrid(lr_range, depth_range)
PERFORMANCE = performance_landscape(LR, DEPTH)

# Simulate Bayesian Optimization trajectory
# Convert to optimization problem (negate performance)
def objective(x):
    lr, depth = x[0], x[1]
    return -performance_landscape(lr, depth)

# Initial random samples
n_init = 5
lr_init = np.random.uniform(np.log10(0.001), np.log10(1.0), n_init)
depth_init = np.random.uniform(3, 15, n_init)
X_obs_raw = np.column_stack([10**lr_init, depth_init])
y_obs = np.array([performance_landscape(lr, d) for lr, d in X_obs_raw])

# For GP, work in transformed space (log lr, depth)
X_obs = np.column_stack([np.log10(X_obs_raw[:, 0]), X_obs_raw[:, 1]])

# Run Bayesian Optimization
n_iterations = 15
trajectory = [X_obs.copy()]

for iteration in range(n_iterations):
    # Fit GP
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                  n_restarts_optimizer=3, random_state=42)
    gp.fit(X_obs, y_obs)

    # Expected Improvement
    y_best = y_obs.max()

    def neg_ei(x):
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        if sigma < 1e-9:
            return 0
        Z = (mu - y_best) / sigma
        ei = (mu - y_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return -ei[0]

    # Optimize acquisition in transformed space
    best_x = None
    best_val = np.inf
    for _ in range(10):
        x0 = np.array([np.random.uniform(-3, 0), np.random.uniform(3, 15)])
        res = minimize(neg_ei, x0, bounds=[(-3, 0), (3, 15)], method='L-BFGS-B')
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    # Evaluate in original space
    lr_next = 10**best_x[0]
    depth_next = best_x[1]
    y_next = performance_landscape(lr_next, depth_next)

    # Update
    X_obs = np.vstack([X_obs, best_x])
    y_obs = np.append(y_obs, y_next)
    trajectory.append(X_obs.copy())

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Contour with BO trajectory
ax1 = axes[0]
contour = ax1.contourf(LR, DEPTH, PERFORMANCE, levels=20, cmap='viridis', alpha=0.8)
contour_lines = ax1.contour(LR, DEPTH, PERFORMANCE, levels=10, colors='white',
                            linewidths=0.5, alpha=0.4)
ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')

# Plot BO trajectory
X_traj = np.vstack(trajectory)
lr_traj = 10**X_traj[:, 0]
depth_traj = X_traj[:, 1]

# Initial points (red)
ax1.scatter(lr_traj[:n_init], depth_traj[:n_init], c='#F44336', s=200,
           marker='o', edgecolors='black', linewidths=2, zorder=5,
           label='Initial random samples')

# BO-selected points (blue)
ax1.scatter(lr_traj[n_init:], depth_traj[n_init:], c='#2196F3', s=200,
           marker='D', edgecolors='black', linewidths=2, zorder=5,
           label='BO-selected points')

# Trajectory arrows
for i in range(len(lr_traj) - 1):
    ax1.annotate('', xy=(lr_traj[i+1], depth_traj[i+1]),
                xytext=(lr_traj[i], depth_traj[i]),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='white',
                               alpha=0.6), zorder=4)

# Best point found (green star)
best_idx = np.argmax(y_obs)
ax1.scatter([lr_traj[best_idx]], [depth_traj[best_idx]], c='#4CAF50',
           s=400, marker='*', edgecolors='black', linewidths=2, zorder=6,
           label=f'Best found (acc={y_obs[best_idx]:.3f})')

# True optimum region
optimal_lr = 0.1
optimal_depth = 8
ax1.scatter([optimal_lr], [optimal_depth], c='yellow', s=300,
           marker='X', edgecolors='black', linewidths=2, zorder=6,
           label='True optimum region')

ax1.set_xscale('log')
ax1.set_xlabel('Learning Rate (log scale)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Max Depth', fontsize=13, fontweight='bold')
ax1.set_title('Bayesian Optimization Trajectory in Hyperparameter Space',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3, which='both')
cbar1 = plt.colorbar(contour, ax=ax1)
cbar1.set_label('Model Accuracy', fontsize=11)

# Right plot: Convergence over time
ax2 = axes[1]

# Best value at each iteration
best_values = []
for i in range(len(y_obs)):
    best_values.append(y_obs[:i+1].max())

iterations = range(1, len(best_values) + 1)
ax2.plot(iterations, best_values, color='#2196F3', linewidth=3,
        marker='o', markersize=8, label='Best accuracy found')

# Mark the initial random phase
ax2.axvspan(1, n_init, alpha=0.2, color='#F44336', label='Random initialization')

# Mark the BO phase
ax2.axvspan(n_init, len(best_values), alpha=0.2, color='#2196F3',
           label='Bayesian Optimization')

# True optimum line
true_optimum = performance_landscape(optimal_lr, optimal_depth)
ax2.axhline(true_optimum, color='#4CAF50', linestyle='--', linewidth=2,
           alpha=0.8, label=f'True optimum ≈ {true_optimum:.3f}')

# Annotations
# Rapid improvement phase
improvement_start = n_init + 1
improvement_end = min(n_init + 5, len(best_values))
if improvement_end > improvement_start:
    ax2.annotate('Rapid improvement\nthrough sequential\noptimization',
                xy=(improvement_end, best_values[improvement_end-1]),
                xytext=(improvement_end + 3, best_values[improvement_end-1] - 0.03),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#2196F3', alpha=0.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#2196F3'))

ax2.set_xlabel('Iteration (Function Evaluations)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Best Accuracy Found', fontsize=13, fontweight='bold')
ax2.set_title('Convergence: Sequential Learning Improves Rapidly',
             fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, len(best_values))
ax2.set_ylim(0.75, 1.0)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-13/ch38/diagrams/hyperparameter_space.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: hyperparameter_space.png")

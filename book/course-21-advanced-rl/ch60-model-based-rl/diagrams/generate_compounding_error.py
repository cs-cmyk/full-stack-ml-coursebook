"""
Generate compounding error visualization for Model-Based RL chapter
"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Color palette from instructions
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Prediction horizons
horizons = np.arange(1, 21)

# Simulate compounding errors
# Single model: exponential growth
# Starting with 1-step error of 0.01
base_error = 0.01
single_model_errors = base_error * (1.5 ** (horizons - 1))

# Ensemble: similar trend but with uncertainty bands
ensemble_mean = single_model_errors * 0.95  # Slightly better
ensemble_std = 0.15 * ensemble_mean * horizons / 10  # Growing uncertainty

# Plot single model
ax.plot(horizons, single_model_errors,
        color=colors['blue'], linewidth=2.5,
        marker='o', markersize=6, label='Single Model')

# Plot ensemble with uncertainty bands
ax.plot(horizons, ensemble_mean,
        color=colors['green'], linewidth=2.5,
        marker='s', markersize=6, label='Ensemble Mean')
ax.fill_between(horizons,
                ensemble_mean - ensemble_std,
                ensemble_mean + ensemble_std,
                color=colors['green'], alpha=0.2,
                label='Ensemble Uncertainty')

# Add horizontal line at error threshold
ax.axhline(y=0.5, color=colors['red'], linestyle='--',
           linewidth=1.5, alpha=0.7, label='High Error Threshold')

# Shade the preferred planning region (5-10 steps)
ax.axvspan(5, 10, alpha=0.1, color=colors['orange'],
           label='Preferred Planning Horizon')

# Annotations
ax.annotate('1-step error ≈ 0.01',
            xy=(1, single_model_errors[0]),
            xytext=(3, 0.02),
            fontsize=11, color=colors['gray'],
            arrowprops=dict(arrowstyle='->', color=colors['gray'], lw=1.5))

ax.annotate('Error compounds\nexponentially',
            xy=(15, single_model_errors[14]),
            xytext=(11, 1.5),
            fontsize=11, color=colors['gray'],
            arrowprops=dict(arrowstyle='->', color=colors['gray'], lw=1.5))

# Styling
ax.set_xlabel('Prediction Horizon (steps)', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
ax.set_title('Compounding Errors in Multi-Step Predictions',
             fontsize=16, fontweight='bold', pad=20)

# Set y-axis to log scale
ax.set_yscale('log')

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

# Set axis limits
ax.set_xlim(0.5, 20.5)
ax.set_ylim(0.005, 5)

# Tick formatting
ax.tick_params(axis='both', labelsize=12)

# Tight layout
plt.tight_layout()

# Save figure
output_path = '/home/chirag/ds-book/book/course-21/ch60/diagrams/compounding_error.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved diagram to {output_path}")

plt.close()

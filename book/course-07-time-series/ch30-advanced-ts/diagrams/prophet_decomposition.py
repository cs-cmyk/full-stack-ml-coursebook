"""
Prophet model decomposition visualization
Shows the additive components: trend + seasonality + holidays + noise
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create synthetic data showing Prophet components
np.random.seed(42)
t = np.linspace(0, 4, 400)

# Components
trend = 20 + 15 * t  # Linear growth
yearly_seasonality = 8 * np.sin(2 * np.pi * t)
holidays = np.zeros_like(t)
holidays[(t % 1 > 0.95) | (t % 1 < 0.05)] = 10  # Holiday spikes
noise = np.random.normal(0, 2, len(t))

# Combined series
y = trend + yearly_seasonality + holidays + noise

# Create figure with subplots
fig, axes = plt.subplots(5, 1, figsize=(14, 10))
fig.suptitle('Prophet Additive Decomposition Model', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Observed data
axes[0].plot(t, y, color='#607D8B', linewidth=1.5, alpha=0.8)
axes[0].set_ylabel('Observed\n(y)', fontsize=12, fontweight='bold')
axes[0].set_title('y = g(t) + s(t) + h(t) + ε', fontsize=13, style='italic', loc='right')
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 4)

# Plot 2: Trend component
axes[1].plot(t, trend, color='#2196F3', linewidth=2.5)
axes[1].set_ylabel('Trend\ng(t)', fontsize=12, fontweight='bold')
axes[1].set_title('Piecewise linear or logistic growth', fontsize=11, loc='right', style='italic', color='#555')
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 4)

# Plot 3: Seasonal component
axes[2].plot(t, yearly_seasonality, color='#4CAF50', linewidth=2.5)
axes[2].set_ylabel('Seasonality\ns(t)', fontsize=12, fontweight='bold')
axes[2].set_title('Fourier series for periodic patterns', fontsize=11, loc='right', style='italic', color='#555')
axes[2].grid(alpha=0.3)
axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
axes[2].set_xlim(0, 4)

# Plot 4: Holiday component
axes[3].plot(t, holidays, color='#F44336', linewidth=2.5)
axes[3].set_ylabel('Holidays\nh(t)', fontsize=12, fontweight='bold')
axes[3].set_title('Special events and anomalies', fontsize=11, loc='right', style='italic', color='#555')
axes[3].grid(alpha=0.3)
axes[3].set_xlim(0, 4)

# Plot 5: Noise component
axes[4].scatter(t, noise, color='#9C27B0', alpha=0.5, s=10)
axes[4].set_ylabel('Noise\nε', fontsize=12, fontweight='bold')
axes[4].set_xlabel('Time', fontsize=13, fontweight='bold')
axes[4].set_title('Random fluctuations ~ N(0, σ²)', fontsize=11, loc='right', style='italic', color='#555')
axes[4].grid(alpha=0.3)
axes[4].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
axes[4].set_xlim(0, 4)

# Add equation annotation
equation_text = (
    "Prophet Model:\n"
    "• Interpretable components\n"
    "• Handles missing data\n"
    "• Multiple seasonalities\n"
    "• Custom holiday effects"
)
fig.text(0.98, 0.5, equation_text, fontsize=11, va='center', ha='right',
         bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8, edgecolor='#CCCCCC'),
         family='monospace')

plt.tight_layout()
plt.subplots_adjust(right=0.88)
plt.savefig('/home/chirag/ds-book/book/course-07-time-series/ch30-advanced-ts/diagrams/prophet_decomposition.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Saved: prophet_decomposition.png")
plt.close()

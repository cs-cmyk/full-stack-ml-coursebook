import matplotlib.pyplot as plt
import numpy as np

# Create interpretability spectrum visualization
fig, ax = plt.subplots(figsize=(12, 4))

# Define model categories and positions
models = [
    'Linear\nRegression',
    'Decision\nTree\n(depth≤5)',
    'Rule\nLists',
    'GAMs',
    'Shallow\nEnsembles',
    'Deep\nDecision Tree',
    'Random\nForest',
    'XGBoost',
    'Deep\nNeural\nNetwork',
    'Large\nTransformer'
]
positions = np.linspace(0, 10, len(models))
interpretability = [9.5, 8.5, 9.0, 7.5, 6.0, 5.0, 4.0, 3.5, 2.0, 1.0]

# Color gradient from green (interpretable) to red (black-box)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(models)))

# Plot bars
bars = ax.barh(range(len(models)), interpretability, color=colors, alpha=0.8, edgecolor='black')

# Add model names
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=10)
ax.set_xlabel('Interpretability Score', fontsize=12, fontweight='bold')
ax.set_title('The Interpretability Spectrum', fontsize=14, fontweight='bold')
ax.set_xlim(0, 10)

# Add vertical regions
ax.axvspan(7, 10, alpha=0.1, color='green', label='Glass-Box Models')
ax.axvspan(4, 7, alpha=0.1, color='yellow', label='Moderately Interpretable')
ax.axvspan(0, 4, alpha=0.1, color='red', label='Black-Box Models')

# Add annotation
ax.text(2, -1.5, 'Explainability methods (SHAP, LIME, PDPs) bridge the gap →',
        fontsize=11, style='italic', ha='left', color='navy')

ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('interpretability_spectrum.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: interpretability_spectrum.png")

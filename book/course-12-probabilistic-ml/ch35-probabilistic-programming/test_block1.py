import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Create workflow diagram
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_prior = '#E8F4F8'
color_data = '#FFE8D6'
color_inference = '#E8F5E9'
color_posterior = '#F3E5F5'
color_check = '#FFF9C4'

# Box coordinates
boxes = [
    (1, 8, 3, 1.5, 'Prior Distribution\nP(θ)', color_prior),
    (6, 8, 3, 1.5, 'Likelihood\nP(y|θ)', color_data),
    (3.5, 5.5, 3, 1.5, 'Inference Algorithm\n(MCMC / VI)', color_inference),
    (3.5, 3, 3, 1.5, 'Posterior\nP(θ|y)', color_posterior),
    (1, 0.5, 3, 1.2, 'Prior Predictive\nCheck', color_check),
    (6, 0.5, 3, 1.2, 'Posterior Predictive\nCheck', color_check),
]

for x, y, w, h, label, color in boxes:
    fancy_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(fancy_box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=11, weight='bold', multialignment='center')

# Add observed data box
data_box = FancyBboxPatch((3.5, 7.2), 3, 1, boxstyle="round,pad=0.05",
                          edgecolor='red', facecolor='white', linewidth=2,
                          linestyle='--')
ax.add_patch(data_box)
ax.text(5, 7.7, 'Observed Data: y', ha='center', va='center',
        fontsize=10, weight='bold', color='red')

# Add arrows
arrow_props = dict(arrowstyle='->', lw=2, color='black')
arrows = [
    (2.5, 8, 4, 6.5),  # Prior to Inference
    (7.5, 8, 6, 6.5),  # Likelihood to Inference
    (5, 7.2, 5, 7),    # Data to Likelihood
    (5, 5.5, 5, 4.5),  # Inference to Posterior
    (2.5, 1.1, 3.5, 3), # Prior check to Posterior
    (6.5, 1.1, 6.5, 3), # Posterior check to Posterior
]

for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props)

# Add iteration arrow
iteration_arrow = mpatches.FancyArrowPatch((6.5, 3.75), (6.5, 5.5),
                                          connectionstyle="arc3,rad=.5",
                                          arrowstyle='->', lw=2, color='purple')
ax.add_patch(iteration_arrow)
ax.text(7.5, 4.5, 'Iterate if\nneeded', ha='center', va='center',
        fontsize=9, style='italic', color='purple')

# Add title
ax.text(5, 9.5, 'Probabilistic Programming Workflow', ha='center',
        fontsize=14, weight='bold')

# Add code annotations
ax.text(2.5, 8.8, 'pm.Normal()', ha='center', fontsize=8,
        style='italic', color='blue')
ax.text(7.5, 8.8, 'observed=data', ha='center', fontsize=8,
        style='italic', color='blue')
ax.text(5, 6.3, 'pm.sample()', ha='center', fontsize=8,
        style='italic', color='blue')

plt.tight_layout()
plt.savefig('diagrams/probabilistic_programming_workflow.png', dpi=300, bbox_inches='tight')
plt.close()

print("Workflow diagram saved to diagrams/probabilistic_programming_workflow.png")

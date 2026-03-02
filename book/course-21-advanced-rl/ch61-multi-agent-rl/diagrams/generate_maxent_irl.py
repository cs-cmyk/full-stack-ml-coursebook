"""
Generate Maximum Entropy IRL training loss visualization
"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate IRL learning curve
n_iterations = 100

# Simulate gradient descent convergence
losses = []
for i in range(n_iterations):
    # Loss starts high and decreases with oscillations
    base_loss = 2.0 * np.exp(-i / 20) + 0.1
    noise = np.random.normal(0, 0.05 * np.exp(-i / 30))
    loss = max(0.05, base_loss + noise)
    losses.append(loss)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(losses, color='#9C27B0', linewidth=2, alpha=0.7)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss (Feature Expectation Mismatch)', fontsize=12)
plt.title('Maximum Entropy IRL Training', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(bottom=0)

# Add annotation for convergence
converged_idx = 70
plt.axvline(x=converged_idx, color='#F44336', linestyle='--', alpha=0.5, linewidth=1.5)
plt.text(converged_idx + 2, 1.5, 'Convergence', fontsize=10, color='#F44336')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-21/ch61/diagrams/maxent_irl_loss.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved MaxEnt IRL loss plot to diagrams/maxent_irl_loss.png")
plt.close()

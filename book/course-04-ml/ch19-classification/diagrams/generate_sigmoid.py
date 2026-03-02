import numpy as np
import matplotlib.pyplot as plt

# Create z values from -6 to 6
z = np.linspace(-6, 6, 200)

# Apply sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sigma_z = sigmoid(z)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(z, sigma_z, linewidth=2.5, color='#2196F3', label='σ(z) = 1/(1+e^(-z))')

# Add decision threshold line
plt.axhline(y=0.5, color='#F44336', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision threshold (0.5)')
plt.axvline(x=0, color='#607D8B', linestyle=':', linewidth=1, alpha=0.5)

# Shade regions
plt.fill_between(z, 0, sigma_z, where=(sigma_z < 0.5), alpha=0.2, color='#2196F3', label='Predict Class 0')
plt.fill_between(z, sigma_z, 1, where=(sigma_z >= 0.5), alpha=0.2, color='#FF9800', label='Predict Class 1')

# Annotations
plt.annotate('z = θ₀ + θ₁x₁ + θ₂x₂ + ...', xy=(2.5, 0.15), fontsize=12,
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
plt.annotate('Output interpreted\nas probability', xy=(-3, 0.85), fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Labels and formatting
plt.xlabel('z (linear combination of features)', fontsize=13)
plt.ylabel('σ(z) - Probability', fontsize=13)
plt.title('The Sigmoid Function: Transforming Any Number Into a Probability', fontsize=14, weight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='center right', fontsize=11)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

# Save the plot
plt.savefig('/home/chirag/ds-book/book/course-04-ml/ch19-classification/diagrams/sigmoid_function.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: sigmoid_function.png")

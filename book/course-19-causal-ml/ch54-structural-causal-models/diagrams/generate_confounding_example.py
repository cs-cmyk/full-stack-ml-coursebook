#!/usr/bin/env python3
"""Generate confounding example diagram for treatment effect."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000

# True causal structure:
# Age → Treatment (older patients get treatment more often)
# Age → Recovery (older patients recover slower)
# Treatment → Recovery (treatment helps recovery)

# Generate data
age = np.random.uniform(20, 80, size=n)

# Treatment assignment depends on age (confounding!)
# Doctors preferentially treat older patients
treatment_prob = 0.2 + 0.6 * (age - 20) / 60  # Increases with age
treatment = np.random.binomial(1, treatment_prob)

# Recovery depends on both age and treatment
# True causal effect of treatment: +15 points
# Age effect: -0.5 points per year
recovery = 100 - 0.5 * age + 15 * treatment + np.random.normal(0, 5, n)

df_medical = pd.DataFrame({
    'Age': age,
    'Treatment': treatment,
    'Recovery': recovery
})

# Visualize the confounding
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Recovery by treatment (ignoring age) - misleading
treated = df_medical[df_medical['Treatment'] == 1]
untreated = df_medical[df_medical['Treatment'] == 0]

axes[0].scatter(untreated['Age'], untreated['Recovery'],
                alpha=0.5, s=20, color='#2196F3', label='No Treatment')
axes[0].scatter(treated['Age'], treated['Recovery'],
                alpha=0.5, s=20, color='#F44336', label='Treatment')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Recovery Score', fontsize=12)
axes[0].set_title('Raw Data: Treatment appears harmful\n(confounded by age)',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: Age-adjusted comparison
# Plot residuals after removing age effect
model_age = LinearRegression()
model_age.fit(df_medical[['Age']], df_medical['Recovery'])
recovery_residuals = df_medical['Recovery'] - model_age.predict(df_medical[['Age']])

treated_mask = df_medical['Treatment'] == 1
axes[1].hist(recovery_residuals[~treated_mask], bins=30, alpha=0.6,
             color='#2196F3', label='No Treatment', density=True)
axes[1].hist(recovery_residuals[treated_mask], bins=30, alpha=0.6,
             color='#F44336', label='Treatment', density=True)
axes[1].set_xlabel('Recovery Score (age-adjusted)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Age-Adjusted: Treatment is beneficial\n(confounder controlled)',
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-19/ch54/diagrams/confounding_example.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated confounding_example.png")

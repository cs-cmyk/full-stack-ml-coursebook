#!/usr/bin/env python3
"""Generate collider bias sports analytics diagram."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 10000

# Generate population data: Height and Speed are INDEPENDENT
height = np.random.normal(loc=180, scale=10, size=n)  # cm
speed = np.random.normal(loc=100, scale=15, size=n)   # arbitrary units

# Professional selection: requires high combined ability
# Pro status is a collider: Height → Pro ← Speed
ability_threshold = 265  # Only top ~15% become professional
combined_ability = height + speed
is_pro = (combined_ability > ability_threshold).astype(int)

# Create DataFrame
df_sports = pd.DataFrame({
    'Height': height,
    'Speed': speed,
    'Combined_Ability': combined_ability,
    'Professional': is_pro
})

# Correlation in general population (unconditional)
corr_general = df_sports['Height'].corr(df_sports['Speed'])

# Correlation among professionals (conditioning on Pro=1)
df_pros = df_sports[df_sports['Professional'] == 1]
corr_pros = df_pros['Height'].corr(df_pros['Speed'])

# Visualize the paradox
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: General population
axes[0].scatter(df_sports['Height'], df_sports['Speed'],
                alpha=0.3, s=10, color='#607D8B')
axes[0].set_xlabel('Height (cm)', fontsize=12)
axes[0].set_ylabel('Speed (units)', fontsize=12)
axes[0].set_title(f'General Population\nCorr = {corr_general:.3f}',
                  fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Right: Professionals only
axes[1].scatter(df_pros['Height'], df_pros['Speed'],
                alpha=0.6, s=30, color='#F44336')
axes[1].set_xlabel('Height (cm)', fontsize=12)
axes[1].set_ylabel('Speed (units)', fontsize=12)
axes[1].set_title(f'Professional Athletes Only\nCorr = {corr_pros:.3f}',
                  fontsize=14, fontweight='bold', color='#F44336')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Collider Bias: Height and Speed in Sports',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-19/ch54/diagrams/collider_sports.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated collider_sports.png")

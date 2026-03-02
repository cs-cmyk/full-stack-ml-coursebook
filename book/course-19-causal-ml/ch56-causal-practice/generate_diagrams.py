#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 56: Causal ML in Practice
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Create diagrams directory if it doesn't exist
os.makedirs('diagrams', exist_ok=True)

print("=" * 60)
print("Generating diagrams for Chapter 56: Causal ML in Practice")
print("=" * 60)

# ============================================================================
# DIAGRAM 1: Four Quadrants for Uplift Modeling
# ============================================================================
print("\n[1/2] Generating uplift_four_quadrants.png...")

fig, ax = plt.subplots(figsize=(10, 8))

# Define quadrants
quadrants = {
    'Persuadables': {'pos': (0, 0.5), 'color': '#2ecc71', 'label': 'TARGET\nTHIS GROUP'},
    'Sure Things': {'pos': (0.5, 0.5), 'color': '#f39c12', 'label': 'WASTE\nBUDGET'},
    'Lost Causes': {'pos': (0, 0), 'color': '#95a5a6', 'label': 'WASTE\nBUDGET'},
    'Sleeping Dogs': {'pos': (0.5, 0), 'color': '#e74c3c', 'label': 'AVOID!\nNEGATIVE EFFECT'}
}

# Draw quadrants
for name, props in quadrants.items():
    rect = Rectangle(props['pos'], 0.5, 0.5,
                     facecolor=props['color'], alpha=0.3, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Add labels
    center_x = props['pos'][0] + 0.25
    center_y = props['pos'][1] + 0.25
    ax.text(center_x, center_y + 0.1, name,
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(center_x, center_y - 0.1, props['label'],
            ha='center', va='center', fontsize=10, style='italic')

# Add axes labels
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('Response under Control (No Treatment)', fontsize=12, fontweight='bold')
ax.set_ylabel('Response under Treatment', fontsize=12, fontweight='bold')
ax.set_title('Four Customer Segments in Uplift Modeling', fontsize=14, fontweight='bold', pad=20)

# Add diagonal line (no effect)
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='No Treatment Effect')

# Add annotations
ax.annotate('Positive Uplift\n(treatment > control)', xy=(0.25, 0.75),
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.annotate('Negative Uplift\n(treatment < control)', xy=(0.75, 0.25),
            fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('diagrams/uplift_four_quadrants.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: diagrams/uplift_four_quadrants.png")

# ============================================================================
# DIAGRAM 2: Uplift Curve
# ============================================================================
print("\n[2/2] Generating uplift_curve.png...")

# Set random seed for reproducibility
np.random.seed(42)

# Simulate marketing campaign data with heterogeneous treatment effects
n_samples = 10000

# Generate customer features
age = np.random.normal(40, 15, n_samples)
income = np.random.normal(60000, 20000, n_samples)
previous_purchases = np.random.poisson(3, n_samples)
engagement_score = np.random.uniform(0, 1, n_samples)

X = np.column_stack([age, income, previous_purchases, engagement_score])

# Randomized treatment assignment (50/50 split)
treatment = np.random.binomial(1, 0.5, n_samples)

# Define heterogeneous treatment effects based on features
base_response = 0.1 + 0.3 * (previous_purchases > 2)

# Treatment effect varies by segment
treatment_effect = np.where(
    (engagement_score > 0.6) & (previous_purchases <= 2),  # Persuadables
    0.3,  # Positive effect
    np.where(
        engagement_score < 0.3,  # Sleeping dogs
        -0.15,  # Negative effect
        0.05  # Minimal effect for others
    )
)

# Generate outcomes
prob_purchase = np.clip(base_response + treatment * treatment_effect, 0, 1)
outcome = np.random.binomial(1, prob_purchase)

# Split data
indices = np.arange(n_samples)
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = outcome[train_idx], outcome[test_idx]
t_train, t_test = treatment[train_idx], treatment[test_idx]

# T-Learner: Train separate models for treatment and control
X_control = X_train[t_train == 0]
y_control = y_train[t_train == 0]
model_control = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_control.fit(X_control, y_control)

X_treatment = X_train[t_train == 1]
y_treatment = y_train[t_train == 1]
model_treatment = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_treatment.fit(X_treatment, y_treatment)

# Predict and compute uplift scores
pred_treatment = model_treatment.predict_proba(X_test)[:, 1]
pred_control = model_control.predict_proba(X_test)[:, 1]
uplift_scores = pred_treatment - pred_control

# Compute uplift curve
def compute_uplift_curve(uplift_scores, treatment, outcome):
    """Compute uplift curve by ranking individuals by predicted uplift."""
    sorted_idx = np.argsort(-uplift_scores)
    treatment_sorted = treatment[sorted_idx]
    outcome_sorted = outcome[sorted_idx]

    percentiles = []
    uplift_values = []

    for i in range(10, 101, 10):
        n_target = int(len(sorted_idx) * i / 100)

        treated_outcomes = outcome_sorted[:n_target][treatment_sorted[:n_target] == 1]
        control_outcomes = outcome_sorted[:n_target][treatment_sorted[:n_target] == 0]

        if len(treated_outcomes) > 0 and len(control_outcomes) > 0:
            treatment_rate = treated_outcomes.sum() / len(treated_outcomes)
            control_rate = control_outcomes.sum() / len(control_outcomes)
            uplift = (treatment_rate - control_rate) * n_target
        else:
            uplift = 0

        percentiles.append(i)
        uplift_values.append(uplift)

    return percentiles, uplift_values

# Compute curves
percentiles, uplift_curve = compute_uplift_curve(uplift_scores, t_test, y_test)

np.random.seed(42)
random_scores = np.random.uniform(0, 1, len(uplift_scores))
_, random_curve = compute_uplift_curve(random_scores, t_test, y_test)

# Plot uplift curves
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(percentiles, uplift_curve, 'o-', linewidth=2, markersize=8,
        label='T-Learner Model', color='#2ecc71')
ax.plot(percentiles, random_curve, 's--', linewidth=2, markersize=6,
        label='Random Targeting', color='#95a5a6')

# Add zero line
ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

# Annotations
ax.set_xlabel('Percentage of Population Targeted', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Incremental Response', fontsize=12, fontweight='bold')
ax.set_title('Uplift Curve: T-Learner vs. Random Targeting', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)

# Add optimal targeting annotation
optimal_idx = np.argmax(np.array(uplift_curve) / np.array(percentiles))
ax.annotate(f'Optimal: Target top {percentiles[optimal_idx]}%\n'
            f'Gain: {uplift_curve[optimal_idx]:.0f} incremental conversions',
            xy=(percentiles[optimal_idx], uplift_curve[optimal_idx]),
            xytext=(percentiles[optimal_idx] + 15, uplift_curve[optimal_idx] - 20),
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

plt.tight_layout()
plt.savefig('diagrams/uplift_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ Saved: diagrams/uplift_curve.png")

# Summary
print("\n" + "=" * 60)
print("All diagrams generated successfully!")
print("=" * 60)
print(f"\nGenerated files:")
print("  1. diagrams/uplift_four_quadrants.png")
print("  2. diagrams/uplift_curve.png")
print("\nAll diagrams saved at 150 DPI with proper styling.")

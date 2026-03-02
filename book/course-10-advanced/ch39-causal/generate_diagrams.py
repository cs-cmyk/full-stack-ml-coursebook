#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 39: Causal Inference & Experimentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.multitest import multipletests

# Set style
plt.style.use('default')
np.random.seed(42)

print("Generating diagrams for Chapter 39: Causal Inference & Experimentation")
print("=" * 70)

# ============================================================================
# Diagram 1: Causal Structure Gallery
# ============================================================================
print("\n1. Generating causal_structures.png...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Fundamental Causal Structures', fontsize=16, fontweight='bold')

def draw_node(ax, x, y, label, color='lightblue'):
    """Draw a circular node."""
    circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold', zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color='black', linewidth=2):
    """Draw a directed arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25,
                           linewidth=linewidth, color=color, zorder=2)
    ax.add_patch(arrow)

# 1. Confounder (Fork)
ax = axes[0, 0]
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.axis('off')
ax.set_title('Confounder (Fork)\nZ → X, Z → Y', fontsize=12, fontweight='bold')

draw_node(ax, 1, 2, 'Z', color='#ff9999')  # Confounder in red
draw_node(ax, 0.5, 0.5, 'X', color='lightblue')
draw_node(ax, 1.5, 0.5, 'Y', color='lightgreen')
draw_arrow(ax, 0.9, 1.85, 0.6, 0.65)
draw_arrow(ax, 1.1, 1.85, 1.4, 0.65)

ax.text(1, 0, 'Control Z to block\nbackdoor path', ha='center', fontsize=10, style='italic')

# 2. Mediator (Chain)
ax = axes[0, 1]
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.axis('off')
ax.set_title('Mediator (Chain)\nX → M → Y', fontsize=12, fontweight='bold')

draw_node(ax, 0.5, 1.25, 'X', color='lightblue')
draw_node(ax, 1.25, 1.25, 'M', color='#ffff99')  # Mediator in yellow
draw_node(ax, 2, 1.25, 'Y', color='lightgreen')
draw_arrow(ax, 0.65, 1.25, 1.1, 1.25)
draw_arrow(ax, 1.4, 1.25, 1.85, 1.25)

ax.text(1.25, 0, 'Do NOT control M\nfor total effect', ha='center', fontsize=10, style='italic')

# 3. Collider (Inverted Fork)
ax = axes[0, 2]
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.axis('off')
ax.set_title('Collider (Inverted Fork)\nX → C ← Y', fontsize=12, fontweight='bold')

draw_node(ax, 0.5, 2, 'X', color='lightblue')
draw_node(ax, 1.5, 2, 'Y', color='lightgreen')
draw_node(ax, 1, 0.75, 'C', color='#ff99ff')  # Collider in purple
draw_arrow(ax, 0.6, 1.85, 0.9, 0.9)
draw_arrow(ax, 1.4, 1.85, 1.1, 0.9)

ax.text(1, 0, 'Do NOT control C\n(creates bias!)', ha='center', fontsize=10, style='italic', color='red')

# 4. Direct Effect
ax = axes[1, 0]
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.axis('off')
ax.set_title('Direct Effect\nX → Y', fontsize=12, fontweight='bold')

draw_node(ax, 0.75, 1.25, 'X', color='lightblue')
draw_node(ax, 1.75, 1.25, 'Y', color='lightgreen')
draw_arrow(ax, 0.9, 1.25, 1.6, 1.25, color='green', linewidth=3)

ax.text(1.25, 0, 'No confounding\n(randomized)', ha='center', fontsize=10, style='italic')

# 5. Complete Confounding Example
ax = axes[1, 1]
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.axis('off')
ax.set_title('Confounding + Direct Effect\nZ → X → Y ← Z', fontsize=12, fontweight='bold')

draw_node(ax, 1, 2, 'Z', color='#ff9999')
draw_node(ax, 0.5, 0.5, 'X', color='lightblue')
draw_node(ax, 1.5, 0.5, 'Y', color='lightgreen')
draw_arrow(ax, 0.9, 1.85, 0.6, 0.65)  # Z -> X
draw_arrow(ax, 1.1, 1.85, 1.4, 0.65)  # Z -> Y (confounding)
draw_arrow(ax, 0.65, 0.5, 1.35, 0.5, color='green', linewidth=2)  # X -> Y (causal)

ax.text(1, 0, 'Control Z to get\ncausal X→Y effect', ha='center', fontsize=10, style='italic')

# 6. Multiple Paths
ax = axes[1, 2]
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.axis('off')
ax.set_title('Backdoor Path\nX ← Z → Y', fontsize=12, fontweight='bold')

draw_node(ax, 0.5, 1.25, 'X', color='lightblue')
draw_node(ax, 1.25, 2, 'Z', color='#ff9999')
draw_node(ax, 2, 1.25, 'Y', color='lightgreen')
draw_arrow(ax, 1.15, 1.85, 0.65, 1.4)
draw_arrow(ax, 1.35, 1.85, 1.85, 1.4)

ax.text(1.25, 0, 'Backdoor path creates\nspurious correlation', ha='center', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('diagrams/causal_structures.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved: diagrams/causal_structures.png")

# ============================================================================
# Diagram 2: A/B Testing Workflow
# ============================================================================
print("2. Generating ab_testing_workflow.png...")

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

def draw_box(ax, x, y, width, height, text, color='lightblue', fontsize=11):
    """Draw a rectangular box with text."""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow_between(ax, x1, y1, x2, y2, label='', color='black'):
    """Draw arrow between boxes."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='->', lw=2, color=color))
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.5, mid_y, label, fontsize=9, style='italic')

def draw_decision(ax, x, y, width, height, text, color='#ffff99'):
    """Draw a diamond-shaped decision box."""
    points = [(x + width/2, y + height),  # top
              (x + width, y + height/2),   # right
              (x + width/2, y),             # bottom
              (x, y + height/2)]            # left
    diamond = plt.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center', fontsize=10, fontweight='bold')

# Step 1: Define hypothesis
draw_box(ax, 3, 10.5, 4, 0.8, '1. Define Hypothesis & Metric', color='#ccffcc')

# Step 2: Power analysis
draw_arrow_between(ax, 5, 10.5, 5, 9.8)
draw_box(ax, 3, 9, 4, 0.8, '2. Power Analysis\n(Calculate Sample Size)', color='#ccffcc')

# Step 3: Randomization
draw_arrow_between(ax, 5, 9, 5, 8.3)
draw_box(ax, 3, 7.5, 4, 0.8, '3. Randomize Users\n(50/50 Treatment/Control)', color='#cce5ff')

# Split into two branches
draw_arrow_between(ax, 4, 7.5, 2, 6.8)
draw_arrow_between(ax, 6, 7.5, 8, 6.8)

draw_box(ax, 0.5, 6, 3, 0.8, 'Treatment Group\n(New Feature)', color='#ffe5cc')
draw_box(ax, 6.5, 6, 3, 0.8, 'Control Group\n(Baseline)', color='#ffe5cc')

# Step 4: Collect data
draw_arrow_between(ax, 2, 6, 2, 5.3)
draw_arrow_between(ax, 8, 6, 8, 5.3)
draw_box(ax, 0.5, 4.5, 3, 0.8, 'Collect Data\n(Treatment)', color='#ffcccc')
draw_box(ax, 6.5, 4.5, 3, 0.8, 'Collect Data\n(Control)', color='#ffcccc')

# Merge back
draw_arrow_between(ax, 2, 4.5, 3.5, 3.8)
draw_arrow_between(ax, 8, 4.5, 6.5, 3.8)

# Step 5: Analysis
draw_box(ax, 3, 3, 4, 0.8, '5. Statistical Analysis\n(t-test, CI)', color='#e5ccff')

# Step 6: Decision
draw_arrow_between(ax, 5, 3, 5, 2.3)
draw_decision(ax, 3.5, 0.8, 3, 1.2, 'Significant\n& Positive?')

# Outcomes
draw_arrow_between(ax, 7, 1.4, 8.5, 1.4, 'Yes', color='green')
draw_box(ax, 8.5, 1, 1.2, 0.8, 'Ship', color='#90EE90')

draw_arrow_between(ax, 5, 0.8, 5, 0.2, 'No')
draw_box(ax, 4, -0.6, 2, 0.8, 'Iterate\nor Abandon', color='#FFB6C6')

ax.text(5, 11.7, 'A/B Testing Workflow', fontsize=16, fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('diagrams/ab_testing_workflow.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved: diagrams/ab_testing_workflow.png")

# ============================================================================
# Diagram 3: Power Curve
# ============================================================================
print("3. Generating power_curve.png...")

# Parameters
baseline_conversion = 0.05
treatment_effect = 0.005
treatment_conversion = baseline_conversion + treatment_effect
n_users = 10000

# Effect size calculation
effect_size = proportion_effectsize(treatment_conversion, baseline_conversion)

# Calculate required sample size for 80% power
required_n = zt_ind_solve_power(effect_size=effect_size,
                                power=0.80,
                                alpha=0.05,
                                ratio=1.0,
                                alternative='two-sided')

# Simulate power curves for different sample sizes
sample_sizes = np.array([100, 500, 1000, 2000, 5000, 10000, 20000, 50000])
powers = []

for n in sample_sizes:
    p = zt_ind_solve_power(effect_size=effect_size,
                           nobs1=n,
                           alpha=0.05,
                           ratio=1.0,
                           alternative='two-sided')
    powers.append(p)

# Plot power curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sample_sizes, powers, marker='o', linewidth=2, markersize=8, color='#2196F3')
ax.axhline(y=0.80, color='#4CAF50', linestyle='--', linewidth=2, label='80% power (target)')
ax.axhline(y=0.05, color='#F44336', linestyle='--', linewidth=2, label='5% (Type I error rate)')
ax.axvline(x=required_n, color='#FF9800', linestyle='--', linewidth=2, alpha=0.7,
          label=f'Required n = {required_n:.0f}')

ax.set_xlabel('Sample Size per Group', fontsize=12, fontweight='bold')
ax.set_ylabel('Statistical Power', fontsize=12, fontweight='bold')
ax.set_title('Power Curve: Probability of Detecting Treatment Effect\n(5% → 5.5% conversion, α=0.05)',
            fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('diagrams/power_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved: diagrams/power_curve.png")

# ============================================================================
# Diagram 4: Multiple Testing
# ============================================================================
print("4. Generating multiple_testing.png...")

# Simulate 20 A/B tests where the NULL is true
n_tests = 20
n_users_per_test = 5000
baseline_rate = 0.05
p_values = []

for test_num in range(n_tests):
    # Both groups have same conversion rate (null is true)
    control = np.random.binomial(1, baseline_rate, n_users_per_test)
    treatment = np.random.binomial(1, baseline_rate, n_users_per_test)  # No effect!

    # Perform t-test
    t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
    p_values.append(p_val)

# Count false positives
false_positives = sum(p < 0.05 for p in p_values)

# Apply Bonferroni correction
bonferroni_alpha = 0.05 / n_tests
bonferroni_positives = sum(p < bonferroni_alpha for p in p_values)

# Apply Benjamini-Hochberg FDR correction
reject_bh, pvals_corrected_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
bh_positives = sum(reject_bh)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: P-value histogram
ax = axes[0]
ax.hist(p_values, bins=20, edgecolor='black', alpha=0.7, color='#2196F3')
ax.axvline(x=0.05, color='#F44336', linestyle='--', linewidth=2, label='α = 0.05 (uncorrected)')
ax.axvline(x=bonferroni_alpha, color='#FF9800', linestyle='--', linewidth=2,
          label=f'α = {bonferroni_alpha:.4f} (Bonferroni)')
ax.set_xlabel('P-value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of P-values (20 tests, all null)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Comparison of correction methods
ax = axes[1]
methods = ['No\nCorrection', 'Bonferroni', 'Benjamini-\nHochberg']
false_pos = [false_positives, bonferroni_positives, bh_positives]
colors = ['#F44336', '#FF9800', '#2196F3']

bars = ax.bar(methods, false_pos, color=colors, edgecolor='black', linewidth=2)
ax.axhline(y=1, color='#4CAF50', linestyle='--', linewidth=2, alpha=0.7,
          label='Expected (1 false positive)')
ax.set_ylabel('Number of False Positives', fontsize=12, fontweight='bold')
ax.set_title('False Positives by Correction Method', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(false_pos) + 1 if max(false_pos) > 0 else 2)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, false_pos):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
           f'{count}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('diagrams/multiple_testing.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved: diagrams/multiple_testing.png")

# ============================================================================
# Diagram 5: Solution 1 - CTR Comparison
# ============================================================================
print("5. Generating solution1_ctr.png...")

# Parameters
n_per_group = 5000
baseline_ctr = 0.03
treatment_ctr = 0.033

# Generate data
control = np.random.binomial(1, baseline_ctr, n_per_group)
treatment = np.random.binomial(1, treatment_ctr, n_per_group)

# Calculate rates
control_rate = control.sum() / n_per_group
treatment_rate = treatment.sum() / n_per_group

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
groups = ['Control', 'Treatment']
rates = [control_rate, treatment_rate]
errors = [1.96*np.sqrt(r*(1-r)/n_per_group) for r in rates]

bars = ax.bar(groups, rates, yerr=errors, capsize=10, alpha=0.7,
              color=['#2196F3', '#F44336'], edgecolor='black', linewidth=2)
ax.set_ylabel('Click-Through Rate', fontsize=12, fontweight='bold')
ax.set_title('Mobile App CTR: Control vs. Treatment\n(Error bars: 95% CI)',
            fontsize=13, fontweight='bold')
ax.set_ylim(0, max(rates)*1.3)

for bar, rate in zip(bars, rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{rate*100:.2f}%', ha='center', va='bottom',
           fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('diagrams/solution1_ctr.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved: diagrams/solution1_ctr.png")

# ============================================================================
# Diagram 6: Solution 3 - Sequential Testing
# ============================================================================
print("6. Generating solution3_sequential.png...")

# Simulate sequential testing
n_days = 30
n_simulations = 1000
users_per_day = 1000
baseline_rate = 0.05

false_positives_fixed = 0
false_positives_sequential = 0
false_positives_peeking = 0

for sim in range(n_simulations):
    cumulative_control = []
    cumulative_treatment = []

    for day in range(n_days):
        control_today = np.random.binomial(1, baseline_rate, users_per_day // 2)
        treatment_today = np.random.binomial(1, baseline_rate, users_per_day // 2)

        cumulative_control.extend(control_today)
        cumulative_treatment.extend(treatment_today)

        # Naive peeking
        if day >= 2:
            t_stat, p_val = stats.ttest_ind(cumulative_treatment, cumulative_control)
            if p_val < 0.05:
                false_positives_peeking += 1
                break

    # Fixed horizon
    t_stat, p_val = stats.ttest_ind(cumulative_treatment, cumulative_control)
    if p_val < 0.05:
        false_positives_fixed += 1

    # Sequential with adjustment
    check_days = [7, 14, 21, 30]
    for check_day in check_days:
        if len(cumulative_control) >= check_day * users_per_day // 2:
            adjusted_alpha = 0.05 * (check_day / n_days) ** 0.5
            control_subset = cumulative_control[:check_day * users_per_day // 2]
            treatment_subset = cumulative_treatment[:check_day * users_per_day // 2]
            t_stat, p_val = stats.ttest_ind(treatment_subset, control_subset)
            if p_val < adjusted_alpha / len(check_days):
                false_positives_sequential += 1
                break

fpr_fixed = false_positives_fixed / n_simulations
fpr_sequential = false_positives_sequential / n_simulations
fpr_peeking = false_positives_peeking / n_simulations

# Visualization
days = np.arange(1, n_days + 1)
cumulative_alpha = 0.05 * (days / n_days) ** 0.5
uniform_alpha = np.linspace(0, 0.05, n_days)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Cumulative alpha spent
ax = axes[0]
ax.plot(days, cumulative_alpha, linewidth=2, label='O\'Brien-Fleming', color='#2196F3')
ax.plot(days, uniform_alpha, linewidth=2, linestyle='--', label='Uniform spending', color='#F44336')
ax.axhline(y=0.05, color='#4CAF50', linestyle=':', linewidth=2, label='Total α = 0.05')
ax.set_xlabel('Day', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative α Spent', fontsize=12, fontweight='bold')
ax.set_title('Alpha Spending Function', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: False positive rate comparison
ax = axes[1]
methods = ['Fixed\nHorizon', 'Sequential\n(OBF)', 'Naive\nPeeking']
fprs = [fpr_fixed, fpr_sequential, fpr_peeking]
colors = ['#4CAF50', '#2196F3', '#F44336']

bars = ax.bar(methods, fprs, color=colors, edgecolor='black', linewidth=2)
ax.axhline(y=0.05, color='#4CAF50', linestyle='--', linewidth=2, label='Target α=0.05')
ax.set_ylabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_title(f'False Positive Rates ({n_simulations} simulations)', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(fprs) * 1.2)

for bar, fpr in zip(bars, fprs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{fpr:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('diagrams/solution3_sequential.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✓ Saved: diagrams/solution3_sequential.png")

print("\n" + "=" * 70)
print("✓ All diagrams generated successfully!")
print("=" * 70)

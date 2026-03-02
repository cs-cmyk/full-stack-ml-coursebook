"""
Code Review Test Script for Chapter 39: Causal Inference
This script tests all code blocks from content.md
"""

import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest, proportion_confint
from statsmodels.stats.multitest import multipletests

# Track test results
results = []

def test_block(block_num, description, code_func):
    """Test a code block and record results"""
    try:
        code_func()
        results.append((block_num, description, "PASS", None))
        print(f"✓ Block {block_num}: {description}")
        return True
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        results.append((block_num, description, "FAIL", error_msg))
        print(f"✗ Block {block_num}: {description}")
        print(f"  Error: {error_msg}")
        traceback.print_exc()
        return False

# ============================================================
# BLOCK 1: Causal Structure Gallery
# ============================================================
def block1_causal_structures():
    np.random.seed(42)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle('Fundamental Causal Structures', fontsize=16, fontweight='bold')

    def draw_node(ax, x, y, label, color='lightblue'):
        circle = plt.Circle((x, y), 0.15, color=color, ec='black', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold', zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2, color='black', linewidth=2):
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

    draw_node(ax, 1, 2, 'Z', color='#ff9999')
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
    draw_node(ax, 1.25, 1.25, 'M', color='#ffff99')
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
    draw_node(ax, 1, 0.75, 'C', color='#ff99ff')
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
    draw_arrow(ax, 0.9, 1.85, 0.6, 0.65)
    draw_arrow(ax, 1.1, 1.85, 1.4, 0.65)
    draw_arrow(ax, 0.65, 0.5, 1.35, 0.5, color='green', linewidth=2)

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
    plt.savefig('book/course-10-advanced/ch39-causal/diagrams/causal_structures.png', dpi=300, bbox_inches='tight')
    plt.close()

test_block(1, "Causal Structure Gallery", block1_causal_structures)

# ============================================================
# BLOCK 2: A/B Testing Workflow
# ============================================================
def block2_ab_workflow():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def draw_box(ax, x, y, width, height, text, color='lightblue', fontsize=11):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_arrow_between(ax, x1, y1, x2, y2, label='', color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color=color))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.5, mid_y, label, fontsize=9, style='italic')

    def draw_decision(ax, x, y, width, height, text, color='#ffff99'):
        points = [(x + width/2, y + height),
                  (x + width, y + height/2),
                  (x + width/2, y),
                  (x, y + height/2)]
        diamond = plt.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=10, fontweight='bold')

    draw_box(ax, 3, 10.5, 4, 0.8, '1. Define Hypothesis & Metric', color='#ccffcc')
    draw_arrow_between(ax, 5, 10.5, 5, 9.8)
    draw_box(ax, 3, 9, 4, 0.8, '2. Power Analysis\n(Calculate Sample Size)', color='#ccffcc')
    draw_arrow_between(ax, 5, 9, 5, 8.3)
    draw_box(ax, 3, 7.5, 4, 0.8, '3. Randomize Users\n(50/50 Treatment/Control)', color='#cce5ff')
    draw_arrow_between(ax, 4, 7.5, 2, 6.8)
    draw_arrow_between(ax, 6, 7.5, 8, 6.8)
    draw_box(ax, 0.5, 6, 3, 0.8, 'Treatment Group\n(New Feature)', color='#ffe5cc')
    draw_box(ax, 6.5, 6, 3, 0.8, 'Control Group\n(Baseline)', color='#ffe5cc')
    draw_arrow_between(ax, 2, 6, 2, 5.3)
    draw_arrow_between(ax, 8, 6, 8, 5.3)
    draw_box(ax, 0.5, 4.5, 3, 0.8, 'Collect Data\n(Treatment)', color='#ffcccc')
    draw_box(ax, 6.5, 4.5, 3, 0.8, 'Collect Data\n(Control)', color='#ffcccc')
    draw_arrow_between(ax, 2, 4.5, 3.5, 3.8)
    draw_arrow_between(ax, 8, 4.5, 6.5, 3.8)
    draw_box(ax, 3, 3, 4, 0.8, '5. Statistical Analysis\n(t-test, CI)', color='#e5ccff')
    draw_arrow_between(ax, 5, 3, 5, 2.3)
    draw_decision(ax, 3.5, 0.8, 3, 1.2, 'Significant\n& Positive?')
    draw_arrow_between(ax, 7, 1.4, 8.5, 1.4, 'Yes', color='green')
    draw_box(ax, 8.5, 1, 1.2, 0.8, 'Ship', color='#90EE90')
    draw_arrow_between(ax, 5, 0.8, 5, 0.2, 'No')
    draw_box(ax, 4, -0.6, 2, 0.8, 'Iterate\nor Abandon', color='#FFB6C6')

    ax.text(5, 11.7, 'A/B Testing Workflow', fontsize=16, fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig('book/course-10-advanced/ch39-causal/diagrams/ab_testing_workflow.png', dpi=300, bbox_inches='tight')
    plt.close()

test_block(2, "A/B Testing Workflow", block2_ab_workflow)

# ============================================================
# BLOCK 3: Simulating RCT (A/B Test)
# ============================================================
def block3_rct_simulation():
    np.random.seed(42)

    n_users = 10000
    baseline_conversion = 0.05
    treatment_effect = 0.005
    treatment_conversion = baseline_conversion + treatment_effect

    treatment = np.random.choice([0, 1], size=n_users, p=[0.5, 0.5])

    conversions = np.zeros(n_users)
    for i in range(n_users):
        if treatment[i] == 0:
            conversions[i] = np.random.binomial(1, baseline_conversion)
        else:
            conversions[i] = np.random.binomial(1, treatment_conversion)

    df = pd.DataFrame({
        'user_id': range(n_users),
        'treatment': treatment,
        'converted': conversions
    })

    conversion_stats = df.groupby('treatment')['converted'].agg(['sum', 'count', 'mean'])
    conversion_stats.index = ['Control', 'Treatment']
    conversion_stats.columns = ['Conversions', 'Total Users', 'Conversion Rate']
    conversion_stats['Conversion Rate (%)'] = conversion_stats['Conversion Rate'] * 100

    control_rate = conversion_stats.loc['Control', 'Conversion Rate']
    treatment_rate = conversion_stats.loc['Treatment', 'Conversion Rate']
    absolute_diff = treatment_rate - control_rate

    assert df.shape == (10000, 3), "DataFrame shape mismatch"
    assert 'user_id' in df.columns, "Missing user_id column"
    assert 'treatment' in df.columns, "Missing treatment column"
    assert 'converted' in df.columns, "Missing converted column"

test_block(3, "Simulating RCT (A/B Test)", block3_rct_simulation)

# ============================================================
# BLOCK 4: Statistical Hypothesis Testing
# ============================================================
def block4_hypothesis_testing():
    np.random.seed(42)

    # Recreate data from Block 3
    n_users = 10000
    baseline_conversion = 0.05
    treatment_effect = 0.005
    treatment_conversion = baseline_conversion + treatment_effect

    treatment = np.random.choice([0, 1], size=n_users, p=[0.5, 0.5])
    conversions = np.zeros(n_users)
    for i in range(n_users):
        if treatment[i] == 0:
            conversions[i] = np.random.binomial(1, baseline_conversion)
        else:
            conversions[i] = np.random.binomial(1, treatment_conversion)

    df = pd.DataFrame({
        'user_id': range(n_users),
        'treatment': treatment,
        'converted': conversions
    })

    control_conversions = df[df['treatment'] == 0]['converted'].values
    treatment_conversions = df[df['treatment'] == 1]['converted'].values

    t_stat, p_value = stats.ttest_ind(treatment_conversions, control_conversions, equal_var=False)

    conversion_stats = df.groupby('treatment')['converted'].agg(['sum', 'count', 'mean'])
    conversion_stats.index = ['Control', 'Treatment']
    conversion_stats.columns = ['Conversions', 'Total Users', 'Conversion Rate']

    p1 = conversion_stats.loc['Treatment', 'Conversion Rate']
    p2 = conversion_stats.loc['Control', 'Conversion Rate']
    n1 = len(treatment_conversions)
    n2 = len(control_conversions)

    se_diff = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    ci_lower = (p1 - p2) - 1.96 * se_diff
    ci_upper = (p1 - p2) + 1.96 * se_diff

    assert isinstance(t_stat, (int, float)), "t_stat should be numeric"
    assert isinstance(p_value, (int, float)), "p_value should be numeric"
    assert 0 <= p_value <= 1, "p_value should be between 0 and 1"

test_block(4, "Statistical Hypothesis Testing", block4_hypothesis_testing)

# ============================================================
# BLOCK 5: Statistical Power Analysis
# ============================================================
def block5_power_analysis():
    np.random.seed(42)

    n_users = 10000
    baseline_conversion = 0.05
    treatment_effect = 0.005
    treatment_conversion = baseline_conversion + treatment_effect

    effect_size = proportion_effectsize(treatment_conversion, baseline_conversion)

    power = zt_ind_solve_power(effect_size=effect_size,
                                nobs1=n_users/2,
                                alpha=0.05,
                                ratio=1.0,
                                alternative='two-sided')

    required_n = zt_ind_solve_power(effect_size=effect_size,
                                     power=0.80,
                                     alpha=0.05,
                                     ratio=1.0,
                                     alternative='two-sided')

    sample_sizes = np.array([100, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    powers = []

    for n in sample_sizes:
        p = zt_ind_solve_power(effect_size=effect_size,
                               nobs1=n,
                               alpha=0.05,
                               ratio=1.0,
                               alternative='two-sided')
        powers.append(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, powers, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax.axhline(y=0.80, color='green', linestyle='--', linewidth=2, label='80% power (target)')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='5% (Type I error rate)')
    ax.axvline(x=required_n, color='orange', linestyle='--', linewidth=2, alpha=0.7,
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
    plt.savefig('book/course-10-advanced/ch39-causal/diagrams/power_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    assert 0 <= power <= 1, "Power should be between 0 and 1"
    assert required_n > 0, "Required sample size should be positive"

test_block(5, "Statistical Power Analysis", block5_power_analysis)

# ============================================================
# BLOCK 6: Multiple Testing Problem
# ============================================================
def block6_multiple_testing():
    np.random.seed(42)

    n_tests = 20
    n_users_per_test = 5000
    baseline_rate = 0.05
    p_values = []

    for test_num in range(n_tests):
        control = np.random.binomial(1, baseline_rate, n_users_per_test)
        treatment = np.random.binomial(1, baseline_rate, n_users_per_test)
        t_stat, p_val = stats.ttest_ind(treatment, control, equal_var=False)
        p_values.append(p_val)

    false_positives = sum(p < 0.05 for p in p_values)
    false_positive_rate = false_positives / n_tests

    bonferroni_alpha = 0.05 / n_tests
    bonferroni_positives = sum(p < bonferroni_alpha for p in p_values)

    reject_bh, pvals_corrected_bh, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    bh_positives = sum(reject_bh)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(p_values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05 (uncorrected)')
    ax.axvline(x=bonferroni_alpha, color='orange', linestyle='--', linewidth=2,
              label=f'α = {bonferroni_alpha:.4f} (Bonferroni)')
    ax.set_xlabel('P-value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of P-values (20 tests, all null)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    methods = ['No\nCorrection', 'Bonferroni', 'Benjamini-\nHochberg']
    false_pos = [false_positives, bonferroni_positives, bh_positives]
    colors = ['#e74c3c', '#f39c12', '#3498db']

    bars = ax.bar(methods, false_pos, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=2, alpha=0.7,
              label='Expected (1 false positive)')
    ax.set_ylabel('Number of False Positives', fontsize=12, fontweight='bold')
    ax.set_title('False Positives by Correction Method', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(false_pos) + 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, false_pos):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('book/course-10-advanced/ch39-causal/diagrams/multiple_testing.png', dpi=300, bbox_inches='tight')
    plt.close()

    assert len(p_values) == 20, "Should have 20 p-values"

test_block(6, "Multiple Testing Problem", block6_multiple_testing)

# ============================================================
# BLOCK 7: Solution 1 - Mobile App CTR
# ============================================================
def block7_solution1():
    np.random.seed(42)

    n_per_group = 5000
    baseline_ctr = 0.03
    treatment_ctr = 0.033
    treatment_effect = treatment_ctr - baseline_ctr

    control = np.random.binomial(1, baseline_ctr, n_per_group)
    treatment = np.random.binomial(1, treatment_ctr, n_per_group)

    control_clicks = control.sum()
    treatment_clicks = treatment.sum()

    count = np.array([treatment_clicks, control_clicks])
    nobs = np.array([n_per_group, n_per_group])
    z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

    control_rate = control_clicks / n_per_group
    treatment_rate = treatment_clicks / n_per_group
    diff = treatment_rate - control_rate

    se_diff = np.sqrt((treatment_rate*(1-treatment_rate)/n_per_group) +
                      (control_rate*(1-control_rate)/n_per_group))
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff

    effect_size = proportion_effectsize(treatment_ctr, baseline_ctr)
    power = zt_ind_solve_power(effect_size=effect_size, nobs1=n_per_group,
                               alpha=0.05, ratio=1.0, alternative='two-sided')

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = ['Control', 'Treatment']
    rates = [control_rate, treatment_rate]
    errors = [1.96*np.sqrt(r*(1-r)/n_per_group) for r in rates]

    bars = ax.bar(groups, rates, yerr=errors, capsize=10, alpha=0.7,
                  color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=2)
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
    plt.savefig('book/course-10-advanced/ch39-causal/diagrams/solution1_ctr.png', dpi=300, bbox_inches='tight')
    plt.close()

    assert isinstance(z_stat, (int, float)), "z_stat should be numeric"
    assert 0 <= power <= 1, "Power should be between 0 and 1"

test_block(7, "Solution 1 - Mobile App CTR", block7_solution1)

# ============================================================
# BLOCK 8: Solution 2 - Multi-variant Test
# ============================================================
def block8_solution2():
    np.random.seed(42)

    baseline_rate = 0.05
    mde = 0.01
    alpha = 0.05
    power_target = 0.80
    n_variants = 5

    effect_size = proportion_effectsize(baseline_rate + mde, baseline_rate)
    n_single = zt_ind_solve_power(effect_size=effect_size, power=power_target,
                                   alpha=alpha, ratio=1.0, alternative='two-sided')

    alpha_bonferroni = alpha / n_variants
    n_bonferroni = zt_ind_solve_power(effect_size=effect_size, power=power_target,
                                       alpha=alpha_bonferroni, ratio=1.0,
                                       alternative='two-sided')

    n = int(n_bonferroni)

    control = np.random.binomial(1, baseline_rate, n)
    variant_A = np.random.binomial(1, baseline_rate, n)
    variant_B = np.random.binomial(1, baseline_rate, n)
    variant_C = np.random.binomial(1, baseline_rate, n)
    variant_D = np.random.binomial(1, baseline_rate + 0.012, n)
    variant_E = np.random.binomial(1, baseline_rate, n)

    variants = [variant_A, variant_B, variant_C, variant_D, variant_E]
    variant_names = ['A', 'B', 'C', 'D', 'E']

    p_values = []
    effects = []

    for variant, name in zip(variants, variant_names):
        count = np.array([variant.sum(), control.sum()])
        nobs = np.array([n, n])
        z_stat, p_val = stats.ttest_ind(variant, control, equal_var=False)
        p_values.append(stats.ttest_ind(variant, control, equal_var=False)[1])

        variant_rate = variant.sum() / n
        control_rate = control.sum() / n
        effect = (variant_rate - control_rate) * 100
        effects.append(effect)

    alpha_3variants = alpha / 3
    n_3variants = zt_ind_solve_power(effect_size=effect_size, power=power_target,
                                     alpha=alpha_3variants, ratio=1.0,
                                     alternative='two-sided')

    assert len(p_values) == 5, "Should have 5 p-values"
    assert n_bonferroni > n_single, "Bonferroni should require more samples"

test_block(8, "Solution 2 - Multi-variant Test", block8_solution2)

# ============================================================
# BLOCK 9: Solution 3 - Sequential Testing
# ============================================================
def block9_solution3():
    np.random.seed(42)

    def obrien_fleming_bounds(n_looks, alpha=0.05):
        alpha_spent = []
        cumulative_alpha = 0

        for k in range(1, n_looks + 1):
            information_fraction = k / n_looks
            z_threshold = stats.norm.ppf(1 - alpha / (2 * np.sqrt(n_looks / k)))
            alpha_spent.append(2 * (1 - stats.norm.cdf(z_threshold)))
            cumulative_alpha += alpha_spent[-1]

        return alpha_spent, z_threshold

    n_days = 30
    users_per_day = 1000
    baseline_rate = 0.05
    n_simulations = 1000

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

            if day >= 2:
                t_stat, p_val = stats.ttest_ind(cumulative_treatment, cumulative_control)
                if p_val < 0.05:
                    false_positives_peeking += 1
                    break
        else:
            pass

        t_stat, p_val = stats.ttest_ind(cumulative_treatment, cumulative_control)
        if p_val < 0.05:
            false_positives_fixed += 1

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

    days = np.arange(1, n_days + 1)
    cumulative_alpha = 0.05 * (days / n_days) ** 0.5
    uniform_alpha = np.linspace(0, 0.05, n_days)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(days, cumulative_alpha, linewidth=2, label='O\'Brien-Fleming', color='#3498db')
    ax.plot(days, uniform_alpha, linewidth=2, linestyle='--', label='Uniform spending', color='#e74c3c')
    ax.axhline(y=0.05, color='green', linestyle=':', linewidth=2, label='Total α = 0.05')
    ax.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative α Spent', fontsize=12, fontweight='bold')
    ax.set_title('Alpha Spending Function', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    methods = ['Fixed\nHorizon', 'Sequential\n(OBF)', 'Naive\nPeeking']
    fprs = [fpr_fixed, fpr_sequential, fpr_peeking]
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    bars = ax.bar(methods, fprs, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2, label='Target α=0.05')
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
    plt.savefig('book/course-10-advanced/ch39-causal/diagrams/solution3_sequential.png', dpi=300, bbox_inches='tight')
    plt.close()

    assert 0 <= fpr_fixed <= 1, "FPR should be between 0 and 1"
    assert fpr_peeking > fpr_fixed, "Peeking should inflate FPR"

test_block(9, "Solution 3 - Sequential Testing", block9_solution3)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("CODE REVIEW SUMMARY")
print("="*60)

passed = sum(1 for r in results if r[2] == "PASS")
failed = sum(1 for r in results if r[2] == "FAIL")

for block_num, desc, status, error in results:
    if status == "FAIL":
        print(f"Block {block_num} ({desc}): {status}")
        print(f"  Error: {error}")

print(f"\nTotal: {passed}/{len(results)} blocks passed")

if failed == 0:
    print("\n✓ ALL TESTS PASSED")
    sys.exit(0)
else:
    print(f"\n✗ {failed} TESTS FAILED")
    sys.exit(1)

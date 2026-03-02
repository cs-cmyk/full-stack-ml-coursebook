import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Change to the chapter directory
os.chdir('/home/chirag/ds-book/book/course-18/ch53')

# Use consistent color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

np.random.seed(42)

# 1. Simulate two models
class CTRModel:
    def __init__(self, name, true_ctr):
        self.name = name
        self.true_ctr = true_ctr

    def serve(self, n_users):
        """Simulate serving model to n_users, return clicks"""
        clicks = np.random.binomial(1, self.true_ctr, n_users)
        return clicks.sum(), n_users

champion = CTRModel("Champion", true_ctr=0.08)
challenger = CTRModel("Challenger", true_ctr=0.09)

# 2. Implement progressive rollout schedule
rollout_schedule = [
    {'day': 1, 'challenger_pct': 0.01, 'total_users': 5000},
    {'day': 2, 'challenger_pct': 0.05, 'total_users': 5000},
    {'day': 3, 'challenger_pct': 0.10, 'total_users': 5000},
    {'day': 4, 'challenger_pct': 0.25, 'total_users': 5000},
    {'day': 5, 'challenger_pct': 0.50, 'total_users': 5000},
    {'day': 6, 'challenger_pct': 1.00, 'total_users': 5000},
]

# Track results
results = []
rollback_triggered = False

for stage in rollout_schedule:
    day = stage['day']
    challenger_pct = stage['challenger_pct']
    total_users = stage['total_users']

    # Allocate users
    n_challenger = int(total_users * challenger_pct)
    n_champion = total_users - n_challenger

    # Simulate interactions
    champion_clicks, champion_impressions = champion.serve(n_champion)
    challenger_clicks, challenger_impressions = challenger.serve(n_challenger)

    # Calculate CTRs
    champion_ctr = champion_clicks / champion_impressions if champion_impressions > 0 else 0
    challenger_ctr = challenger_clicks / challenger_impressions if challenger_impressions > 0 else 0

    # Calculate 95% confidence intervals (Wilson score interval)
    def wilson_ci(successes, n, z=1.96):
        if n == 0:
            return 0, 0
        p_hat = successes / n
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
        return center - margin, center + margin

    champion_ci_low, champion_ci_high = wilson_ci(champion_clicks, champion_impressions)
    challenger_ci_low, challenger_ci_high = wilson_ci(challenger_clicks, challenger_impressions)

    # Statistical significance test (chi-square)
    if challenger_impressions > 0 and champion_impressions > 0:
        contingency = [
            [challenger_clicks, challenger_impressions - challenger_clicks],
            [champion_clicks, champion_impressions - champion_clicks]
        ]
        _, pvalue, _, _ = chi2_contingency(contingency)
    else:
        pvalue = 1.0

    # 4. Automated rollback logic
    rollback = False
    if challenger_impressions > 100:  # Need minimum samples
        if challenger_ctr < champion_ctr - 0.005 and pvalue < 0.05:
            rollback = True
            rollback_triggered = True

    results.append({
        'Day': day,
        'Challenger_Pct': challenger_pct,
        'Champion_Impressions': champion_impressions,
        'Champion_Clicks': champion_clicks,
        'Champion_CTR': champion_ctr,
        'Champion_CI_Low': champion_ci_low,
        'Champion_CI_High': champion_ci_high,
        'Challenger_Impressions': challenger_impressions,
        'Challenger_Clicks': challenger_clicks,
        'Challenger_CTR': challenger_ctr,
        'Challenger_CI_Low': challenger_ci_low,
        'Challenger_CI_High': challenger_ci_high,
        'P_Value': pvalue,
        'Rollback': rollback
    })

    if rollback:
        break

results_df = pd.DataFrame(results)

# 5. Visualize traffic allocation and performance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

# Traffic allocation over time
days = results_df['Day']
axes[0, 0].fill_between(days, 0, (1 - results_df['Challenger_Pct']) * 100,
                        alpha=0.7, color=colors['blue'], label='Champion')
axes[0, 0].fill_between(days, (1 - results_df['Challenger_Pct']) * 100, 100,
                        alpha=0.7, color=colors['orange'], label='Challenger')
axes[0, 0].set_xlabel('Day', fontsize=12)
axes[0, 0].set_ylabel('Traffic Allocation (%)', fontsize=12)
axes[0, 0].set_title('Canary Release: Progressive Rollout', fontweight='bold', fontsize=13)
axes[0, 0].legend(loc='center left', fontsize=11)
axes[0, 0].grid(alpha=0.3, axis='y')
axes[0, 0].set_ylim([0, 100])

# CTR comparison with confidence intervals
axes[0, 1].errorbar(days, results_df['Champion_CTR'] * 100,
                   yerr=[(results_df['Champion_CTR'] - results_df['Champion_CI_Low']) * 100,
                         (results_df['Champion_CI_High'] - results_df['Champion_CTR']) * 100],
                   marker='o', linewidth=2, capsize=5, label='Champion', color=colors['blue'])
axes[0, 1].errorbar(days, results_df['Challenger_CTR'] * 100,
                   yerr=[(results_df['Challenger_CTR'] - results_df['Challenger_CI_Low']) * 100,
                         (results_df['Challenger_CI_High'] - results_df['Challenger_CTR']) * 100],
                   marker='s', linewidth=2, capsize=5, label='Challenger', color=colors['orange'])
axes[0, 1].set_xlabel('Day', fontsize=12)
axes[0, 1].set_ylabel('CTR (%)', fontsize=12)
axes[0, 1].set_title('CTR Over Time (with 95% CI)', fontweight='bold', fontsize=13)
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(alpha=0.3)

# Cumulative clicks
champion_cumulative = results_df['Champion_Clicks'].cumsum()
challenger_cumulative = results_df['Challenger_Clicks'].cumsum()
axes[1, 0].plot(days, champion_cumulative, marker='o', linewidth=2.5,
               color=colors['blue'], label='Champion')
axes[1, 0].plot(days, challenger_cumulative, marker='s', linewidth=2.5,
               color=colors['orange'], label='Challenger')
axes[1, 0].set_xlabel('Day', fontsize=12)
axes[1, 0].set_ylabel('Cumulative Clicks', fontsize=12)
axes[1, 0].set_title('Cumulative Clicks Over Time', fontweight='bold', fontsize=13)
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(alpha=0.3)

# 6. Compare to instant cutover (blue-green)
blue_green_clicks = 0
blue_green_impressions = 0
for stage in rollout_schedule:
    clicks, impressions = challenger.serve(stage['total_users'])
    blue_green_clicks += clicks
    blue_green_impressions += impressions

canary_total_clicks = (results_df['Champion_Clicks'].sum() +
                       results_df['Challenger_Clicks'].sum())

# If challenger had been defective (CTR = 0.04 instead of 0.09)
defective_challenger = CTRModel("Defective", true_ctr=0.04)

# Canary approach: limited exposure
canary_protected_clicks = 0
for stage in rollout_schedule[:3]:  # Assume caught by day 3
    n_challenger = int(stage['total_users'] * stage['challenger_pct'])
    n_champion = stage['total_users'] - n_challenger
    defect_clicks, _ = defective_challenger.serve(n_challenger)
    champ_clicks, _ = champion.serve(n_champion)
    canary_protected_clicks += (defect_clicks + champ_clicks)

# Blue-green: full exposure
blue_green_defective_clicks, _ = defective_challenger.serve(30000)

comparison_data = {
    'Deployment': ['Canary (Good)', 'Blue-Green (Good)',
                   'Canary (if Defective)', 'Blue-Green (if Defective)'],
    'Total_Clicks': [
        canary_total_clicks,
        blue_green_clicks,
        canary_protected_clicks,
        blue_green_defective_clicks
    ]
}

comparison_df = pd.DataFrame(comparison_data)
bar_colors = [colors['green'], colors['blue'], colors['orange'], colors['red']]
axes[1, 1].barh(comparison_df['Deployment'], comparison_df['Total_Clicks'],
               color=bar_colors, alpha=0.7)
axes[1, 1].set_xlabel('Total Clicks', fontsize=12)
axes[1, 1].set_title('Canary vs Blue-Green: Risk Comparison', fontweight='bold', fontsize=13)
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('diagrams/exercise2_canary_release.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diagrams/exercise2_canary_release.png")
plt.close()

"""
Data Quality Metrics Dashboard
Shows quality check results and trends over time
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
purple = '#9C27B0'
gray = '#607D8B'

# Subplot 1: Quality Check Results (Pie Chart)
ax1 = fig.add_subplot(gs[0, 0])
checks = ['Row Count', 'Null Values', 'Range Check', 'Schema', 'Duplicates']
passed = [28, 27, 25, 30, 29]
failed = [2, 3, 5, 0, 1]

x = np.arange(len(checks))
width = 0.7

bars1 = ax1.bar(x, passed, width, label='Passed', color=green, alpha=0.8)
bars2 = ax1.bar(x, failed, width, bottom=passed, label='Failed', color=red, alpha=0.8)

ax1.set_ylabel('Number of Runs', fontsize=11, fontweight='bold')
ax1.set_title('Quality Checks (Last 30 Days)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(checks, rotation=45, ha='right', fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add percentage labels
for i, (p, f) in enumerate(zip(passed, failed)):
    total = p + f
    pct = (p / total) * 100
    ax1.text(i, total + 0.5, f'{pct:.0f}%', ha='center', fontsize=9, fontweight='bold')

# Subplot 2: Data Volume Trends
ax2 = fig.add_subplot(gs[0, 1:])
dates = np.arange(30)
volume = 20000 + np.random.normal(0, 1000, 30).cumsum()
volume = np.clip(volume, 15000, 30000)

ax2.plot(dates, volume, color=blue, linewidth=2, marker='o', markersize=4, alpha=0.8)
ax2.fill_between(dates, volume, alpha=0.2, color=blue)

# Add threshold lines
expected_min, expected_max = 18000, 25000
ax2.axhline(y=expected_min, color=green, linestyle='--', linewidth=2, alpha=0.7, label='Expected Range')
ax2.axhline(y=expected_max, color=green, linestyle='--', linewidth=2, alpha=0.7)
ax2.fill_between(dates, expected_min, expected_max, alpha=0.1, color=green)

# Highlight anomalies
anomalies = volume < expected_min
if anomalies.any():
    ax2.scatter(dates[anomalies], volume[anomalies], color=red, s=100, zorder=5, marker='x', linewidths=3, label='Anomaly')

ax2.set_xlabel('Days Ago', fontsize=11, fontweight='bold')
ax2.set_ylabel('Row Count', fontsize=11, fontweight='bold')
ax2.set_title('Daily Data Volume Trend', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, linestyle='--')

# Subplot 3: Null Rate by Column
ax3 = fig.add_subplot(gs[1, 0])
columns = ['user_id', 'timestamp', 'amount', 'category', 'location']
null_rates = [0.1, 0.0, 2.5, 1.8, 4.2]
colors_null = [green if rate < 1 else orange if rate < 3 else red for rate in null_rates]

bars = ax3.barh(columns, null_rates, color=colors_null, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, rate) in enumerate(zip(bars, null_rates)):
    ax3.text(rate + 0.2, i, f'{rate}%', va='center', fontsize=10, fontweight='bold')

ax3.set_xlabel('Null Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('Null Values by Column', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 5)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# Add threshold line
ax3.axvline(x=1, color=green, linestyle='--', linewidth=2, alpha=0.5, label='Acceptable (<1%)')
ax3.axvline(x=3, color=orange, linestyle='--', linewidth=2, alpha=0.5, label='Warning (<3%)')
ax3.legend(fontsize=8, loc='lower right')

# Subplot 4: Pipeline Success Rate Over Time
ax4 = fig.add_subplot(gs[1, 1:])
weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
success_rates = [92, 88, 95, 97]
failure_rates = [8, 12, 5, 3]

x4 = np.arange(len(weeks))
bars_s = ax4.bar(x4, success_rates, width=0.6, label='Success', color=green, alpha=0.8)
bars_f = ax4.bar(x4, failure_rates, width=0.6, bottom=success_rates, label='Failed', color=red, alpha=0.8)

# Add value labels
for i, (s, f) in enumerate(zip(success_rates, failure_rates)):
    ax4.text(i, s/2, f'{s}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax4.text(i, s + f/2, f'{f}%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

ax4.set_ylabel('Percentage', fontsize=11, fontweight='bold')
ax4.set_title('Pipeline Success Rate (Weekly)', fontsize=12, fontweight='bold')
ax4.set_xticks(x4)
ax4.set_xticklabels(weeks)
ax4.set_ylim(0, 100)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add SLA line
ax4.axhline(y=95, color=blue, linestyle='--', linewidth=2, alpha=0.7, label='SLA (95%)')

# Subplot 5: Data Quality Score Breakdown
ax5 = fig.add_subplot(gs[2, :2])
metrics = ['Completeness', 'Validity', 'Consistency', 'Timeliness', 'Accuracy']
current_scores = [98, 94, 96, 92, 95]
target_scores = [95, 95, 95, 95, 95]

x5 = np.arange(len(metrics))
width5 = 0.35

bars_current = ax5.bar(x5 - width5/2, current_scores, width5, label='Current Score', color=blue, alpha=0.8)
bars_target = ax5.bar(x5 + width5/2, target_scores, width5, label='Target (SLA)', color=gray, alpha=0.5, edgecolor='black', linewidth=2, linestyle='--')

# Add value labels
for bars in [bars_current, bars_target]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax5.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
ax5.set_title('Data Quality Score by Dimension', fontsize=12, fontweight='bold')
ax5.set_xticks(x5)
ax5.set_xticklabels(metrics)
ax5.set_ylim(80, 100)
ax5.legend(fontsize=9)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

# Subplot 6: Overall Quality Summary
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

# Create quality score gauge effect
overall_score = 95.0
score_color = green if overall_score >= 95 else orange if overall_score >= 90 else red

summary_text = f"""
Overall Quality Score

{overall_score}%

━━━━━━━━━━━━━━━━

Last 30 Days:
✓ Runs: 28 / 30
✗ Failures: 2
⟳ Retries: 5

Top Issues:
• Range violations: 5
• Late arrivals: 3
• Schema changes: 2

SLA Status: {'✓ PASS' if overall_score >= 95 else '✗ FAIL'}
"""

ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes,
        fontsize=11, verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round,pad=1', facecolor=score_color, alpha=0.2, edgecolor=score_color, linewidth=3),
        family='monospace', fontweight='bold')

plt.suptitle('Data Pipeline Quality Dashboard', fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/home/chirag/ds-book/book/course-08-big-data/ch32-data-pipelines/diagrams/data_quality_metrics.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: data_quality_metrics.png")

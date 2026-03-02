"""
Airflow Task Execution Timeline
Visualizes parallel execution, dependencies, and retry behavior
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
purple = '#9C27B0'
gray = '#607D8B'

# Chart 1: Normal execution with parallel tasks
tasks = ['extract_data', 'validate_raw', 'transform_sales', 'transform_customers', 'load_warehouse', 'quality_check', 'notify']
start_times = [0, 5, 10, 10, 20, 25, 30]
durations = [5, 5, 10, 10, 5, 5, 2]
colors_normal = [blue, orange, green, green, purple, orange, blue]

y_pos = np.arange(len(tasks))

for i, (task, start, duration, color) in enumerate(zip(tasks, start_times, durations, colors_normal)):
    ax1.barh(i, duration, left=start, height=0.6, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
    # Add task duration label
    ax1.text(start + duration/2, i, f'{duration}m', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

# Add dependency arrows
arrow_props = dict(arrowstyle='->', lw=2, color=gray)
ax1.annotate('', xy=(10, 2), xytext=(10, 1.4), arrowprops=arrow_props)
ax1.annotate('', xy=(10, 3), xytext=(10, 1.4), arrowprops=arrow_props)
ax1.annotate('', xy=(20, 4), xytext=(20, 2.4), arrowprops=arrow_props)
ax1.annotate('', xy=(20, 4), xytext=(20, 3.4), arrowprops=arrow_props)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(tasks, fontsize=11)
ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
ax1.set_title('Airflow DAG Execution Timeline: Parallel Tasks', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 35)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.axvline(x=32, color=green, linestyle='--', linewidth=2, alpha=0.7)
ax1.text(32.5, 6.5, 'Total: 32m', fontsize=11, fontweight='bold', color=green)

# Add legend
legend_elements = [
    mpatches.Patch(color=blue, label='Extract/Load', alpha=0.8),
    mpatches.Patch(color=green, label='Transform', alpha=0.8),
    mpatches.Patch(color=orange, label='Validate', alpha=0.8),
    mpatches.Patch(color=purple, label='Load', alpha=0.8)
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Chart 2: Execution with retry behavior
tasks2 = ['extract_data', 'transform', 'load (attempt 1)', 'load (retry 1)', 'load (retry 2)', 'validate', 'notify']
start_times2 = [0, 5, 15, 21, 27, 33, 38]
durations2 = [5, 10, 3, 3, 3, 5, 2]
colors_retry = [blue, green, red, orange, green, orange, blue]
statuses = ['success', 'success', 'failed', 'failed', 'success', 'success', 'success']

y_pos2 = np.arange(len(tasks2))

for i, (task, start, duration, color, status) in enumerate(zip(tasks2, start_times2, durations2, colors_retry, statuses)):
    edgecolor = red if status == 'failed' else 'black'
    linewidth = 2.5 if status == 'failed' else 1.5
    alpha = 0.5 if status == 'failed' else 0.8

    ax2.barh(i, duration, left=start, height=0.6, color=color, alpha=alpha,
            edgecolor=edgecolor, linewidth=linewidth)

    # Add status icon
    if status == 'failed':
        ax2.text(start + duration/2, i, '✗', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
    else:
        ax2.text(start + duration/2, i, f'{duration}m', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

# Add retry delay indicators
for retry_idx, (start, prev_end) in enumerate([(21, 18), (27, 24)], 1):
    delay = start - prev_end
    mid_x = (start + prev_end) / 2
    ax2.annotate('', xy=(start, 2+retry_idx), xytext=(prev_end, 2+retry_idx),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=gray, linestyle='--'))
    ax2.text(mid_x, 2.3+retry_idx, f'{delay}m delay', ha='center', va='bottom',
            fontsize=9, color=gray, style='italic')

ax2.set_yticks(y_pos2)
ax2.set_yticklabels(tasks2, fontsize=11)
ax2.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
ax2.set_title('Airflow DAG with Retry Logic (3 attempts, 5-min delay)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 45)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.axvline(x=40, color=orange, linestyle='--', linewidth=2, alpha=0.7)
ax2.text(40.5, 6.5, 'Total: 40m\n(+8m retries)', fontsize=11, fontweight='bold', color=orange)

# Add legend for retry chart
legend_elements2 = [
    mpatches.Patch(color=green, label='Success', alpha=0.8),
    mpatches.Patch(color=red, label='Failed', alpha=0.5),
    mpatches.Patch(color=orange, label='Retry', alpha=0.8)
]
ax2.legend(handles=legend_elements2, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-08-big-data/ch32-data-pipelines/diagrams/task_execution_timeline.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: task_execution_timeline.png")

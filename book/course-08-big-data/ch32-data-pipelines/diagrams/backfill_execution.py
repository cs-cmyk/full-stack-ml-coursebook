"""
Backfill Execution Pattern
Shows how Airflow processes historical dates during backfill operations
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, timedelta

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
purple = '#9C27B0'
gray = '#607D8B'

# Chart 1: Backfill execution sequence
dates = ['Jan 01', 'Jan 02', 'Jan 03', 'Jan 04', 'Jan 05', 'Jan 06', 'Jan 07']
n_dates = len(dates)

# Simulate execution timeline
execution_blocks = []
current_time = 0
for i, date in enumerate(dates):
    # Each date has extract, transform, load tasks
    extract_duration = 3
    transform_duration = 5
    load_duration = 2

    execution_blocks.append({
        'date': date,
        'extract': (current_time, extract_duration),
        'transform': (current_time + extract_duration, transform_duration),
        'load': (current_time + extract_duration + transform_duration, load_duration)
    })
    current_time += extract_duration + transform_duration + load_duration

# Plot the backfill execution
y_positions = np.arange(n_dates)
bar_height = 0.7

for i, block in enumerate(execution_blocks):
    # Extract
    start, dur = block['extract']
    ax1.barh(i, dur, left=start, height=bar_height, color=blue, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.text(start + dur/2, i, 'E', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Transform
    start, dur = block['transform']
    ax1.barh(i, dur, left=start, height=bar_height, color=green, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.text(start + dur/2, i, 'T', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Load
    start, dur = block['load']
    ax1.barh(i, dur, left=start, height=bar_height, color=purple, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.text(start + dur/2, i, 'L', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax1.set_yticks(y_positions)
ax1.set_yticklabels(dates, fontsize=11)
ax1.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Execution Date', fontsize=12, fontweight='bold')
ax1.set_title('Sequential Backfill: max_active_runs=1 (Safe, Slower)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add total time indicator
total_time = current_time
ax1.axvline(x=total_time, color=red, linestyle='--', linewidth=2, alpha=0.7)
ax1.text(total_time + 1, n_dates - 0.5, f'Total: {total_time}m', fontsize=11, fontweight='bold', color=red)

# Legend
legend_elements = [
    mpatches.Patch(color=blue, label='Extract', alpha=0.8),
    mpatches.Patch(color=green, label='Transform', alpha=0.8),
    mpatches.Patch(color=purple, label='Load', alpha=0.8)
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Chart 2: Parallel backfill execution
# With max_active_runs=3, multiple dates can run simultaneously
parallel_groups = [
    ['Jan 01', 'Jan 02', 'Jan 03'],
    ['Jan 04', 'Jan 05', 'Jan 06'],
    ['Jan 07']
]

y_pos_parallel = np.arange(n_dates)
current_batch_time = 0

for batch_idx, batch_dates in enumerate(parallel_groups):
    for local_idx, date in enumerate(batch_dates):
        global_idx = batch_idx * 3 + local_idx
        if global_idx >= n_dates:
            break

        # All dates in the batch start at the same time
        start_time = current_batch_time

        # Extract
        ax2.barh(global_idx, 3, left=start_time, height=bar_height, color=blue, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.text(start_time + 1.5, global_idx, 'E', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Transform
        ax2.barh(global_idx, 5, left=start_time + 3, height=bar_height, color=green, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.text(start_time + 5.5, global_idx, 'T', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        # Load
        ax2.barh(global_idx, 2, left=start_time + 8, height=bar_height, color=purple, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.text(start_time + 9, global_idx, 'L', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Move to next batch
    current_batch_time += 10  # Duration of one complete DAG run

# Add batch boundaries
for i in [3, 6]:
    if i < n_dates:
        ax2.axhline(y=i - 0.5, color=gray, linestyle='--', linewidth=2, alpha=0.5)

# Add batch labels
ax2.text(-2, 1, 'Batch 1\n(parallel)', ha='right', va='center', fontsize=10,
        fontweight='bold', color=gray, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
ax2.text(-2, 4.5, 'Batch 2\n(parallel)', ha='right', va='center', fontsize=10,
        fontweight='bold', color=gray, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
ax2.text(-2, 6, 'Batch 3', ha='right', va='center', fontsize=10,
        fontweight='bold', color=gray, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

ax2.set_yticks(y_pos_parallel)
ax2.set_yticklabels(dates, fontsize=11)
ax2.set_xlabel('Time (minutes)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Execution Date', fontsize=12, fontweight='bold')
ax2.set_title('Parallel Backfill: max_active_runs=3 (Faster, Higher Load)', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(-3, 80)

# Add total time indicator
parallel_total = 30
ax2.axvline(x=parallel_total, color=green, linestyle='--', linewidth=2, alpha=0.7)
ax2.text(parallel_total + 1, n_dates - 0.5, f'Total: {parallel_total}m', fontsize=11, fontweight='bold', color=green)

# Add speedup annotation
speedup = total_time / parallel_total
ax2.text(40, -1.2, f'Speedup: {speedup:.1f}x faster', fontsize=12, fontweight='bold', color=green,
        bbox=dict(boxstyle='round,pad=0.8', facecolor=green, alpha=0.2, edgecolor=green, linewidth=2))

# Legend
ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-08-big-data/ch32-data-pipelines/diagrams/backfill_execution.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: backfill_execution.png")

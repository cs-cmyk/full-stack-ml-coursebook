"""
ETL vs ELT Performance Comparison
Shows when each pattern is more efficient based on data volume and transformation complexity
"""
import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
gray = '#607D8B'

# Chart 1: Processing time comparison
data_sizes = ['Small\n(< 1GB)', 'Medium\n(1-10GB)', 'Large\n(10-100GB)', 'Very Large\n(> 100GB)']
etl_times = [5, 20, 90, 300]  # minutes
elt_times = [8, 15, 40, 80]   # minutes

x = np.arange(len(data_sizes))
width = 0.35

bars1 = ax1.bar(x - width/2, etl_times, width, label='ETL', color=blue, alpha=0.8)
bars2 = ax1.bar(x + width/2, elt_times, width, label='ELT', color=green, alpha=0.8)

ax1.set_xlabel('Data Volume', fontsize=12, fontweight='bold')
ax1.set_ylabel('Processing Time (minutes)', fontsize=12, fontweight='bold')
ax1.set_title('ETL vs ELT: Processing Time by Data Volume', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(data_sizes)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}m',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Chart 2: Use case recommendations
categories = ['Data\nVolume', 'Transform\nComplexity', 'Warehouse\nCompute', 'Compliance\nNeeds', 'Flexibility']
etl_scores = [6, 8, 3, 9, 4]
elt_scores = [9, 6, 9, 5, 9]

x2 = np.arange(len(categories))
bars3 = ax2.bar(x2 - width/2, etl_scores, width, label='ETL Better', color=blue, alpha=0.8)
bars4 = ax2.bar(x2 + width/2, elt_scores, width, label='ELT Better', color=green, alpha=0.8)

ax2.set_xlabel('Selection Criteria', fontsize=12, fontweight='bold')
ax2.set_ylabel('Suitability Score', fontsize=12, fontweight='bold')
ax2.set_title('ETL vs ELT: When to Use Each Pattern', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylim(0, 10)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Add score labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-08-big-data/ch32-data-pipelines/diagrams/etl_vs_elt_performance.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: etl_vs_elt_performance.png")

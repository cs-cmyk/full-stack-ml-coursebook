#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 31: Big Data Ecosystem
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Create diagrams directory if it doesn't exist
os.makedirs('diagrams', exist_ok=True)

# Set consistent style
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# Color palette (consistent across all diagrams)
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B',
    'dark_blue': '#3498DB',
    'dark_red': '#E74C3C',
    'dark_green': '#2ECC71',
    'dark_orange': '#F39C12',
    'dark_gray': '#2C3E50',
    'light_gray': '#ECF0F1',
}

print("Generating diagrams for Chapter 31: Big Data Ecosystem\n")

# ============================================================================
# Diagram 1: Distributed Computing Architecture
# ============================================================================
print("Creating Diagram 1: Distributed Computing Architecture...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Distributed Computing Architecture',
        fontsize=18, fontweight='bold', ha='center')

# Driver/Master Node
driver = FancyBboxPatch((3.5, 7), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor=COLORS['dark_gray'],
                         facecolor=COLORS['dark_blue'],
                         linewidth=2)
ax.add_patch(driver)
ax.text(5, 7.5, 'Driver/Master\n(Coordinator)',
        fontsize=12, fontweight='bold', ha='center', va='center', color='white')

# Worker/Executor Nodes
worker_positions = [(1, 4), (4, 4), (7, 4)]
worker_colors = [COLORS['dark_red'], COLORS['dark_green'], COLORS['dark_orange']]

for i, (x, y) in enumerate(worker_positions, 1):
    worker = FancyBboxPatch((x-0.75, y-0.5), 1.5, 1,
                             boxstyle="round,pad=0.05",
                             edgecolor=COLORS['dark_gray'],
                             facecolor=worker_colors[i-1],
                             linewidth=2)
    ax.add_patch(worker)
    ax.text(x, y, f'Executor {i}\n(Worker)',
            fontsize=11, ha='center', va='center', color='white', fontweight='bold')

    # Data partitions
    data_box = FancyBboxPatch((x-0.75, y-2), 1.5, 0.8,
                               boxstyle="round,pad=0.05",
                               edgecolor='#34495E',
                               facecolor=COLORS['light_gray'],
                               linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(x, y-1.6, f'Partition {i}\nData',
            fontsize=10, ha='center', va='center', color=COLORS['dark_gray'])

# Arrows: Driver to Workers (Task Assignment)
for x, y in worker_positions:
    arrow = FancyArrowPatch((5, 7), (x, 4.5),
                             arrowstyle='->', mutation_scale=20, linewidth=2,
                             color=COLORS['purple'])
    ax.add_patch(arrow)

ax.text(2.5, 6, 'Task\nAssignment', fontsize=10, ha='center',
        color=COLORS['purple'], fontweight='bold')

# Arrows: Workers to Driver (Results)
for x, y in worker_positions:
    arrow = FancyArrowPatch((x, 4.5), (5, 7),
                             arrowstyle='->', mutation_scale=20, linewidth=1.5,
                             color='#16A085', linestyle='dashed')
    ax.add_patch(arrow)

ax.text(7.5, 6, 'Results', fontsize=10, ha='center',
        color='#16A085', fontweight='bold')

# Data locality annotation
ax.text(5, 1.5, 'Key: Computation happens where data lives (data locality)',
        fontsize=11, ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDEBD0',
                  edgecolor='#E67E22', linewidth=2))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['dark_blue'], edgecolor=COLORS['dark_gray'],
                   label='Coordinator'),
    mpatches.Patch(facecolor=COLORS['dark_red'], edgecolor=COLORS['dark_gray'],
                   label='Workers/Executors'),
    mpatches.Patch(facecolor=COLORS['light_gray'], edgecolor='#34495E',
                   label='Data Partitions'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, frameon=True)

plt.tight_layout()
plt.savefig('diagrams/distributed_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("  ✓ Saved: diagrams/distributed_architecture.png")
plt.close()

# ============================================================================
# Diagram 2: Storage Paradigms Comparison
# ============================================================================
print("\nCreating Diagram 2: Storage Paradigms Comparison...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Big Data Storage Paradigms',
        fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)

# Define storage types
storage_types = [
    {
        'name': 'Data Warehouse',
        'x': 0.15,
        'color': COLORS['blue'],
        'features': [
            '• Schema-on-write',
            '• Structured data (tables)',
            '• Optimized for SQL queries',
            '• Columnar storage',
            '• High query performance',
            '• Examples: Redshift, BigQuery'
        ]
    },
    {
        'name': 'Data Lake',
        'x': 0.5,
        'color': COLORS['green'],
        'features': [
            '• Schema-on-read',
            '• Any format (raw data)',
            '• Low storage cost',
            '• Flexible ingestion',
            '• Good for ML training',
            '• Examples: S3, ADLS, GCS'
        ]
    },
    {
        'name': 'Data Lakehouse',
        'x': 0.85,
        'color': COLORS['purple'],
        'features': [
            '• Best of both worlds',
            '• ACID transactions',
            '• Schema enforcement',
            '• Time travel',
            '• Unified architecture',
            '• Examples: Delta Lake, Iceberg'
        ]
    }
]

y_start = 0.75

for storage in storage_types:
    # Draw box
    box = mpatches.FancyBboxPatch(
        (storage['x'] - 0.12, 0.25), 0.24, 0.6,
        boxstyle="round,pad=0.02",
        edgecolor=storage['color'],
        facecolor=storage['color'],
        alpha=0.2,
        linewidth=3,
        transform=ax.transAxes
    )
    ax.add_patch(box)

    # Title
    ax.text(storage['x'], y_start, storage['name'],
            fontsize=14, fontweight='bold', ha='center',
            transform=ax.transAxes, color=storage['color'])

    # Features
    y_pos = y_start - 0.08
    for feature in storage['features']:
        ax.text(storage['x'], y_pos, feature,
                fontsize=10, ha='center', va='top',
                transform=ax.transAxes)
        y_pos -= 0.06

# Add comparison table at bottom
ax.text(0.5, 0.12, 'Choose based on use case:',
        fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.5, 0.07,
        'Warehouse → BI dashboards & reports  |  Lake → ML & exploration  |  Lakehouse → Unified analytics',
        fontsize=10, ha='center', transform=ax.transAxes, style='italic')

plt.tight_layout()
plt.savefig('diagrams/storage_paradigms.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("  ✓ Saved: diagrams/storage_paradigms.png")
plt.close()

# ============================================================================
# Diagram 3: Spark Architecture and Lazy Evaluation
# ============================================================================
print("\nCreating Diagram 3: Spark Architecture and Lazy Evaluation...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Spark Architecture
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Spark Cluster Architecture', fontsize=14, fontweight='bold', pad=20)

# Driver
driver = FancyBboxPatch((3, 7.5), 4, 1.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=COLORS['dark_gray'],
                         facecolor=COLORS['blue'],
                         linewidth=2)
ax1.add_patch(driver)
ax1.text(5, 8.25, 'SparkContext\n(Driver Program)',
         fontsize=11, fontweight='bold', ha='center', va='center', color='white')

# Cluster Manager
manager = FancyBboxPatch((3, 5.5), 4, 1,
                          boxstyle="round,pad=0.05",
                          edgecolor=COLORS['dark_gray'],
                          facecolor=COLORS['orange'],
                          linewidth=2)
ax1.add_patch(manager)
ax1.text(5, 6, 'Cluster Manager\n(YARN/Mesos/K8s)',
         fontsize=10, ha='center', va='center', color='white', fontweight='bold')

# Executors
executor_positions = [(1.5, 2.5), (4.5, 2.5), (7.5, 2.5)]
for i, (x, y) in enumerate(executor_positions, 1):
    exec_box = FancyBboxPatch((x-1, y-0.75), 2, 2,
                               boxstyle="round,pad=0.05",
                               edgecolor=COLORS['dark_gray'],
                               facecolor=COLORS['green'],
                               linewidth=2,
                               alpha=0.7)
    ax1.add_patch(exec_box)
    ax1.text(x, y+0.5, f'Executor {i}',
             fontsize=10, ha='center', fontweight='bold', color='white')

    # Tasks inside executor
    task_y = y - 0.2
    for j in range(2):
        task = FancyBboxPatch((x-0.6+j*0.7, task_y-0.4), 0.5, 0.3,
                               boxstyle="round,pad=0.02",
                               edgecolor='white',
                               facecolor=COLORS['dark_green'],
                               linewidth=1)
        ax1.add_patch(task)
        ax1.text(x-0.35+j*0.7, task_y-0.25, f'Task',
                 fontsize=8, ha='center', va='center', color='white')

# Arrows
for x, y in executor_positions:
    # Driver to manager
    arrow1 = FancyArrowPatch((5, 7.5), (5, 6.5),
                              arrowstyle='<->', mutation_scale=15, linewidth=1.5,
                              color=COLORS['dark_gray'])
    # Manager to executors
    arrow2 = FancyArrowPatch((5, 5.5), (x, y+1.25),
                              arrowstyle='<->', mutation_scale=15, linewidth=1.5,
                              color=COLORS['dark_gray'])
    ax1.add_patch(arrow2)

arrow1 = FancyArrowPatch((5, 7.5), (5, 6.5),
                          arrowstyle='<->', mutation_scale=15, linewidth=1.5,
                          color=COLORS['dark_gray'])
ax1.add_patch(arrow1)

# Right panel: Lazy Evaluation
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Lazy Evaluation: Transformations vs Actions',
              fontsize=14, fontweight='bold', pad=20)

# Transformations (lazy)
trans_y = 7.5
ax2.text(5, trans_y+0.5, 'Transformations (Lazy)',
         fontsize=12, fontweight='bold', ha='center',
         color=COLORS['blue'])

transformations = ['filter()', 'map()', 'groupBy()', 'join()']
for i, trans in enumerate(transformations):
    y = trans_y - i*0.7
    box = FancyBboxPatch((2, y-0.25), 6, 0.5,
                          boxstyle="round,pad=0.05",
                          edgecolor=COLORS['blue'],
                          facecolor=COLORS['blue'],
                          linewidth=2,
                          alpha=0.3)
    ax2.add_patch(box)
    ax2.text(5, y, trans,
             fontsize=11, ha='center', va='center',
             fontweight='bold', color=COLORS['blue'])

    if i < len(transformations) - 1:
        arrow = FancyArrowPatch((5, y-0.3), (5, y-0.65),
                                 arrowstyle='->', mutation_scale=15, linewidth=2,
                                 color=COLORS['blue'])
        ax2.add_patch(arrow)

# Build DAG annotation
ax2.text(8.5, 5.5, 'Build\nDAG',
         fontsize=10, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                   edgecolor=COLORS['orange'], linewidth=2))

# Actions (eager)
action_y = 3.5
arrow = FancyArrowPatch((5, 4.5), (5, action_y+0.8),
                         arrowstyle='->', mutation_scale=20, linewidth=3,
                         color=COLORS['red'])
ax2.add_patch(arrow)

ax2.text(5, action_y+0.3, 'Actions (Trigger Execution)',
         fontsize=12, fontweight='bold', ha='center',
         color=COLORS['red'])

actions = ['count()', 'collect()', 'show()', 'save()']
for i, action in enumerate(actions):
    y = action_y - i*0.6
    box = FancyBboxPatch((2, y-0.2), 6, 0.4,
                          boxstyle="round,pad=0.05",
                          edgecolor=COLORS['red'],
                          facecolor=COLORS['red'],
                          linewidth=2,
                          alpha=0.3)
    ax2.add_patch(box)
    ax2.text(5, y, action,
             fontsize=11, ha='center', va='center',
             fontweight='bold', color=COLORS['red'])

# Execute annotation
ax2.text(1.2, 1.5, 'Execute\nDAG',
         fontsize=10, ha='center', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral',
                   edgecolor=COLORS['red'], linewidth=2))

plt.tight_layout()
plt.savefig('diagrams/spark_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("  ✓ Saved: diagrams/spark_architecture.png")
plt.close()

# ============================================================================
# Diagram 4: Partitioning Strategy Visualization
# ============================================================================
print("\nCreating Diagram 4: Partitioning Strategy Visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Left: Unpartitioned data
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Unpartitioned Storage', fontsize=14, fontweight='bold', pad=20)

# Single large file
file_box = FancyBboxPatch((2, 3), 6, 4,
                           boxstyle="round,pad=0.1",
                           edgecolor=COLORS['gray'],
                           facecolor=COLORS['light_gray'],
                           linewidth=2)
ax1.add_patch(file_box)
ax1.text(5, 7.5, 'sales_data.parquet', fontsize=12, ha='center', fontweight='bold')

# Sample data rows with different categories (color-coded)
categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple'], COLORS['red']]

y = 6.5
for i in range(15):
    cat_idx = i % 5
    row = FancyBboxPatch((2.5, y-i*0.22), 5, 0.18,
                          edgecolor='none',
                          facecolor=colors[cat_idx],
                          alpha=0.5)
    ax1.add_patch(row)

# Query indicator
query_box = mpatches.FancyBboxPatch((1, 1), 8, 0.8,
                                     boxstyle="round,pad=0.05",
                                     edgecolor=COLORS['red'],
                                     facecolor='white',
                                     linewidth=3,
                                     linestyle='--')
ax1.add_patch(query_box)
ax1.text(5, 1.4, 'Query: category = "Electronics"',
         fontsize=11, ha='center', fontweight='bold', color=COLORS['red'])
ax1.text(5, 0.5, '❌ Must scan entire file', fontsize=10, ha='center',
         color=COLORS['red'], style='italic')

# Right: Partitioned data
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Partitioned by Category', fontsize=14, fontweight='bold', pad=20)

# Separate files for each category
partition_x = [1.5, 3.5, 5.5, 7.5]
partition_y = [7, 5.5, 4, 2.5]

for i in range(5):
    x = 1.5 + (i % 2) * 4
    y = 7 - (i // 2) * 1.5

    # Partition folder
    folder = FancyBboxPatch((x-0.7, y-0.9), 1.4, 1.2,
                             boxstyle="round,pad=0.05",
                             edgecolor=colors[i],
                             facecolor=colors[i],
                             linewidth=2,
                             alpha=0.3)
    ax2.add_patch(folder)

    # Label
    ax2.text(x, y+0.1, f'category=', fontsize=9, ha='center')
    ax2.text(x, y-0.15, f'{categories[i]}', fontsize=9, ha='center', fontweight='bold')

    # File icon
    file_icon = FancyBboxPatch((x-0.3, y-0.6), 0.6, 0.35,
                                boxstyle="round,pad=0.02",
                                edgecolor=COLORS['dark_gray'],
                                facecolor=colors[i],
                                linewidth=1)
    ax2.add_patch(file_icon)
    ax2.text(x, y-0.42, 'data', fontsize=7, ha='center', color='white')

# Query indicator - only reads Electronics partition
query_arrow = FancyArrowPatch((4, 0.8), (1.5, 6.2),
                               arrowstyle='->', mutation_scale=25, linewidth=3,
                               color=COLORS['red'])
ax2.add_patch(query_arrow)

query_box2 = mpatches.FancyBboxPatch((2.5, 0.2), 5, 0.8,
                                      boxstyle="round,pad=0.05",
                                      edgecolor=COLORS['green'],
                                      facecolor='lightgreen',
                                      linewidth=3,
                                      alpha=0.3)
ax2.add_patch(query_box2)
ax2.text(5, 0.6, 'Query: category = "Electronics"',
         fontsize=11, ha='center', fontweight='bold', color=COLORS['red'])
ax2.text(5, 0.2, '✓ Only reads 1 partition (5× faster)', fontsize=10, ha='center',
         color=COLORS['green'], style='italic', fontweight='bold')

plt.tight_layout()
plt.savefig('diagrams/partitioning_strategy.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("  ✓ Saved: diagrams/partitioning_strategy.png")
plt.close()

# ============================================================================
# Diagram 5: Cloud Big Data Services Comparison
# ============================================================================
print("\nCreating Diagram 5: Cloud Big Data Services Comparison...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Title
ax.text(0.5, 0.97, 'Cloud Big Data Services Comparison',
        fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)

# Define cloud providers and their services
providers = [
    {
        'name': 'AWS',
        'color': COLORS['orange'],
        'logo_color': '#FF9900',
        'x': 0.15,
        'services': [
            ('Storage', 'S3 (Data Lake)'),
            ('Processing', 'EMR (Spark)'),
            ('Warehouse', 'Redshift'),
            ('Streaming', 'Kinesis'),
            ('Lakehouse', 'Lake Formation'),
            ('ML Platform', 'SageMaker')
        ]
    },
    {
        'name': 'Google Cloud',
        'color': COLORS['blue'],
        'logo_color': '#4285F4',
        'x': 0.5,
        'services': [
            ('Storage', 'GCS (Data Lake)'),
            ('Processing', 'Dataproc (Spark)'),
            ('Warehouse', 'BigQuery'),
            ('Streaming', 'Dataflow'),
            ('Lakehouse', 'BigLake'),
            ('ML Platform', 'Vertex AI')
        ]
    },
    {
        'name': 'Azure',
        'color': COLORS['blue'],
        'logo_color': '#0078D4',
        'x': 0.85,
        'services': [
            ('Storage', 'ADLS Gen2'),
            ('Processing', 'HDInsight/Databricks'),
            ('Warehouse', 'Synapse Analytics'),
            ('Streaming', 'Stream Analytics'),
            ('Lakehouse', 'Synapse + Delta'),
            ('ML Platform', 'Azure ML')
        ]
    }
]

# Draw each provider's services
for provider in providers:
    y_start = 0.85

    # Provider header
    header = mpatches.FancyBboxPatch(
        (provider['x'] - 0.12, y_start), 0.24, 0.08,
        boxstyle="round,pad=0.01",
        edgecolor=provider['logo_color'],
        facecolor=provider['logo_color'],
        linewidth=2,
        transform=ax.transAxes
    )
    ax.add_patch(header)
    ax.text(provider['x'], y_start+0.04, provider['name'],
            fontsize=14, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes, color='white')

    # Services
    y_pos = y_start - 0.08
    for category, service in provider['services']:
        # Category label
        ax.text(provider['x'], y_pos, category,
                fontsize=10, ha='center', fontweight='bold',
                transform=ax.transAxes, color=provider['color'])

        # Service box
        service_box = mpatches.FancyBboxPatch(
            (provider['x'] - 0.11, y_pos - 0.055), 0.22, 0.04,
            boxstyle="round,pad=0.005",
            edgecolor=provider['color'],
            facecolor=provider['color'],
            alpha=0.2,
            linewidth=1.5,
            transform=ax.transAxes
        )
        ax.add_patch(service_box)

        ax.text(provider['x'], y_pos - 0.035, service,
                fontsize=9, ha='center',
                transform=ax.transAxes)

        y_pos -= 0.095

# Add comparison notes at bottom
notes_y = 0.15
ax.text(0.5, notes_y, 'Key Considerations:',
        fontsize=13, fontweight='bold', ha='center', transform=ax.transAxes)

notes = [
    'AWS: Most mature ecosystem, widest service catalog, strong for S3-based data lakes',
    'GCP: Best analytics performance (BigQuery), unified data platform, great for ML',
    'Azure: Enterprise integration, Databricks partnership, good for hybrid cloud'
]

notes_y -= 0.035
for note in notes:
    ax.text(0.5, notes_y, f'• {note}',
            fontsize=10, ha='center', transform=ax.transAxes)
    notes_y -= 0.035

# Add footer
ax.text(0.5, 0.02, 'Choose based on: existing infrastructure, team expertise, pricing model, and specific workload requirements',
        fontsize=9, ha='center', style='italic', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('diagrams/cloud_services_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("  ✓ Saved: diagrams/cloud_services_comparison.png")
plt.close()

print("\n" + "="*60)
print("All diagrams generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. distributed_architecture.png")
print("  2. storage_paradigms.png")
print("  3. spark_architecture.png")
print("  4. partitioning_strategy.png")
print("  5. cloud_services_comparison.png")

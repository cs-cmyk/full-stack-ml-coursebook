"""
Deployment Strategy Comparison
Compares serverless, VM, and Kubernetes deployment strategies
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color palette
color_serverless = '#9C27B0'
color_vm = '#2196F3'
color_k8s = '#4CAF50'

# Title
ax.text(7, 9.5, 'Deployment Strategy Comparison', ha='center', fontsize=16, fontweight='bold')

# Strategy boxes
strategies = [
    {
        'name': 'Serverless\n(AWS Lambda, Cloud Functions)',
        'x': 1,
        'y': 5,
        'color': color_serverless,
        'pros': [
            '✓ No infrastructure management',
            '✓ Pay per invocation',
            '✓ Auto-scales to zero',
            '✓ Simple deployment'
        ],
        'cons': [
            '✗ Cold start latency (1-5s)',
            '✗ Timeout limits (15 min)',
            '✗ Memory limits (10GB)',
            '✗ Not for always-on services'
        ],
        'use_case': 'Sporadic Traffic\n(Image processing, batch jobs)'
    },
    {
        'name': 'Single VM\n(EC2, GCE, DigitalOcean)',
        'x': 5.5,
        'y': 5,
        'color': color_vm,
        'pros': [
            '✓ Simple to understand',
            '✓ Full control',
            '✓ Predictable costs',
            '✓ No cold starts'
        ],
        'cons': [
            '✗ Manual scaling',
            '✗ No high availability',
            '✗ Requires maintenance',
            '✗ Single point of failure'
        ],
        'use_case': 'Low Traffic\n(MVPs, internal tools)'
    },
    {
        'name': 'Kubernetes\n(EKS, GKE, AKS)',
        'x': 10,
        'y': 5,
        'color': color_k8s,
        'pros': [
            '✓ Auto-scaling (HPA)',
            '✓ High availability',
            '✓ Self-healing',
            '✓ Zero-downtime updates'
        ],
        'cons': [
            '✗ Complex setup',
            '✗ Requires expertise',
            '✗ Higher minimum cost',
            '✗ Overhead for simple apps'
        ],
        'use_case': 'High Traffic\n(Production, mission-critical)'
    }
]

for strategy in strategies:
    # Main box
    ax.add_patch(FancyBboxPatch((strategy['x'] - 0.8, strategy['y'] - 2),
                                 3.2, 6.5, boxstyle="round,pad=0.15",
                                 edgecolor=strategy['color'], facecolor=strategy['color'],
                                 alpha=0.15, linewidth=3))

    # Title
    ax.text(strategy['x'] + 0.8, strategy['y'] + 4.2, strategy['name'],
            ha='center', fontsize=11, fontweight='bold', color=strategy['color'])

    # Pros section
    ax.add_patch(FancyBboxPatch((strategy['x'] - 0.6, strategy['y'] + 1.8),
                                 2.8, 2, boxstyle="round,pad=0.1",
                                 edgecolor='#4CAF50', facecolor='#4CAF50',
                                 alpha=0.2, linewidth=1.5))
    ax.text(strategy['x'] + 0.8, strategy['y'] + 3.5, 'Pros',
            ha='center', fontsize=10, fontweight='bold', color='#2E7D32')

    y_offset = strategy['y'] + 3.1
    for i, pro in enumerate(strategy['pros']):
        ax.text(strategy['x'] - 0.5, y_offset - i * 0.35, pro,
                ha='left', fontsize=8, color='#2E7D32')

    # Cons section
    ax.add_patch(FancyBboxPatch((strategy['x'] - 0.6, strategy['y'] - 0.5),
                                 2.8, 2, boxstyle="round,pad=0.1",
                                 edgecolor='#F44336', facecolor='#F44336',
                                 alpha=0.2, linewidth=1.5))
    ax.text(strategy['x'] + 0.8, strategy['y'] + 1.2, 'Cons',
            ha='center', fontsize=10, fontweight='bold', color='#C62828')

    y_offset = strategy['y'] + 0.8
    for i, con in enumerate(strategy['cons']):
        ax.text(strategy['x'] - 0.5, y_offset - i * 0.35, con,
                ha='left', fontsize=8, color='#C62828')

    # Use case
    ax.add_patch(FancyBboxPatch((strategy['x'] - 0.6, strategy['y'] - 2.2),
                                 2.8, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor=strategy['color'], facecolor=strategy['color'],
                                 alpha=0.3, linewidth=1.5))
    ax.text(strategy['x'] + 0.8, strategy['y'] - 1.8, 'Best For:',
            ha='center', fontsize=9, fontweight='bold', color=strategy['color'])
    ax.text(strategy['x'] + 0.8, strategy['y'] - 2.2, strategy['use_case'],
            ha='center', fontsize=9, style='italic', color=strategy['color'])

# Bottom comparison metrics
metrics_y = 0.8
ax.text(7, metrics_y + 1, 'Quick Decision Matrix', ha='center',
        fontsize=12, fontweight='bold', color='#607D8B')

metrics = [
    ('Traffic Pattern', 'Sporadic', 'Steady/Low', 'High/Variable'),
    ('Latency Tolerance', 'High (1-5s OK)', 'Low (<100ms)', 'Very Low (<50ms)'),
    ('Cost Model', 'Pay per request', 'Fixed monthly', 'Scales with usage'),
    ('Setup Time', '< 1 hour', '< 1 day', '1-2 weeks'),
    ('Expertise Needed', 'Low', 'Medium', 'High')
]

y_offset = metrics_y + 0.3
for i, (metric, serverless, vm, k8s) in enumerate(metrics):
    y = y_offset - i * 0.25

    # Metric name
    ax.text(2, y, metric + ':', ha='right', fontsize=8, fontweight='bold', color='#424242')

    # Values
    ax.text(3.5, y, serverless, ha='center', fontsize=8, color=color_serverless)
    ax.text(7, y, vm, ha='center', fontsize=8, color=color_vm)
    ax.text(11, y, k8s, ha='center', fontsize=8, color=color_k8s)

# Vertical separators
for x in [2.5, 5.2, 8.8, 12]:
    ax.plot([x, x], [metrics_y - 0.5, metrics_y + 0.4], 'k--', alpha=0.2, linewidth=1)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-09-mlops/ch33-model-deployment/diagrams/deployment_strategies.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated: deployment_strategies.png")

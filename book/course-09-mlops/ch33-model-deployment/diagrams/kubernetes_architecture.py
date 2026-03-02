"""
Kubernetes Architecture Diagram
Visualizes pods, services, load balancing, and autoscaling
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color palette
color_user = '#9C27B0'
color_lb = '#2196F3'
color_service = '#4CAF50'
color_pod = '#FF9800'
color_hpa = '#F44336'
color_container = '#607D8B'

# Title
ax.text(7, 9.5, 'Kubernetes Deployment Architecture', ha='center', fontsize=16, fontweight='bold')

# 1. External Users
user_y = 8
for i, x in enumerate([1, 2, 3]):
    circle = Circle((x, user_y), 0.3, color=color_user, alpha=0.7)
    ax.add_patch(circle)
    ax.text(x, user_y, '👤', ha='center', va='center', fontsize=16)
ax.text(2, user_y + 0.7, 'Users', ha='center', fontsize=11, fontweight='bold')

# 2. Load Balancer
lb_y = 6.5
ax.add_patch(FancyBboxPatch((0.5, lb_y), 3, 1, boxstyle="round,pad=0.1",
                             edgecolor=color_lb, facecolor=color_lb, alpha=0.3, linewidth=2))
ax.text(2, lb_y + 0.5, 'Load Balancer\n(External IP)', ha='center', va='center',
        fontsize=11, fontweight='bold')

# Arrow from users to LB
arrow = FancyArrowPatch((2, user_y - 0.4), (2, lb_y + 1),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                        color=color_user, alpha=0.7)
ax.add_patch(arrow)
ax.text(2.5, 7.3, 'HTTP\nRequest', ha='center', fontsize=9, style='italic', color=color_user)

# 3. Kubernetes Service
service_y = 4.5
ax.add_patch(FancyBboxPatch((0.5, service_y), 3, 1.2, boxstyle="round,pad=0.1",
                             edgecolor=color_service, facecolor=color_service, alpha=0.3, linewidth=2))
ax.text(2, service_y + 0.8, 'Service', ha='center', fontsize=12, fontweight='bold')
ax.text(2, service_y + 0.4, 'iris-api-service\n(ClusterIP: 10.96.123.45)', ha='center',
        fontsize=9, style='italic')

# Arrow from LB to Service
arrow = FancyArrowPatch((2, lb_y), (2, service_y + 1.2),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5,
                        color=color_lb, alpha=0.7)
ax.add_patch(arrow)
ax.text(1.2, 5.9, 'Routes to\nService', ha='center', fontsize=9, style='italic', color=color_lb)

# 4. Pod Cluster
pod_positions = [
    (5.5, 5.5, 'Pod 1'),
    (8, 5.5, 'Pod 2'),
    (10.5, 5.5, 'Pod 3')
]

for i, (x, y, label) in enumerate(pod_positions):
    # Pod box
    ax.add_patch(FancyBboxPatch((x - 1, y - 1.5), 2.2, 2.5, boxstyle="round,pad=0.1",
                                 edgecolor=color_pod, facecolor=color_pod, alpha=0.2, linewidth=2))
    ax.text(x, y + 0.9, label, ha='center', fontsize=11, fontweight='bold', color=color_pod)

    # Container inside pod
    ax.add_patch(FancyBboxPatch((x - 0.8, y - 1.2), 1.8, 1.8, boxstyle="round,pad=0.05",
                                 edgecolor=color_container, facecolor=color_container, alpha=0.4, linewidth=1.5))
    ax.text(x, y - 0.6, 'Container\niris-api:v1', ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')
    ax.text(x, y + 0.1, 'Model\nLoaded', ha='center', va='center', fontsize=8, style='italic')

    # Resource labels
    ax.text(x, y - 1.8, 'CPU: 100m-500m\nMem: 128Mi-256Mi', ha='center', fontsize=7,
            style='italic', color='#607D8B')

# Arrows from Service to Pods
for x, y, _ in pod_positions:
    arrow = FancyArrowPatch((2.8, service_y + 0.6), (x - 1, y + 0.5),
                            arrowstyle='->', mutation_scale=20, linewidth=2,
                            color=color_service, alpha=0.5, linestyle='dashed')
    ax.add_patch(arrow)

ax.text(5.5, 4, 'Load Balanced\nDistribution', ha='center', fontsize=9,
        style='italic', color=color_service)

# 5. Horizontal Pod Autoscaler (HPA)
hpa_y = 8
ax.add_patch(FancyBboxPatch((5, hpa_y), 6, 1, boxstyle="round,pad=0.1",
                             edgecolor=color_hpa, facecolor=color_hpa, alpha=0.3, linewidth=2))
ax.text(8, hpa_y + 0.5, 'Horizontal Pod Autoscaler (HPA)', ha='center', va='center',
        fontsize=11, fontweight='bold')
ax.text(8, hpa_y + 0.1, 'Target: 70% CPU | Min: 2 | Max: 10', ha='center',
        fontsize=9, style='italic')

# HPA monitoring arrows
for x, y, _ in pod_positions:
    arrow = FancyArrowPatch((8, hpa_y), (x, y + 1.2),
                            arrowstyle='<->', mutation_scale=20, linewidth=1.5,
                            color=color_hpa, alpha=0.5, linestyle='dotted')
    ax.add_patch(arrow)

ax.text(12.5, 6.8, 'Monitors\nCPU/Memory', ha='center', fontsize=9,
        style='italic', color=color_hpa)

# 6. Health Probes Box
probe_y = 1.5
ax.add_patch(FancyBboxPatch((0.5, probe_y), 3, 2, boxstyle="round,pad=0.1",
                             edgecolor='#4CAF50', facecolor='#4CAF50', alpha=0.2, linewidth=2))
ax.text(2, probe_y + 1.6, 'Health Probes', ha='center', fontsize=11, fontweight='bold', color='#4CAF50')

probes = [
    ('Startup Probe', 'Max 60s for model loading'),
    ('Liveness Probe', 'Restart if /health fails'),
    ('Readiness Probe', 'Remove from service if not ready')
]
y_offset = probe_y + 1.2
for i, (probe_name, desc) in enumerate(probes):
    y = y_offset - i * 0.4
    ax.text(0.7, y, f'• {probe_name}:', ha='left', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.text(0.85, y - 0.15, desc, ha='left', fontsize=7, style='italic', color='#607D8B')

# 7. Deployment Config Box
deploy_y = 1.5
ax.add_patch(FancyBboxPatch((10.5, deploy_y), 3, 2, boxstyle="round,pad=0.1",
                             edgecolor='#2196F3', facecolor='#2196F3', alpha=0.2, linewidth=2))
ax.text(12, deploy_y + 1.6, 'Deployment', ha='center', fontsize=11, fontweight='bold', color='#2196F3')

deploy_details = [
    'Replicas: 3',
    'Rolling updates',
    'Self-healing',
    'Version: v1'
]
y_offset = deploy_y + 1.1
for i, detail in enumerate(deploy_details):
    y = y_offset - i * 0.3
    ax.text(10.7, y, f'• {detail}', ha='left', fontsize=8, color='#1976D2')

# 8. Traffic flow annotation
ax.text(7, 0.5, 'Traffic Flow: User → Load Balancer → Service → Pods (Round-robin)',
        ha='center', fontsize=10, style='italic', color='#607D8B',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', alpha=0.8))

# Legend
legend_elements = [
    mpatches.Patch(color=color_user, alpha=0.7, label='External Users'),
    mpatches.Patch(color=color_lb, alpha=0.7, label='Load Balancer'),
    mpatches.Patch(color=color_service, alpha=0.7, label='Service'),
    mpatches.Patch(color=color_pod, alpha=0.7, label='Pods'),
    mpatches.Patch(color=color_hpa, alpha=0.7, label='Autoscaler')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-09-mlops/ch33-model-deployment/diagrams/kubernetes_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated: kubernetes_architecture.png")

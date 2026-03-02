"""
Docker Layer Architecture
Visualizes multi-stage Docker build and layer caching strategy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Color palette
color_base = '#607D8B'
color_deps = '#2196F3'
color_code = '#4CAF50'
color_build = '#FF9800'

# LEFT: Multi-Stage Build
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('Multi-Stage Build', fontsize=14, fontweight='bold', pad=20)

# Stage 1: Builder
stage1_y = 7
ax1.add_patch(FancyBboxPatch((0.5, stage1_y), 4, 4, boxstyle="round,pad=0.1",
                              edgecolor=color_build, facecolor=color_build, alpha=0.2, linewidth=2))
ax1.text(2.5, stage1_y + 3.5, 'Stage 1: Builder', ha='center', fontsize=12, fontweight='bold')

# Builder layers
layers_builder = [
    ('Base Image\npython:3.10-slim', color_base),
    ('Install Dependencies\npip install -r requirements.txt', color_deps),
    ('Build Tools\npip, wheel, setuptools', color_build)
]
y_offset = stage1_y + 2.5
for i, (layer, color) in enumerate(layers_builder):
    y = y_offset - i * 0.8
    ax1.add_patch(FancyBboxPatch((0.8, y), 3.4, 0.6, edgecolor=color, facecolor=color, alpha=0.6, linewidth=1.5))
    ax1.text(2.5, y + 0.3, layer, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Stage 2: Runtime
stage2_y = 1
ax1.add_patch(FancyBboxPatch((5.5, stage2_y), 4, 4, boxstyle="round,pad=0.1",
                              edgecolor=color_code, facecolor=color_code, alpha=0.2, linewidth=2))
ax1.text(7.5, stage2_y + 3.5, 'Stage 2: Runtime', ha='center', fontsize=12, fontweight='bold')

# Runtime layers
layers_runtime = [
    ('Base Image\npython:3.10-slim', color_base),
    ('Copy venv\nfrom builder', color_deps),
    ('Application Code\napp.py, model.joblib', color_code)
]
y_offset = stage2_y + 2.5
for i, (layer, color) in enumerate(layers_runtime):
    y = y_offset - i * 0.8
    ax1.add_patch(FancyBboxPatch((5.8, y), 3.4, 0.6, edgecolor=color, facecolor=color, alpha=0.6, linewidth=1.5))
    ax1.text(7.5, y + 0.3, layer, ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Arrow between stages
arrow = FancyArrowPatch((4.5, stage1_y + 2), (5.5, stage2_y + 2),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5,
                        color=color_deps, alpha=0.7)
ax1.add_patch(arrow)
ax1.text(5, 5.5, 'Copy only\nvenv', ha='center', fontsize=9, style='italic', color=color_deps)

# Size annotation
ax1.text(2.5, 0.5, 'Build: ~450 MB', ha='center', fontsize=10, fontweight='bold', color=color_build)
ax1.text(7.5, 0.5, 'Final: ~215 MB', ha='center', fontsize=10, fontweight='bold', color=color_code)

# RIGHT: Layer Caching Strategy
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title('Layer Caching Strategy', fontsize=14, fontweight='bold', pad=20)

# Good caching
good_y = 7
ax2.add_patch(FancyBboxPatch((0.5, good_y), 4, 4, boxstyle="round,pad=0.1",
                              edgecolor='#4CAF50', facecolor='#4CAF50', alpha=0.2, linewidth=2))
ax2.text(2.5, good_y + 3.5, '✓ Optimal Order', ha='center', fontsize=12, fontweight='bold', color='#4CAF50')

good_layers = [
    ('Base Image', color_base, 'Rarely changes'),
    ('COPY requirements.txt', color_deps, 'Changes rarely'),
    ('RUN pip install', color_deps, 'Cached!'),
    ('COPY app.py', color_code, 'Changes often')
]
y_offset = good_y + 2.5
for i, (layer, color, note) in enumerate(good_layers):
    y = y_offset - i * 0.75
    ax2.add_patch(FancyBboxPatch((0.8, y), 2, 0.5, edgecolor=color, facecolor=color, alpha=0.6, linewidth=1.5))
    ax2.text(1.8, y + 0.25, layer, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax2.text(3.3, y + 0.25, note, ha='left', va='center', fontsize=8, style='italic', color='#607D8B')

# Bad caching
bad_y = 1
ax2.add_patch(FancyBboxPatch((5.5, bad_y), 4, 4, boxstyle="round,pad=0.1",
                              edgecolor='#F44336', facecolor='#F44336', alpha=0.2, linewidth=2))
ax2.text(7.5, bad_y + 3.5, '✗ Poor Order', ha='center', fontsize=12, fontweight='bold', color='#F44336')

bad_layers = [
    ('Base Image', color_base, 'Rarely changes'),
    ('COPY . .', color_code, 'Changes often'),
    ('RUN pip install', color_deps, 'Reinstall every time!'),
    ('Run uvicorn', color_code, 'Slow builds')
]
y_offset = bad_y + 2.5
for i, (layer, color, note) in enumerate(bad_layers):
    y = y_offset - i * 0.75
    ax2.add_patch(FancyBboxPatch((5.8, y), 2, 0.5, edgecolor=color, facecolor=color, alpha=0.6, linewidth=1.5))
    ax2.text(6.8, y + 0.25, layer, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax2.text(8.3, y + 0.25, note, ha='left', va='center', fontsize=8, style='italic', color='#607D8B')

# Add rebuild time comparison
ax2.text(2.5, 0.3, 'Rebuild: 5 sec', ha='center', fontsize=11, fontweight='bold', color='#4CAF50')
ax2.text(7.5, 0.3, 'Rebuild: 2+ min', ha='center', fontsize=11, fontweight='bold', color='#F44336')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-09-mlops/ch33-model-deployment/diagrams/docker_layers.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated: docker_layers.png")

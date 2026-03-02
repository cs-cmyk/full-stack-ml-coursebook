#!/usr/bin/env python3
"""
Generate all diagrams for Chapter 47: Advanced Vision Tasks
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
import os

# Ensure diagrams directory exists
os.makedirs('diagrams', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Color palette for consistency
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

print("Generating Chapter 47 diagrams...")

# ============================================================================
# Diagram 1: Advanced Vision Overview
# ============================================================================
print("1/10: Generating advanced_vision_overview.png...")

fig = plt.figure(figsize=(16, 12))

# 1. Self-Supervised Learning: MAE masking strategy
ax1 = plt.subplot(3, 3, 1)
patch_size = 8
img_size = 64
n_patches = img_size // patch_size
mask_ratio = 0.75

# Create grid
grid = np.ones((n_patches, n_patches))
# Randomly mask 75% of patches
mask = np.random.rand(n_patches, n_patches) < mask_ratio
grid[mask] = 0

ax1.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1)
ax1.set_title('MAE: 75% Masking Ratio', fontsize=11, fontweight='bold')
ax1.set_xlabel(f'Visible: {(~mask).sum()}/{n_patches**2} patches')
ax1.set_xticks([])
ax1.set_yticks([])
for i in range(n_patches):
    for j in range(n_patches):
        ax1.add_patch(Rectangle((j-0.5, i-0.5), 1, 1,
                                fill=False, edgecolor='black', linewidth=0.5))

# 2. DINO: Student-Teacher architecture
ax2 = plt.subplot(3, 3, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_title('DINO: Self-Distillation', fontsize=11, fontweight='bold')

# Student box
student = FancyBboxPatch((0.5, 1), 3, 2, boxstyle="round,pad=0.1",
                         facecolor='lightblue', edgecolor='blue', linewidth=2)
ax2.add_patch(student)
ax2.text(2, 2, 'Student\nViT', ha='center', va='center', fontsize=9, fontweight='bold')

# Teacher box
teacher = FancyBboxPatch((6.5, 1), 3, 2, boxstyle="round,pad=0.1",
                         facecolor='lightcoral', edgecolor='red', linewidth=2)
ax2.add_patch(teacher)
ax2.text(8, 2, 'Teacher\nViT', ha='center', va='center', fontsize=9, fontweight='bold')

# Momentum update arrow
arrow1 = FancyArrowPatch((3.7, 2.5), (6.3, 2.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='purple')
ax2.add_artist(arrow1)
ax2.text(5, 3, 'Momentum\nUpdate', ha='center', fontsize=8, color='purple')

# Distillation loss arrow
arrow2 = FancyArrowPatch((6.3, 1.5), (3.7, 1.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='green')
ax2.add_artist(arrow2)
ax2.text(5, 0.8, 'Distillation\nLoss', ha='center', fontsize=8, color='green')

# Input crops
ax2.text(2, 4.5, 'Global + Local Crops', ha='center', fontsize=8)
ax2.text(8, 4.5, 'Global Crops Only', ha='center', fontsize=8)

# 3. SAM: Prompting modes
ax3 = plt.subplot(3, 3, 3)
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('SAM: Prompt Types', fontsize=11, fontweight='bold')

# Image encoder
encoder = FancyBboxPatch((1, 6), 2.5, 2, boxstyle="round,pad=0.05",
                         facecolor='lightgray', edgecolor='black', linewidth=1.5)
ax3.add_patch(encoder)
ax3.text(2.25, 7, 'Image\nEncoder', ha='center', va='center', fontsize=8)

# Prompt types
y_pos = [6.5, 5, 3.5]
prompts = ['Points', 'Boxes', 'Text']
colors = ['red', 'blue', 'green']

for i, (prompt, color) in enumerate(zip(prompts, colors)):
    prompt_box = FancyBboxPatch((5, y_pos[i]), 1.5, 0.8,
                               facecolor=color, alpha=0.3,
                               edgecolor=color, linewidth=1.5)
    ax3.add_patch(prompt_box)
    ax3.text(5.75, y_pos[i]+0.4, prompt, ha='center', va='center', fontsize=8)

    # Arrows to decoder
    arrow = FancyArrowPatch((6.6, y_pos[i]+0.4), (8.4, 7),
                           arrowstyle='->', mutation_scale=15,
                           linewidth=1.5, color=color, alpha=0.6)
    ax3.add_artist(arrow)

# Mask decoder
decoder = FancyBboxPatch((8.5, 6), 2.5, 2, boxstyle="round,pad=0.05",
                        facecolor='orange', alpha=0.3,
                        edgecolor='darkorange', linewidth=1.5)
ax3.add_patch(decoder)
ax3.text(9.75, 7, 'Mask\nDecoder', ha='center', va='center', fontsize=8)

# Output
ax3.text(9.75, 4.5, '↓', fontsize=20, ha='center')
ax3.text(9.75, 3.8, 'Segmentation\nMasks', ha='center', fontsize=8, fontweight='bold')

# 4. NeRF: Volume rendering
ax4 = plt.subplot(3, 3, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 8)
ax4.axis('off')
ax4.set_title('NeRF: Volume Rendering', fontsize=11, fontweight='bold')

# Camera
ax4.plot([1, 1], [3.5, 4.5], 'k-', linewidth=3)
ax4.plot([1, 1.5], [4.5, 4.7], 'k-', linewidth=2)
ax4.plot([1, 1.5], [3.5, 3.3], 'k-', linewidth=2)
ax4.text(1, 2.8, 'Camera', ha='center', fontsize=8)

# Ray
ray_x = np.linspace(1, 9, 100)
ray_y = 4 + 0.3 * (ray_x - 1)
ax4.plot(ray_x, ray_y, 'b-', linewidth=2, alpha=0.7)
ax4.text(5, 5.5, 'Camera Ray', fontsize=8, color='blue')

# Sample points along ray
sample_points = np.linspace(2, 8, 6)
for sp in sample_points:
    ax4.plot(sp, 4 + 0.3*(sp-1), 'ro', markersize=6)

# MLP
mlp = FancyBboxPatch((3.5, 0.5), 2.5, 1.5, boxstyle="round,pad=0.05",
                     facecolor='lightgreen', edgecolor='darkgreen', linewidth=1.5)
ax4.add_patch(mlp)
ax4.text(4.75, 1.25, 'MLP\nF_θ', ha='center', va='center', fontsize=9, fontweight='bold')

# Arrow from points to MLP
ax4.annotate('', xy=(4.75, 2.1), xytext=(5, 3.8),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='purple'))
ax4.text(3.2, 2.8, '(x,y,z,θ,φ)', fontsize=7, color='purple')

# Output
ax4.text(4.75, 0.1, '(color, density)', ha='center', fontsize=8, fontweight='bold')

# 5. Medical Imaging: Windowing
ax5 = plt.subplot(3, 3, 5)
hu_range = np.linspace(-1000, 1000, 256)
windows = {
    'Lung': (-1200, 600),
    'Soft Tissue': (40, 400),
    'Bone': (-1400, 2400)
}

colors_map = {'Lung': 'blue', 'Soft Tissue': 'green', 'Bone': 'red'}

for window_name, (center, width) in windows.items():
    wmin = center - width/2
    wmax = center + width/2

    # Create window function
    intensity = np.zeros_like(hu_range)
    mask = (hu_range >= wmin) & (hu_range <= wmax)
    intensity[mask] = (hu_range[mask] - wmin) / width

    ax5.plot(hu_range, intensity, label=window_name,
            linewidth=2, color=colors_map[window_name])

ax5.set_xlabel('Hounsfield Units (HU)', fontsize=9)
ax5.set_ylabel('Display Intensity', fontsize=9)
ax5.set_title('CT Windowing', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax5.text(0, 0.9, 'Water', fontsize=7, ha='center')

# 6. Depth Estimation: Relative vs Metric
ax6 = plt.subplot(3, 3, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 8)
ax6.axis('off')
ax6.set_title('Monocular Depth Estimation', fontsize=11, fontweight='bold')

# Relative depth
ax6.text(2.5, 7, 'Relative Depth', fontsize=9, fontweight='bold', ha='center')
objects = [('Far', 1, 4, 0.7), ('Mid', 2.5, 4.5, 1), ('Near', 4, 5, 1.3)]
for name, x, y, size in objects:
    circle = plt.Circle((x, y), size*0.3, color='blue', alpha=0.3)
    ax6.add_patch(circle)
    ax6.text(x, y-0.8, name, ha='center', fontsize=7)
ax6.text(2.5, 2.5, 'Order: Far < Mid < Near\n(no actual distances)',
        ha='center', fontsize=7, style='italic')

# Metric depth
ax6.text(7.5, 7, 'Metric Depth', fontsize=9, fontweight='bold', ha='center')
objects_metric = [('5.2m', 6, 4, 0.7), ('2.8m', 7.5, 4.5, 1), ('1.1m', 9, 5, 1.3)]
for name, x, y, size in objects_metric:
    circle = plt.Circle((x, y), size*0.3, color='green', alpha=0.3)
    ax6.add_patch(circle)
    ax6.text(x, y-0.8, name, ha='center', fontsize=7, fontweight='bold')
ax6.text(7.5, 2.5, 'Absolute distances\n(requires calibration/stereo)',
        ha='center', fontsize=7, style='italic')

# 7. Document AI Pipeline
ax7 = plt.subplot(3, 3, 7)
ax7.set_xlim(0, 10)
ax7.set_ylim(0, 10)
ax7.axis('off')
ax7.set_title('Document AI Pipeline', fontsize=11, fontweight='bold')

stages = [
    ('Document\nImage', 8.5, 'lightgray'),
    ('Text\nDetection', 6.8, 'lightblue'),
    ('OCR', 5.1, 'lightgreen'),
    ('Layout\nAnalysis', 3.4, 'lightyellow'),
    ('Structured\nOutput', 1.7, 'lightcoral')
]

for i, (stage, y, color) in enumerate(stages):
    box = FancyBboxPatch((1, y-0.4), 8, 1, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax7.add_patch(box)
    ax7.text(5, y+0.1, stage, ha='center', va='center', fontsize=9, fontweight='bold')

    if i < len(stages) - 1:
        arrow = FancyArrowPatch((5, y-0.5), (5, stages[i+1][1]+0.6),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='black')
        ax7.add_artist(arrow)

# 8. Foundation Model vs Task-Specific
ax8 = plt.subplot(3, 3, 8)
categories = ['General\nDomains', 'Specialized\nDomains', 'Low Data\nScenarios', 'High Data\nScenarios']
foundation = [90, 65, 85, 75]
task_specific = [85, 95, 60, 92]

x = np.arange(len(categories))
width = 0.35

bars1 = ax8.bar(x - width/2, foundation, width, label='Foundation Model', color='steelblue', alpha=0.8)
bars2 = ax8.bar(x + width/2, task_specific, width, label='Task-Specific', color='coral', alpha=0.8)

ax8.set_ylabel('Performance (%)', fontsize=9)
ax8.set_title('Foundation vs Task-Specific Models', fontsize=11, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(categories, fontsize=7)
ax8.legend(fontsize=8)
ax8.set_ylim(0, 100)
ax8.grid(axis='y', alpha=0.3)

# 9. Self-Supervised Learning Benefits
ax9 = plt.subplot(3, 3, 9)

# Training paradigms
paradigms = ['Supervised\n(1M labeled)', 'Self-Supervised\n(100M unlabeled)',
             'Combined\n(SSL + finetune)']
data_efficiency = [75, 68, 88]
transfer_quality = [70, 82, 92]

x = np.arange(len(paradigms))
width = 0.35

bars1 = ax9.bar(x - width/2, data_efficiency, width,
               label='Task Performance', color='mediumpurple', alpha=0.8)
bars2 = ax9.bar(x + width/2, transfer_quality, width,
               label='Transfer Quality', color='mediumseagreen', alpha=0.8)

ax9.set_ylabel('Quality Score', fontsize=9)
ax9.set_title('Self-Supervised Learning Benefits', fontsize=11, fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels(paradigms, fontsize=7)
ax9.legend(fontsize=8)
ax9.set_ylim(0, 100)
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('diagrams/advanced_vision_overview.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ advanced_vision_overview.png generated")

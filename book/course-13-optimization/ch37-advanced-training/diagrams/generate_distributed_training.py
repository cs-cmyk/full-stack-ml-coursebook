"""
Generate diagram showing different distributed training strategies
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B',
    'light_blue': '#BBDEFB',
    'light_green': '#C8E6C9',
    'light_orange': '#FFE0B2'
}

fig, axes = plt.subplots(1, 3, figsize=(14, 6))

# ============= Data Parallelism =============
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Data Parallelism', fontsize=13, fontweight='bold', pad=15)

# GPU 1
gpu1_box = FancyBboxPatch((0.5, 5.5), 4, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['blue'],
                         facecolor=colors['light_blue'],
                         linewidth=2, alpha=0.3)
ax.add_patch(gpu1_box)
ax.text(2.5, 8.7, 'GPU 1', fontsize=11, ha='center', fontweight='bold')

# Model replica 1
model1 = Rectangle((1, 7.5), 3, 0.8,
                   edgecolor=colors['blue'], facecolor=colors['blue'],
                   linewidth=1.5, alpha=0.7)
ax.add_patch(model1)
ax.text(2.5, 7.9, 'Model Copy', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# Data batch 1
data1 = Rectangle((1, 6.3), 3, 0.8,
                 edgecolor=colors['green'], facecolor=colors['green'],
                 linewidth=1.5, alpha=0.7)
ax.add_patch(data1)
ax.text(2.5, 6.7, 'Batch 1 (Data)', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# GPU 2
gpu2_box = FancyBboxPatch((5.5, 5.5), 4, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['blue'],
                         facecolor=colors['light_blue'],
                         linewidth=2, alpha=0.3)
ax.add_patch(gpu2_box)
ax.text(7.5, 8.7, 'GPU 2', fontsize=11, ha='center', fontweight='bold')

# Model replica 2
model2 = Rectangle((6, 7.5), 3, 0.8,
                   edgecolor=colors['blue'], facecolor=colors['blue'],
                   linewidth=1.5, alpha=0.7)
ax.add_patch(model2)
ax.text(7.5, 7.9, 'Model Copy', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# Data batch 2
data2 = Rectangle((6, 6.3), 3, 0.8,
                 edgecolor=colors['orange'], facecolor=colors['orange'],
                 linewidth=1.5, alpha=0.7)
ax.add_patch(data2)
ax.text(7.5, 6.7, 'Batch 2 (Data)', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# Gradient sync
sync_box = Rectangle((2, 4.5), 6, 0.6,
                     edgecolor=colors['purple'], facecolor=colors['purple'],
                     linewidth=1.5, alpha=0.5)
ax.add_patch(sync_box)
ax.text(5, 4.8, 'Gradient Synchronization', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# Arrows
arrow1 = FancyArrowPatch((2.5, 5.5), (2.5, 5.1),
                        arrowstyle='->', mutation_scale=15,
                        linewidth=1.5, color=colors['gray'])
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((7.5, 5.5), (7.5, 5.1),
                        arrowstyle='->', mutation_scale=15,
                        linewidth=1.5, color=colors['gray'])
ax.add_patch(arrow2)

# Description
ax.text(5, 3.5, '• Same model, different data\n• Efficient for most use cases\n• Near-linear speedup',
       fontsize=9, ha='center', va='top', linespacing=1.5,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============= Model Parallelism =============
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Model Parallelism', fontsize=13, fontweight='bold', pad=15)

# GPU 1
gpu1_box = FancyBboxPatch((0.5, 5.5), 4, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['blue'],
                         facecolor=colors['light_blue'],
                         linewidth=2, alpha=0.3)
ax.add_patch(gpu1_box)
ax.text(2.5, 8.7, 'GPU 1', fontsize=11, ha='center', fontweight='bold')

# Model part 1
model1a = Rectangle((1, 7.5), 3, 0.5,
                   edgecolor=colors['blue'], facecolor=colors['blue'],
                   linewidth=1.5, alpha=0.7)
ax.add_patch(model1a)
ax.text(2.5, 7.75, 'Layers 1-3', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# Data
data = Rectangle((1, 6.5), 3, 0.6,
                edgecolor=colors['green'], facecolor=colors['green'],
                linewidth=1.5, alpha=0.7)
ax.add_patch(data)
ax.text(2.5, 6.8, 'Same Data', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# GPU 2
gpu2_box = FancyBboxPatch((5.5, 5.5), 4, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['blue'],
                         facecolor=colors['light_blue'],
                         linewidth=2, alpha=0.3)
ax.add_patch(gpu2_box)
ax.text(7.5, 8.7, 'GPU 2', fontsize=11, ha='center', fontweight='bold')

# Model part 2
model2a = Rectangle((6, 7.5), 3, 0.5,
                   edgecolor=colors['orange'], facecolor=colors['orange'],
                   linewidth=1.5, alpha=0.7)
ax.add_patch(model2a)
ax.text(7.5, 7.75, 'Layers 4-6', fontsize=9, ha='center', va='center', color='white', fontweight='bold')

# Activation passing
arrow_forward = FancyArrowPatch((4, 7.75), (6, 7.75),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color=colors['red'])
ax.add_patch(arrow_forward)
ax.text(5, 8.2, 'Activations →', fontsize=8, ha='center', color=colors['red'], fontweight='bold')

arrow_backward = FancyArrowPatch((6, 7.3), (4, 7.3),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color=colors['purple'])
ax.add_patch(arrow_backward)
ax.text(5, 6.9, '← Gradients', fontsize=8, ha='center', color=colors['purple'], fontweight='bold')

# Description
ax.text(5, 3.5, '• Different layers on different GPUs\n• For models too large for 1 GPU\n• Communication overhead',
       fontsize=9, ha='center', va='top', linespacing=1.5,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============= FSDP (Fully Sharded Data Parallel) =============
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('FSDP (Fully Sharded)', fontsize=13, fontweight='bold', pad=15)

# GPU 1
gpu1_box = FancyBboxPatch((0.5, 5.5), 4, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['blue'],
                         facecolor=colors['light_blue'],
                         linewidth=2, alpha=0.3)
ax.add_patch(gpu1_box)
ax.text(2.5, 8.7, 'GPU 1', fontsize=11, ha='center', fontweight='bold')

# Sharded components 1
params1 = Rectangle((1, 8), 1.2, 0.4,
                   edgecolor=colors['blue'], facecolor=colors['blue'],
                   linewidth=1, alpha=0.7)
ax.add_patch(params1)
ax.text(1.6, 8.2, 'Params\nShard 1', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

grads1 = Rectangle((2.3, 8), 1.2, 0.4,
                  edgecolor=colors['orange'], facecolor=colors['orange'],
                  linewidth=1, alpha=0.7)
ax.add_patch(grads1)
ax.text(2.9, 8.2, 'Grads\nShard 1', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

opt1 = Rectangle((1, 7.3), 1.2, 0.4,
                edgecolor=colors['green'], facecolor=colors['green'],
                linewidth=1, alpha=0.7)
ax.add_patch(opt1)
ax.text(1.6, 7.5, 'Opt State\nShard 1', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

data1 = Rectangle((2.3, 7.3), 1.2, 0.4,
                 edgecolor=colors['purple'], facecolor=colors['purple'],
                 linewidth=1, alpha=0.7)
ax.add_patch(data1)
ax.text(2.9, 7.5, 'Batch 1', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

# GPU 2
gpu2_box = FancyBboxPatch((5.5, 5.5), 4, 3.5,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['blue'],
                         facecolor=colors['light_blue'],
                         linewidth=2, alpha=0.3)
ax.add_patch(gpu2_box)
ax.text(7.5, 8.7, 'GPU 2', fontsize=11, ha='center', fontweight='bold')

# Sharded components 2
params2 = Rectangle((6, 8), 1.2, 0.4,
                   edgecolor=colors['blue'], facecolor=colors['blue'],
                   linewidth=1, alpha=0.7)
ax.add_patch(params2)
ax.text(6.6, 8.2, 'Params\nShard 2', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

grads2 = Rectangle((7.3, 8), 1.2, 0.4,
                  edgecolor=colors['orange'], facecolor=colors['orange'],
                  linewidth=1, alpha=0.7)
ax.add_patch(grads2)
ax.text(7.9, 8.2, 'Grads\nShard 2', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

opt2 = Rectangle((6, 7.3), 1.2, 0.4,
                edgecolor=colors['green'], facecolor=colors['green'],
                linewidth=1, alpha=0.7)
ax.add_patch(opt2)
ax.text(6.6, 7.5, 'Opt State\nShard 2', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

data2 = Rectangle((7.3, 7.3), 1.2, 0.4,
                 edgecolor=colors['purple'], facecolor=colors['purple'],
                 linewidth=1, alpha=0.7)
ax.add_patch(data2)
ax.text(7.9, 7.5, 'Batch 2', fontsize=7, ha='center', va='center', color='white', fontweight='bold')

# All-gather/reduce-scatter
sync_box = Rectangle((2, 6), 6, 0.5,
                     edgecolor=colors['red'], facecolor=colors['red'],
                     linewidth=1.5, alpha=0.4)
ax.add_patch(sync_box)
ax.text(5, 6.25, 'All-Gather / Reduce-Scatter', fontsize=8, ha='center', va='center', color='white', fontweight='bold')

# Arrows
arrow1 = FancyArrowPatch((2.5, 6.9), (2.5, 6.5),
                        arrowstyle='<->', mutation_scale=12,
                        linewidth=1.5, color=colors['gray'])
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((7.5, 6.9), (7.5, 6.5),
                        arrowstyle='<->', mutation_scale=12,
                        linewidth=1.5, color=colors['gray'])
ax.add_patch(arrow2)

# Description
ax.text(5, 3.5, '• Shards params, grads, optimizer\n• Maximum memory efficiency\n• Ideal for billion-parameter models',
       fontsize=9, ha='center', va='top', linespacing=1.5,
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('distributed_training_strategies.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: distributed_training_strategies.png")
plt.close()

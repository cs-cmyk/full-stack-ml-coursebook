"""
Generate a conceptual diagram showing mixed precision training workflow
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 11
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
    'light_orange': '#FFE0B2',
    'light_gray': '#E0E0E0'
}

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Mixed-Precision Training Workflow',
        fontsize=16, fontweight='bold', ha='center', va='top')

# FP32 Master Weights (top)
master_box = FancyBboxPatch((0.5, 7.5), 2.5, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor=colors['blue'],
                            facecolor=colors['light_blue'],
                            linewidth=2)
ax.add_patch(master_box)
ax.text(1.75, 8, 'FP32 Master\nWeights',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Copy to FP16
arrow1 = FancyArrowPatch((1.75, 7.5), (1.75, 6.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=colors['gray'])
ax.add_patch(arrow1)
ax.text(2.5, 7, 'Copy &\nConvert', fontsize=9, ha='left', va='center', style='italic')

# FP16 Weights
fp16_weights = FancyBboxPatch((0.5, 5.5), 2.5, 0.8,
                              boxstyle="round,pad=0.1",
                              edgecolor=colors['orange'],
                              facecolor=colors['light_orange'],
                              linewidth=2)
ax.add_patch(fp16_weights)
ax.text(1.75, 5.9, 'FP16 Weights',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Forward Pass (FP16)
arrow2 = FancyArrowPatch((3, 5.9), (4.5, 5.9),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=colors['gray'])
ax.add_patch(arrow2)
ax.text(3.75, 6.3, 'Forward\n(FP16)', fontsize=9, ha='center', va='bottom', style='italic')

# FP16 Activations & Loss
forward_box = FancyBboxPatch((4.5, 5.5), 2, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor=colors['orange'],
                            facecolor=colors['light_orange'],
                            linewidth=2)
ax.add_patch(forward_box)
ax.text(5.5, 5.9, 'FP16 Loss',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Loss Scaling
arrow3 = FancyArrowPatch((5.5, 5.5), (5.5, 4.5),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=colors['gray'])
ax.add_patch(arrow3)
ax.text(6.3, 5, 'Scale Loss\n(×2048)', fontsize=9, ha='left', va='center',
        style='italic', color=colors['red'])

# Scaled Loss
scaled_box = FancyBboxPatch((4.5, 3.7), 2, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor=colors['red'],
                           facecolor='#FFCCBC',
                           linewidth=2)
ax.add_patch(scaled_box)
ax.text(5.5, 4.1, 'Scaled Loss',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Backward Pass
arrow4 = FancyArrowPatch((4.5, 4.1), (3, 4.1),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=colors['gray'])
ax.add_patch(arrow4)
ax.text(3.75, 4.5, 'Backward\n(FP16)', fontsize=9, ha='center', va='bottom', style='italic')

# FP16 Gradients (Scaled)
grad_box = FancyBboxPatch((0.5, 3.7), 2.5, 0.8,
                         boxstyle="round,pad=0.1",
                         edgecolor=colors['red'],
                         facecolor='#FFCCBC',
                         linewidth=2)
ax.add_patch(grad_box)
ax.text(1.75, 4.1, 'Scaled FP16\nGradients',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Unscale and Convert
arrow5 = FancyArrowPatch((1.75, 3.7), (1.75, 2.7),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=colors['gray'])
ax.add_patch(arrow5)
ax.text(2.8, 3.2, 'Unscale (÷2048)\n& Convert to FP32',
        fontsize=9, ha='left', va='center', style='italic', color=colors['green'])

# FP32 Gradients
fp32_grad = FancyBboxPatch((0.5, 1.9), 2.5, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor=colors['green'],
                          facecolor=colors['light_green'],
                          linewidth=2)
ax.add_patch(fp32_grad)
ax.text(1.75, 2.3, 'FP32 Gradients',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Optimizer Update
arrow6 = FancyArrowPatch((1.75, 1.9), (1.75, 0.9),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=colors['gray'])
ax.add_patch(arrow6)
ax.text(2.5, 1.4, 'Optimizer\nUpdate', fontsize=9, ha='left', va='center', style='italic')

# Updated FP32 Weights
update_box = FancyBboxPatch((0.5, 0.1), 2.5, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor=colors['blue'],
                           facecolor=colors['light_blue'],
                           linewidth=2)
ax.add_patch(update_box)
ax.text(1.75, 0.5, 'Updated FP32\nWeights',
        fontsize=11, ha='center', va='center', fontweight='bold')

# Cycle arrow back
arrow7 = FancyArrowPatch((0.4, 0.5), (0.4, 8),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2, color=colors['purple'],
                        linestyle='--', alpha=0.6,
                        connectionstyle="arc3,rad=-.5")
ax.add_patch(arrow7)

# Add benefits box
benefits_box = FancyBboxPatch((7, 3), 2.8, 4.5,
                             boxstyle="round,pad=0.15",
                             edgecolor=colors['purple'],
                             facecolor='#F3E5F5',
                             linewidth=2)
ax.add_patch(benefits_box)
ax.text(8.4, 7.2, 'Benefits', fontsize=12, ha='center', va='top', fontweight='bold')

benefits_text = """
✓ 1.5-2× Faster Training
  (Tensor Core acceleration)

✓ 40-50% Memory Savings
  (Smaller activations)

✓ Larger Batch Sizes
  (More GPU memory)

✓ Same Final Accuracy
  (FP32 precision preserved)

⚠ Requires GPU Support
  (NVIDIA Tensor Cores)
"""
ax.text(8.4, 6.8, benefits_text, fontsize=9, ha='center', va='top',
        family='monospace', linespacing=1.6)

# Add note
note_text = "Note: Loss scaling prevents gradient underflow in FP16"
ax.text(5, 0.3, note_text, fontsize=9, ha='center', va='center',
        style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('mixed_precision_training.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: mixed_precision_training.png")
plt.close()

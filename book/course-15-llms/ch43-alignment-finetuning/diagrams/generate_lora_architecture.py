"""Generate LoRA architecture comparison diagram"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Full fine-tuning
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Full Fine-Tuning', fontsize=14, fontweight='bold')

# Draw weight matrix
weight_box = FancyBboxPatch((1, 3), 3, 4, boxstyle="round,pad=0.1",
                            edgecolor='#F44336', facecolor='#ffcccc', linewidth=2)
ax1.add_patch(weight_box)
ax1.text(2.5, 5, 'W\n(all trainable)', ha='center', va='center', fontsize=12, fontweight='bold')

# Input and output
ax1.arrow(0.5, 5, 0.3, 0, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax1.text(0.3, 5, 'x', ha='center', va='center', fontsize=12)
ax1.arrow(4.2, 5, 0.3, 0, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax1.text(4.8, 5, 'h', ha='center', va='center', fontsize=12)

ax1.text(5, 1.5, 'Trainable: 124M params', ha='left', va='center', fontsize=11, color='#F44336', fontweight='bold')
ax1.text(5, 0.8, 'Memory: ~500 MB', ha='left', va='center', fontsize=11)

# Right: LoRA
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('LoRA (Low-Rank Adaptation)', fontsize=14, fontweight='bold')

# Frozen weight matrix
frozen_box = FancyBboxPatch((1, 5), 3, 2, boxstyle="round,pad=0.1",
                            edgecolor='#2196F3', facecolor='#cce5ff', linewidth=2)
ax2.add_patch(frozen_box)
ax2.text(2.5, 6, 'W\n(frozen)', ha='center', va='center', fontsize=12, fontweight='bold')

# LoRA matrices A and B
a_box = FancyBboxPatch((1, 2.5), 1.5, 0.5, boxstyle="round,pad=0.05",
                       edgecolor='#4CAF50', facecolor='#ccffcc', linewidth=2)
ax2.add_patch(a_box)
ax2.text(1.75, 2.75, 'A', ha='center', va='center', fontsize=10, fontweight='bold')

b_box = FancyBboxPatch((2.7, 2.5), 1.5, 0.5, boxstyle="round,pad=0.05",
                       edgecolor='#4CAF50', facecolor='#ccffcc', linewidth=2)
ax2.add_patch(b_box)
ax2.text(3.45, 2.75, 'B', ha='center', va='center', fontsize=10, fontweight='bold')

# Input
ax2.arrow(0.5, 6, 0.3, 0, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax2.text(0.3, 6, 'x', ha='center', va='center', fontsize=12)

# Path through W
ax2.arrow(4.2, 6, 0.5, 0, head_width=0.2, head_length=0.1, fc='#2196F3', ec='#2196F3')
ax2.text(4.8, 6.3, 'Wx', ha='center', va='bottom', fontsize=10)

# Path through A and B
arrow1 = FancyArrowPatch((0.8, 5.7), (1, 2.75), arrowstyle='->',
                        connectionstyle='arc3,rad=0.3', color='#4CAF50', linewidth=1.5)
ax2.add_patch(arrow1)
arrow2 = FancyArrowPatch((2.5, 2.75), (4.5, 2.75), arrowstyle='->', color='#4CAF50', linewidth=1.5)
ax2.add_patch(arrow2)
ax2.text(3.5, 2.3, 'BAx', ha='center', va='top', fontsize=10)

# Addition
ax2.text(5, 6, '+', ha='center', va='center', fontsize=16, fontweight='bold')
ax2.text(5, 2.75, '↗', ha='center', va='center', fontsize=16, rotation=45)

# Output
ax2.arrow(5.3, 5, 0.5, 0, head_width=0.3, head_length=0.1, fc='black', ec='black')
ax2.text(6.1, 5, 'h = Wx + BAx', ha='left', va='center', fontsize=11)

ax2.text(5, 1.5, 'Trainable: 295K params', ha='left', va='center', fontsize=11, color='#4CAF50', fontweight='bold')
ax2.text(5, 0.8, 'Memory: ~2 MB', ha='left', va='center', fontsize=11)
ax2.text(5, 0.1, '421× fewer parameters!', ha='left', va='center', fontsize=11,
         fontweight='bold', color='#4CAF50')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-15/ch43/diagrams/lora_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ LoRA architecture diagram saved")

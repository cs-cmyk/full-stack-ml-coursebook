"""Generate visual representation of Stable Diffusion architecture"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
purple = '#9C27B0'
red = '#F44336'
gray = '#607D8B'

# Title
ax.text(5, 11.5, 'Stable Diffusion Architecture', fontsize=20, fontweight='bold',
        ha='center', va='top')

# Input: Text Prompt
box1 = FancyBboxPatch((0.5, 9.5), 2, 1, boxstyle="round,pad=0.1",
                      edgecolor=blue, facecolor='#E3F2FD', linewidth=2)
ax.add_patch(box1)
ax.text(1.5, 10, 'Text Prompt', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(1.5, 9.7, '"A cat wearing', fontsize=8, ha='center', va='center', style='italic')
ax.text(1.5, 9.5, 'a spacesuit"', fontsize=8, ha='center', va='center', style='italic')

# Arrow down
arrow1 = FancyArrowPatch((1.5, 9.4), (1.5, 8.6), arrowstyle='->', lw=2, color=gray,
                         mutation_scale=20)
ax.add_patch(arrow1)

# CLIP Tokenizer
box2 = FancyBboxPatch((0.5, 7.8), 2, 0.7, boxstyle="round,pad=0.05",
                      edgecolor=gray, facecolor='#ECEFF1', linewidth=2)
ax.add_patch(box2)
ax.text(1.5, 8.15, 'CLIP Tokenizer', fontsize=10, ha='center', va='center')

# Arrow down
arrow2 = FancyArrowPatch((1.5, 7.7), (1.5, 7.2), arrowstyle='->', lw=2, color=gray,
                         mutation_scale=20)
ax.add_patch(arrow2)

# Text Encoder
box3 = FancyBboxPatch((0.5, 6.2), 2, 0.9, boxstyle="round,pad=0.1",
                      edgecolor=green, facecolor='#E8F5E9', linewidth=2)
ax.add_patch(box3)
ax.text(1.5, 6.75, 'CLIP Text Encoder', fontsize=10, ha='center', va='center',
        fontweight='bold')
ax.text(1.5, 6.45, '(frozen)', fontsize=8, ha='center', va='center', style='italic')
ax.text(1.5, 6.25, '77 × 768 embeddings', fontsize=8, ha='center', va='center')

# Random Noise (right side)
box4 = FancyBboxPatch((7, 9.5), 2, 1, boxstyle="round,pad=0.1",
                      edgecolor=orange, facecolor='#FFF3E0', linewidth=2)
ax.add_patch(box4)
ax.text(8, 10.2, 'Random Noise', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(8, 9.85, 'Latent:', fontsize=9, ha='center', va='center')
ax.text(8, 9.6, '4 × 64 × 64', fontsize=9, ha='center', va='center')

# Arrow down from noise
arrow3 = FancyArrowPatch((8, 9.4), (8, 8.6), arrowstyle='->', lw=2, color=gray,
                         mutation_scale=20)
ax.add_patch(arrow3)

# U-Net (center-right)
box5 = FancyBboxPatch((6.5, 5.5), 3, 2.9, boxstyle="round,pad=0.1",
                      edgecolor=purple, facecolor='#F3E5F5', linewidth=3)
ax.add_patch(box5)
ax.text(8, 8.1, 'U-Net Denoiser', fontsize=12, ha='center', va='center',
        fontweight='bold', color=purple)
ax.text(8, 7.7, 'with Cross-Attention', fontsize=10, ha='center', va='center')
ax.text(6.7, 7.3, '• Self-attention layers', fontsize=8, ha='left', va='center')
ax.text(6.7, 7.0, '• Cross-attention to text', fontsize=8, ha='left', va='center')
ax.text(6.7, 6.7, '• Iterative denoising', fontsize=8, ha='left', va='center')
ax.text(6.7, 6.3, '• 50-100 steps', fontsize=8, ha='left', va='center')
ax.text(8, 5.9, 'Output: 4 × 64 × 64', fontsize=9, ha='center', va='center',
        style='italic')

# Arrow from text embeddings to U-Net (conditioning)
arrow4 = FancyArrowPatch((2.6, 6.65), (6.4, 7), arrowstyle='->', lw=2.5,
                         color=green, mutation_scale=20, linestyle='dashed')
ax.add_patch(arrow4)
ax.text(4.5, 7.2, 'conditioning', fontsize=9, ha='center', va='center',
        color=green, style='italic')

# Loop arrow for iterative process
from matplotlib.patches import Arc
arc = Arc((9.5, 7), 1.2, 3, angle=0, theta1=270, theta2=90,
          color=purple, linewidth=2, linestyle='--')
ax.add_patch(arc)
ax.text(10.5, 7, 'iterate', fontsize=8, ha='left', va='center',
        color=purple, style='italic', rotation=90)

# Arrow down from U-Net
arrow5 = FancyArrowPatch((8, 5.4), (8, 4.6), arrowstyle='->', lw=2, color=gray,
                         mutation_scale=20)
ax.add_patch(arrow5)

# VAE Decoder
box6 = FancyBboxPatch((6.5, 3.3), 3, 1.2, boxstyle="round,pad=0.1",
                      edgecolor='#009688', facecolor='#E0F2F1', linewidth=2)
ax.add_patch(box6)
ax.text(8, 4.2, 'VAE Decoder', fontsize=11, ha='center', va='center', fontweight='bold')
ax.text(8, 3.85, 'Upsampling:', fontsize=9, ha='center', va='center')
ax.text(8, 3.55, '4×64×64 → 3×512×512', fontsize=9, ha='center', va='center')

# Arrow down
arrow6 = FancyArrowPatch((8, 3.2), (8, 2.4), arrowstyle='->', lw=2, color=gray,
                         mutation_scale=20)
ax.add_patch(arrow6)

# Output Image
box7 = FancyBboxPatch((6.5, 0.8), 3, 1.5, boxstyle="round,pad=0.1",
                      edgecolor=red, facecolor='#FFEBEE', linewidth=2)
ax.add_patch(box7)
ax.text(8, 1.9, 'Generated Image', fontsize=11, ha='center', va='center',
        fontweight='bold')
ax.text(8, 1.55, 'RGB: 3 × 512 × 512', fontsize=9, ha='center', va='center')
ax.text(8, 1.2, 'High-resolution', fontsize=9, ha='center', va='center', style='italic')
ax.text(8, 0.95, 'photorealistic output', fontsize=9, ha='center', va='center',
        style='italic')

# Add key innovation box
innovation_box = FancyBboxPatch((0.5, 0.3), 5, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='#FF6F00', facecolor='#FFF8E1',
                                linewidth=2, linestyle='--')
ax.add_patch(innovation_box)
ax.text(3, 1.3, '💡 Key Innovation', fontsize=11, ha='center', va='center',
        fontweight='bold', color='#FF6F00')
ax.text(3, 0.95, 'Latent Diffusion: Work in compressed latent space', fontsize=9,
        ha='center', va='center')
ax.text(3, 0.7, 'Compression: 512×512×3 ÷ (64×64×4) ≈ 48× reduction', fontsize=9,
        ha='center', va='center')
ax.text(3, 0.45, 'Enables high-res generation on consumer GPUs', fontsize=9,
        ha='center', va='center', style='italic')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch40/diagrams/stable_diffusion_arch.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Stable Diffusion architecture visual saved.")

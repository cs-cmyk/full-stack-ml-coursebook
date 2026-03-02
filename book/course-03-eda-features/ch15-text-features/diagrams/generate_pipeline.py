import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
color_input = '#E8F4F8'
color_process = '#FFF4E6'
color_output = '#E8F5E9'
color_arrow = '#424242'

# Step 1: Raw Text
step1_box = FancyBboxPatch((0.5, 8), 3, 1.2, boxstyle="round,pad=0.1",
                           edgecolor='#0277BD', facecolor=color_input, linewidth=2)
ax.add_patch(step1_box)
ax.text(2, 8.9, 'Raw Text', ha='center', va='top', fontsize=11, weight='bold')
ax.text(2, 8.4, '"The food was great!"', ha='center', va='center',
        fontsize=10, style='italic', family='monospace')

# Arrow 1
arrow1 = FancyArrowPatch((2, 7.8), (2, 7.3), arrowstyle='->',
                        mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow1)

# Step 2: Cleaning & Lowercasing
step2_box = FancyBboxPatch((0.5, 6), 3, 1.2, boxstyle="round,pad=0.1",
                           edgecolor='#F57C00', facecolor=color_process, linewidth=2)
ax.add_patch(step2_box)
ax.text(2, 6.9, 'Clean & Lowercase', ha='center', va='top', fontsize=11, weight='bold')
ax.text(2, 6.4, '"the food was great"', ha='center', va='center',
        fontsize=10, family='monospace')

# Arrow 2
arrow2 = FancyArrowPatch((2, 5.8), (2, 5.3), arrowstyle='->',
                        mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow2)

# Step 3: Tokenization
step3_box = FancyBboxPatch((0.5, 4), 3, 1.2, boxstyle="round,pad=0.1",
                           edgecolor='#F57C00', facecolor=color_process, linewidth=2)
ax.add_patch(step3_box)
ax.text(2, 4.9, 'Tokenization', ha='center', va='top', fontsize=11, weight='bold')
ax.text(2, 4.4, '["the", "food", "was", "great"]', ha='center', va='center',
        fontsize=9, family='monospace')

# Arrow 3
arrow3 = FancyArrowPatch((2, 3.8), (2, 3.3), arrowstyle='->',
                        mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow3)

# Step 4: Stop Word Removal
step4_box = FancyBboxPatch((0.5, 2), 3, 1.2, boxstyle="round,pad=0.1",
                           edgecolor='#F57C00', facecolor=color_process, linewidth=2)
ax.add_patch(step4_box)
ax.text(2, 2.9, 'Remove Stop Words', ha='center', va='top', fontsize=11, weight='bold')
ax.text(2, 2.4, '["food", "great"]', ha='center', va='center',
        fontsize=10, family='monospace')

# Arrow 4
arrow4 = FancyArrowPatch((3.5, 3), (5, 3), arrowstyle='->',
                        mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow4)

# Step 5: Vocabulary
vocab_box = FancyBboxPatch((5.2, 6.5), 2.3, 3, boxstyle="round,pad=0.1",
                           edgecolor='#6A1B9A', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(vocab_box)
ax.text(6.35, 9.2, 'Vocabulary', ha='center', va='top', fontsize=11, weight='bold')
vocab_words = ['amazing', 'bad', 'delicious', 'food', 'good', 'great', 'terrible', 'was']
for i, word in enumerate(vocab_words):
    ax.text(6.35, 8.7 - i*0.28, f'{i}: {word}', ha='center', va='center',
            fontsize=8, family='monospace')

# Arrow 5
arrow5 = FancyArrowPatch((7.5, 7.5), (8.5, 7.5), arrowstyle='->',
                        mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow5)

# Step 6: Vector Representation
vector_box = FancyBboxPatch((5.2, 2), 2.3, 1.5, boxstyle="round,pad=0.1",
                            edgecolor='#2E7D32', facecolor=color_output, linewidth=2)
ax.add_patch(vector_box)
ax.text(6.35, 3.2, 'Count Vector', ha='center', va='top', fontsize=11, weight='bold')
ax.text(6.35, 2.8, '[0, 0, 0, 1,', ha='center', va='center',
        fontsize=9, family='monospace')
ax.text(6.35, 2.5, ' 0, 1, 0, 0]', ha='center', va='center',
        fontsize=9, family='monospace')

# Arrow 6 - connecting to final matrix
arrow6 = FancyArrowPatch((6.35, 5.3), (6.35, 3.6), arrowstyle='->',
                        mutation_scale=20, linewidth=2, color=color_arrow)
ax.add_patch(arrow6)

# Step 7: Feature Matrix X
matrix_box = FancyBboxPatch((8.5, 6), 1.3, 3.5, boxstyle="round,pad=0.1",
                            edgecolor='#2E7D32', facecolor=color_output, linewidth=2)
ax.add_patch(matrix_box)
ax.text(9.15, 9.2, 'Matrix X', ha='center', va='top', fontsize=11, weight='bold')
ax.text(9.15, 8.7, '(n × p)', ha='center', va='center', fontsize=9, style='italic')
ax.text(9.15, 8.2, '[0, 0, 0, 1, ...]', ha='center', va='center',
        fontsize=7, family='monospace')
ax.text(9.15, 7.9, '[1, 0, 0, 0, ...]', ha='center', va='center',
        fontsize=7, family='monospace')
ax.text(9.15, 7.6, '[0, 1, 0, 1, ...]', ha='center', va='center',
        fontsize=7, family='monospace')
ax.text(9.15, 7.3, '[0, 0, 1, 0, ...]', ha='center', va='center',
        fontsize=7, family='monospace')
ax.text(9.15, 7.0, '...', ha='center', va='center', fontsize=9)
ax.text(9.15, 6.6, 'n docs', ha='center', va='center', fontsize=8, style='italic')

# Annotations
ax.text(2, 0.8, 'Each document → fixed-length vector', ha='center', va='center',
        fontsize=10, style='italic', color='#424242')
ax.text(2, 0.4, 'Ready for machine learning!', ha='center', va='center',
        fontsize=10, weight='bold', color='#1B5E20')

# Title
ax.text(5, 9.7, 'Text-to-Numbers Transformation Pipeline', ha='center', va='center',
        fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('text_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Pipeline diagram saved to text_pipeline.png")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Color palette
COLOR_RED = '#F44336'
COLOR_GREEN = '#4CAF50'
COLOR_ORANGE = '#FF9800'
COLOR_BLUE = '#2196F3'

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(6, 9.5, 'Word Error Rate (WER) Calculation', fontsize=16, fontweight='bold', ha='center')

# Example sentences
reference = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
hypothesis = ['the', 'quik', 'brown', 'fox', 'jump', 'over', 'the', 'lazy', 'dog']

# Reference text
ax.text(0.5, 8.5, 'Reference:', fontsize=12, fontweight='bold')
y_ref = 8.0
x_start = 0.5
for i, word in enumerate(reference):
    ax.text(x_start + i * 1.2, y_ref, word, fontsize=11, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_GREEN, alpha=0.2, edgecolor=COLOR_GREEN))

# Hypothesis text
ax.text(0.5, 7.0, 'Hypothesis:', fontsize=12, fontweight='bold')
y_hyp = 6.5
colors = [COLOR_GREEN, COLOR_RED, COLOR_GREEN, COLOR_GREEN, COLOR_RED,
          COLOR_GREEN, COLOR_GREEN, COLOR_GREEN, COLOR_GREEN]
for i, (word, color) in enumerate(zip(hypothesis, colors)):
    alpha = 0.5 if color == COLOR_RED else 0.2
    ax.text(x_start + i * 1.2, y_hyp, word, fontsize=11, ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=alpha, edgecolor=color))

# Draw arrows for errors
# Substitution 1: quick -> quik
ax.annotate('', xy=(x_start + 1 * 1.2 + 0.3, y_hyp + 0.3), xytext=(x_start + 1 * 1.2 + 0.3, y_ref - 0.3),
            arrowprops=dict(arrowstyle='<->', color=COLOR_RED, lw=2))
ax.text(x_start + 1 * 1.2 + 0.7, 7.25, 'S', fontsize=10, color=COLOR_RED, fontweight='bold')

# Substitution 2: jumps -> jump
ax.annotate('', xy=(x_start + 4 * 1.2 + 0.3, y_hyp + 0.3), xytext=(x_start + 4 * 1.2 + 0.3, y_ref - 0.3),
            arrowprops=dict(arrowstyle='<->', color=COLOR_RED, lw=2))
ax.text(x_start + 4 * 1.2 + 0.7, 7.25, 'S', fontsize=10, color=COLOR_RED, fontweight='bold')

# Error types legend
ax.add_patch(FancyBboxPatch((0.5, 4.5), 5, 1.3, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='white', linewidth=2))
ax.text(3, 5.5, 'Error Types', fontsize=12, fontweight='bold', ha='center')
ax.text(1, 5.1, 'S = Substitution (wrong word)', fontsize=10, ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_RED, alpha=0.3))
ax.text(1, 4.7, 'I = Insertion (extra word)', fontsize=10, ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_ORANGE, alpha=0.3))
ax.text(4.2, 4.7, 'D = Deletion (missing word)', fontsize=10, ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLOR_BLUE, alpha=0.3))

# WER Formula
ax.add_patch(FancyBboxPatch((6.5, 4.5), 5, 1.3, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='wheat', alpha=0.5, linewidth=2))
ax.text(9, 5.5, 'WER Formula', fontsize=12, fontweight='bold', ha='center')
ax.text(9, 5.1, r'$\mathrm{WER} = \frac{S + I + D}{N}$', fontsize=14, ha='center')
ax.text(9, 4.7, 'N = total words in reference', fontsize=9, ha='center', style='italic')

# Calculation
ax.add_patch(FancyBboxPatch((0.5, 2.5), 11, 1.7, boxstyle="round,pad=0.1",
                            edgecolor=COLOR_BLUE, facecolor=COLOR_BLUE, alpha=0.1, linewidth=2))
ax.text(6, 4.0, 'Example Calculation', fontsize=12, fontweight='bold', ha='center')

calc_y = 3.5
ax.text(1, calc_y, '• Substitutions (S) = 2', fontsize=11, ha='left')
ax.text(1, calc_y - 0.3, '  ("quick" → "quik", "jumps" → "jump")', fontsize=9, ha='left', style='italic')
ax.text(1, calc_y - 0.7, '• Insertions (I) = 0', fontsize=11, ha='left')
ax.text(1, calc_y - 1.0, '• Deletions (D) = 0', fontsize=11, ha='left')
ax.text(1, calc_y - 1.3, '• Total words (N) = 9', fontsize=11, ha='left')

ax.text(7, calc_y - 0.2, r'$\mathrm{WER} = \frac{2 + 0 + 0}{9} = 0.222$', fontsize=13, ha='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
ax.text(7, calc_y - 0.8, 'WER = 22.22%', fontsize=14, ha='left', fontweight='bold', color=COLOR_RED)

# Performance benchmarks
ax.add_patch(FancyBboxPatch((0.5, 0.2), 11, 2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightgray', alpha=0.2, linewidth=2))
ax.text(6, 2.0, 'Typical WER Benchmarks', fontsize=12, fontweight='bold', ha='center')

bench_y = 1.5
ax.text(1.5, bench_y, '• Human transcriptionists: ~4%', fontsize=10, ha='left')
ax.text(1.5, bench_y - 0.3, '• Whisper Large (clean): 3-5%', fontsize=10, ha='left')
ax.text(1.5, bench_y - 0.6, '• Whisper Base (clean): 8-12%', fontsize=10, ha='left')

ax.text(7, bench_y, '• Challenging conditions: 15-30%', fontsize=10, ha='left')
ax.text(7, bench_y - 0.3, '• Non-adult speech: 30-56%', fontsize=10, ha='left')
ax.text(7, bench_y - 0.6, '• Lower is better ✓', fontsize=10, ha='left', fontweight='bold', color=COLOR_GREEN)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/wer_calculation.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated wer_calculation.png")

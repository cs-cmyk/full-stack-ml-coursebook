import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Color palette
COLOR_BLUE = '#2196F3'
COLOR_GREEN = '#4CAF50'
COLOR_ORANGE = '#FF9800'
COLOR_PURPLE = '#9C27B0'
COLOR_GRAY = '#607D8B'

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Whisper Architecture', fontsize=18, fontweight='bold', ha='center')

# Input audio waveform
ax.add_patch(FancyBboxPatch((0.5, 10), 2, 0.6, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_GRAY, facecolor=COLOR_GRAY, alpha=0.3, linewidth=2))
ax.text(1.5, 10.3, 'Audio\nWaveform', fontsize=11, ha='center', va='center', fontweight='bold')

# Arrow to mel-spectrogram
ax.annotate('', xy=(3.5, 10.3), xytext=(2.5, 10.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Mel-Spectrogram
ax.add_patch(FancyBboxPatch((3.5, 9.7), 2.5, 1.2, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_BLUE, facecolor=COLOR_BLUE, alpha=0.3, linewidth=2))
ax.text(4.75, 10.5, 'Mel-Spectrogram', fontsize=11, ha='center', fontweight='bold')
ax.text(4.75, 10.1, '80 freq bins', fontsize=9, ha='center', style='italic')
ax.text(4.75, 9.9, '(30 sec chunks)', fontsize=9, ha='center', style='italic')

# Arrow to encoder
ax.annotate('', xy=(4.75, 9.2), xytext=(4.75, 9.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Encoder
ax.add_patch(FancyBboxPatch((3, 7.2), 3.5, 2, boxstyle="round,pad=0.1",
                            edgecolor=COLOR_GREEN, facecolor=COLOR_GREEN, alpha=0.3, linewidth=3))
ax.text(4.75, 8.7, 'Transformer Encoder', fontsize=12, ha='center', fontweight='bold')
ax.text(4.75, 8.3, 'Multi-head Self-Attention', fontsize=9, ha='center')
ax.text(4.75, 8.0, 'Feed-Forward Networks', fontsize=9, ha='center')
ax.text(4.75, 7.7, 'Layer Norm', fontsize=9, ha='center')
ax.text(4.75, 7.4, '→ Encoded Audio Features', fontsize=9, ha='center', style='italic')

# Arrow from encoder to decoder (cross-attention)
ax.annotate('', xy=(4.75, 6.7), xytext=(4.75, 7.2),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(5.3, 6.95, 'H (encoded)', fontsize=9, ha='left', style='italic')

# Special tokens (left side)
ax.add_patch(FancyBboxPatch((0.3, 5.5), 2.2, 1, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_ORANGE, facecolor=COLOR_ORANGE, alpha=0.2, linewidth=2))
ax.text(1.4, 6.3, 'Special Tokens', fontsize=10, ha='center', fontweight='bold')
ax.text(1.4, 6.0, '<|startoftranscript|>', fontsize=8, ha='center', family='monospace')
ax.text(1.4, 5.8, '<|en|> (language)', fontsize=8, ha='center', family='monospace')
ax.text(1.4, 5.6, '<|transcribe|>', fontsize=8, ha='center', family='monospace')

# Arrow from special tokens to decoder
ax.annotate('', xy=(3, 6.0), xytext=(2.5, 6.0),
            arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ORANGE))

# Decoder
ax.add_patch(FancyBboxPatch((3, 4.7), 3.5, 2, boxstyle="round,pad=0.1",
                            edgecolor=COLOR_PURPLE, facecolor=COLOR_PURPLE, alpha=0.3, linewidth=3))
ax.text(4.75, 6.2, 'Transformer Decoder', fontsize=12, ha='center', fontweight='bold')
ax.text(4.75, 5.8, 'Masked Self-Attention', fontsize=9, ha='center')
ax.text(4.75, 5.5, 'Cross-Attention to Encoder', fontsize=9, ha='center')
ax.text(4.75, 5.2, 'Feed-Forward Networks', fontsize=9, ha='center')
ax.text(4.75, 4.9, '→ Autoregressive Generation', fontsize=9, ha='center', style='italic')

# Arrow from decoder
ax.annotate('', xy=(4.75, 4.2), xytext=(4.75, 4.7),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Output tokens
ax.add_patch(FancyBboxPatch((3.5, 3.2), 2.5, 1, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_GREEN, facecolor=COLOR_GREEN, alpha=0.3, linewidth=2))
ax.text(4.75, 3.9, 'Output Tokens', fontsize=11, ha='center', fontweight='bold')
ax.text(4.75, 3.5, '"the quick brown fox..."', fontsize=10, ha='center', style='italic')

# Feedback loop (autoregressive)
ax.annotate('', xy=(6.5, 5.5), xytext=(6.5, 3.7),
            arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_PURPLE, linestyle='dashed'))
ax.annotate('', xy=(6.5, 5.5), xytext=(7.2, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_PURPLE, linestyle='dashed'))
ax.annotate('', xy=(3, 5.5), xytext=(7.2, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_PURPLE, linestyle='dashed'))
ax.text(7.5, 4.5, 'Autoregressive\nFeedback', fontsize=9, ha='left',
        style='italic', color=COLOR_PURPLE)

# Add training info box
ax.add_patch(FancyBboxPatch((7, 8.5), 2.7, 2.7, boxstyle="round,pad=0.1",
                            edgecolor=COLOR_GRAY, facecolor='white', alpha=0.9, linewidth=2))
ax.text(8.35, 10.9, 'Training Details', fontsize=11, ha='center', fontweight='bold')
ax.text(8.35, 10.5, '• 680,000 hours', fontsize=9, ha='center')
ax.text(8.35, 10.2, '• 97 languages', fontsize=9, ha='center')
ax.text(8.35, 9.9, '• Weak supervision', fontsize=9, ha='center')
ax.text(8.35, 9.6, '• Multi-task', fontsize=9, ha='center')
ax.text(8.35, 9.3, '  (transcribe/translate)', fontsize=8, ha='center', style='italic')
ax.text(8.35, 9.0, '• 39M - 1550M params', fontsize=9, ha='center')
ax.text(8.35, 8.7, '  (tiny to large)', fontsize=8, ha='center', style='italic')

# Add legend for data flow
ax.text(0.5, 2.5, 'Data Flow:', fontsize=10, fontweight='bold')
ax.text(0.5, 2.2, '1. Audio → Mel-Spectrogram (feature extraction)', fontsize=9)
ax.text(0.5, 1.9, '2. Encoder processes acoustic features', fontsize=9)
ax.text(0.5, 1.6, '3. Decoder generates text autoregressively', fontsize=9)
ax.text(0.5, 1.3, '4. Previous tokens fed back for next prediction', fontsize=9)

# Add model equation
ax.text(0.5, 0.7, r'$p(t_i | t_{<i}, \mathbf{H}) = \mathrm{Decoder}(t_{<i}, \mathbf{H})$',
        fontsize=11, ha='left', style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/whisper_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated whisper_architecture.png")

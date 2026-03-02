import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Color palette
COLOR_BLUE = '#2196F3'
COLOR_GREEN = '#4CAF50'
COLOR_ORANGE = '#FF9800'
COLOR_PURPLE = '#9C27B0'
COLOR_GRAY = '#607D8B'

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'Whisper ASR Pipeline', fontsize=18, fontweight='bold', ha='center')

# Pipeline stages
stages = [
    {'x': 1, 'y': 5, 'w': 1.8, 'h': 1.2, 'color': COLOR_GRAY,
     'title': 'Audio\nInput', 'details': ['Raw waveform', '16 kHz', 'Mono channel']},

    {'x': 3.5, 'y': 5, 'w': 2, 'h': 1.2, 'color': COLOR_BLUE,
     'title': 'Preprocessing', 'details': ['Resample to 16kHz', 'Normalize [-1, 1]', '30s chunks']},

    {'x': 6.2, 'y': 5, 'w': 2, 'h': 1.2, 'color': COLOR_ORANGE,
     'title': 'Feature\nExtraction', 'details': ['STFT', 'Mel filterbank', '80 mel bins']},

    {'x': 9, 'y': 5, 'w': 2, 'h': 1.2, 'color': COLOR_GREEN,
     'title': 'Encoder', 'details': ['Transformer', 'Self-attention', 'Audio features']},

    {'x': 11.7, 'y': 5, 'w': 2, 'h': 1.2, 'color': COLOR_PURPLE,
     'title': 'Decoder', 'details': ['Autoregressive', 'Cross-attention', 'Text generation']},
]

# Draw stages
for stage in stages:
    # Box
    ax.add_patch(FancyBboxPatch((stage['x'], stage['y']), stage['w'], stage['h'],
                                boxstyle="round,pad=0.1", edgecolor=stage['color'],
                                facecolor=stage['color'], alpha=0.3, linewidth=3))

    # Title
    y_text = stage['y'] + stage['h'] - 0.25
    ax.text(stage['x'] + stage['w']/2, y_text, stage['title'],
            fontsize=11, ha='center', fontweight='bold', va='top')

    # Details
    y_detail = stage['y'] + 0.65
    for i, detail in enumerate(stage['details']):
        ax.text(stage['x'] + stage['w']/2, y_detail - i*0.2, f'• {detail}',
                fontsize=8, ha='center', va='center')

# Draw arrows between stages
arrow_y = 5.6
for i in range(len(stages)-1):
    x_start = stages[i]['x'] + stages[i]['w']
    x_end = stages[i+1]['x']
    ax.annotate('', xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Add special tokens input to decoder
ax.add_patch(FancyBboxPatch((11.7, 3.2), 2, 1, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_ORANGE, facecolor=COLOR_ORANGE,
                            alpha=0.2, linewidth=2))
ax.text(12.7, 3.9, 'Task Tokens', fontsize=10, ha='center', fontweight='bold')
ax.text(12.7, 3.6, '<|start|>', fontsize=8, ha='center', family='monospace')
ax.text(12.7, 3.4, '<|en|>', fontsize=8, ha='center', family='monospace')

# Arrow from task tokens to decoder
ax.annotate('', xy=(12.7, 5.0), xytext=(12.7, 4.2),
            arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_ORANGE, linestyle='--'))

# Output
ax.add_patch(FancyBboxPatch((11.7, 1.5), 2, 1, boxstyle="round,pad=0.05",
                            edgecolor=COLOR_GREEN, facecolor=COLOR_GREEN,
                            alpha=0.3, linewidth=2))
ax.text(12.7, 2.2, 'Text Output', fontsize=11, ha='center', fontweight='bold')
ax.text(12.7, 1.9, '"Hello world"', fontsize=9, ha='center', style='italic')

# Arrow from decoder to output
ax.annotate('', xy=(12.7, 2.5), xytext=(12.7, 5.0),
            arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Autoregressive feedback loop
ax.annotate('', xy=(13.8, 5.3), xytext=(13.8, 2.5),
            arrowprops=dict(arrowstyle='->', lw=2, color=COLOR_PURPLE,
                          linestyle='--', connectionstyle="arc3,rad=.3"))
ax.text(13.9, 3.8, 'Token\nfeedback', fontsize=8, ha='left', style='italic',
        color=COLOR_PURPLE)

# Processing metrics box
ax.add_patch(FancyBboxPatch((0.3, 0.3), 5.5, 2.2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightyellow',
                            alpha=0.5, linewidth=2))
ax.text(3.05, 2.3, 'Processing Metrics', fontsize=11, ha='center', fontweight='bold')
ax.text(0.8, 2.0, '• Real-Time Factor (RTF):', fontsize=9, ha='left', fontweight='bold')
ax.text(1.0, 1.75, 'RTF = Processing Time / Audio Duration', fontsize=8, ha='left', style='italic')
ax.text(1.0, 1.5, 'RTF < 1.0 → faster than real-time ✓', fontsize=8, ha='left')

ax.text(0.8, 1.15, '• Typical Performance:', fontsize=9, ha='left', fontweight='bold')
ax.text(1.0, 0.9, 'Tiny: RTF ≈ 0.03 (33× faster)', fontsize=8, ha='left')
ax.text(1.0, 0.65, 'Base: RTF ≈ 0.06 (16× faster)', fontsize=8, ha='left')
ax.text(1.0, 0.4, 'Large: RTF ≈ 1.0 (real-time on GPU)', fontsize=8, ha='left')

# Key features box
ax.add_patch(FancyBboxPatch((6.2, 0.3), 5.5, 2.2, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='lightblue',
                            alpha=0.3, linewidth=2))
ax.text(8.95, 2.3, 'Key Capabilities', fontsize=11, ha='center', fontweight='bold')

features = [
    '✓ 97 languages supported',
    '✓ Automatic language detection',
    '✓ English translation from any language',
    '✓ Word-level timestamps',
    '✓ Robust to noise and accents',
    '✓ No fine-tuning needed'
]

y_feat = 1.95
for feat in features:
    ax.text(6.5, y_feat, feat, fontsize=9, ha='left')
    y_feat -= 0.27

# Add processing time annotation
ax.text(7, 6.8, 'End-to-end latency = preprocessing + encoding + decoding + post-processing',
        fontsize=9, ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/pipeline_flow.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated pipeline_flow.png")

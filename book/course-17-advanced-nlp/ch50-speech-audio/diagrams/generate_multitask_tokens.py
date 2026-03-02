#!/usr/bin/env python3
"""Generate visualization of Whisper's multi-task token system"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))

# Define token sequences for different tasks
tasks = [
    {
        'name': 'English Transcription\n(no timestamps)',
        'tokens': ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|notimestamps|>', 'The', 'weather', 'is', '...'],
        'color': '#2196F3'
    },
    {
        'name': 'Spanish → English\nTranslation',
        'tokens': ['<|startoftranscript|>', '<|es|>', '<|translate|>', '<|notimestamps|>', 'The', 'weather', 'is', '...'],
        'color': '#4CAF50'
    },
    {
        'name': 'English Transcription\n(with timestamps)',
        'tokens': ['<|startoftranscript|>', '<|en|>', '<|transcribe|>', '<|0.00|>', 'The', '<|0.50|>', 'weather', '...'],
        'color': '#FF9800'
    },
    {
        'name': 'Multi-language\nDetection',
        'tokens': ['<|startoftranscript|>', '<|fr|>', '<|transcribe|>', '<|notimestamps|>', 'Le', 'temps', 'est', '...'],
        'color': '#9C27B0'
    }
]

# Token type styling
token_styles = {
    'control': {'facecolor': '#CFD8DC', 'edgecolor': '#607D8B', 'style': 'round,pad=0.1'},
    'language': {'facecolor': '#FFECB3', 'edgecolor': '#FFA000', 'style': 'round,pad=0.1'},
    'task': {'facecolor': '#C8E6C9', 'edgecolor': '#388E3C', 'style': 'round,pad=0.1'},
    'timestamp': {'facecolor': '#E1BEE7', 'edgecolor': '#7B1FA2', 'style': 'round,pad=0.1'},
    'text': {'facecolor': '#FFFFFF', 'edgecolor': '#424242', 'style': 'round,pad=0.1'}
}

def get_token_type(token):
    """Determine token type for styling"""
    if token == '<|startoftranscript|>':
        return 'control'
    elif token.startswith('<|') and len(token) == 6 and token[2:-2].isalpha() and len(token[2:-2]) == 2:
        return 'language'
    elif token in ['<|transcribe|>', '<|translate|>']:
        return 'task'
    elif token == '<|notimestamps|>' or (token.startswith('<|') and '.' in token):
        return 'timestamp'
    else:
        return 'text'

# Layout parameters
y_start = 0.85
y_spacing = 0.20
x_start = 0.05
token_width = 0.10
token_height = 0.06
token_spacing = 0.005

# Draw each task sequence
for i, task in enumerate(tasks):
    y_pos = y_start - i * y_spacing

    # Task label
    ax.text(x_start - 0.02, y_pos, task['name'],
            fontsize=11, fontweight='bold', ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=task['color'],
                     edgecolor='black', linewidth=1.5, alpha=0.3))

    # Draw token sequence
    x_pos = x_start
    for j, token in enumerate(task['tokens']):
        token_type = get_token_type(token)
        style = token_styles[token_type]

        # Adjust width for longer tokens
        if len(token) > 12:
            width = 0.15
        elif len(token) > 8:
            width = 0.12
        else:
            width = token_width

        # Create token box
        box = FancyBboxPatch((x_pos, y_pos - token_height/2), width, token_height,
                            boxstyle=style['style'],
                            facecolor=style['facecolor'],
                            edgecolor=style['edgecolor'],
                            linewidth=1.5,
                            transform=ax.transAxes,
                            zorder=2)
        ax.add_patch(box)

        # Add token text
        ax.text(x_pos + width/2, y_pos, token,
               fontsize=8 if len(token) > 8 else 9,
               ha='center', va='center',
               fontfamily='monospace',
               fontweight='bold' if token_type != 'text' else 'normal',
               transform=ax.transAxes,
               zorder=3)

        x_pos += width + token_spacing

# Add legend for token types
legend_y = 0.05
legend_elements = [
    ('Control Token', token_styles['control']),
    ('Language ID', token_styles['language']),
    ('Task Type', token_styles['task']),
    ('Timestamp', token_styles['timestamp']),
    ('Generated Text', token_styles['text'])
]

legend_x = 0.05
for label, style in legend_elements:
    # Draw small box
    box = FancyBboxPatch((legend_x, legend_y), 0.06, 0.04,
                        boxstyle=style['style'],
                        facecolor=style['facecolor'],
                        edgecolor=style['edgecolor'],
                        linewidth=1.2,
                        transform=ax.transAxes)
    ax.add_patch(box)

    # Add label
    ax.text(legend_x + 0.07, legend_y + 0.02, label,
           fontsize=10, ha='left', va='center',
           transform=ax.transAxes)

    legend_x += 0.18

# Title
ax.text(0.5, 0.95, "Whisper Multi-Task Token System",
        fontsize=15, fontweight='bold', ha='center', va='top',
        transform=ax.transAxes)

# Clean up axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/multitask_tokens.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated multitask_tokens.png")

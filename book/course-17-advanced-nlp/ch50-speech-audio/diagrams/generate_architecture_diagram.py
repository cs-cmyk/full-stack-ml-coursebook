#!/usr/bin/env python3
"""Generate Whisper architecture diagram using matplotlib"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 10))

# Define colors
COLOR_INPUT = '#E3F2FD'
COLOR_ENCODER = '#E8F5E9'
COLOR_TASK = '#FFF3E0'
COLOR_DECODER = '#F3E5F5'
COLOR_OUTPUT = '#FCE4EC'

EDGE_INPUT = '#2196F3'
EDGE_ENCODER = '#4CAF50'
EDGE_TASK = '#FF9800'
EDGE_DECODER = '#9C27B0'
EDGE_OUTPUT = '#F44336'

# Box dimensions
box_width = 0.18
box_height = 0.10
small_box_height = 0.06

# Positions (in axes coordinates)
# Column 1: Input
x1 = 0.08
y_input = [0.75, 0.60, 0.45, 0.30]

# Column 2: Encoder
x2 = 0.33
y_encoder = [0.70, 0.55, 0.40, 0.25, 0.10]

# Column 3: Task specification (sidebar)
x3_task = 0.58
y_task = [0.75, 0.65, 0.55, 0.45]

# Column 4: Decoder
x4 = 0.58
y_decoder = [0.30, 0.20, 0.10]

# Column 5: Output
x5 = 0.83
y_output = [0.30, 0.20, 0.10]

def draw_box(ax, x, y, width, height, text, facecolor, edgecolor, fontsize=10, fontweight='bold'):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle='round,pad=0.01',
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=2.5,
                         transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
           fontsize=fontsize, fontweight=fontweight,
           transform=ax.transAxes)

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', linewidth=2):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color=color,
                           linewidth=linewidth,
                           transform=ax.transAxes,
                           connectionstyle='arc3,rad=0',
                           mutation_scale=20)
    ax.add_patch(arrow)

# Draw Input Processing boxes
input_boxes = [
    ('Raw Audio Waveform\n16 kHz, 30-sec chunks', y_input[0]),
    ('Short-Time Fourier\nTransform (STFT)', y_input[1]),
    ('Mel-Frequency Binning\n80 mel bins', y_input[2]),
    ('Log-Mel Spectrogram\n80 × T\' frames', y_input[3])
]

for text, y in input_boxes:
    draw_box(ax, x1, y, box_width, box_height, text, COLOR_INPUT, EDGE_INPUT, fontsize=9)

# Draw arrows between input boxes
for i in range(len(input_boxes) - 1):
    draw_arrow(ax, x1, y_input[i] - box_height/2, x1, y_input[i+1] + box_height/2, EDGE_INPUT)

# Draw Encoder boxes
encoder_boxes = [
    ('Positional Encoding', y_encoder[0]),
    ('Conv1D + GELU', y_encoder[1]),
    ('Multi-Head\nSelf-Attention × N', y_encoder[2]),
    ('Feed-Forward\nNetworks', y_encoder[3]),
    ('Encoder Output\nH ∈ ℝ^(T\'×d)', y_encoder[4])
]

for text, y in encoder_boxes:
    draw_box(ax, x2, y, box_width, box_height, text, COLOR_ENCODER, EDGE_ENCODER, fontsize=9)

# Draw arrows between encoder boxes
for i in range(len(encoder_boxes) - 1):
    draw_arrow(ax, x2, y_encoder[i] - box_height/2, x2, y_encoder[i+1] + box_height/2, EDGE_ENCODER)

# Arrow from input to encoder
draw_arrow(ax, x1 + box_width/2, y_input[3], x2 - box_width/2, y_encoder[0], EDGE_INPUT)

# Draw Task Specification boxes (smaller)
task_boxes = [
    '<|startoftranscript|>',
    '<|en|> / <|es|> / ...',
    '<|transcribe|> /\n<|translate|>',
    '<|notimestamps|> /\ntimestamp tokens'
]

for i, text in enumerate(task_boxes):
    draw_box(ax, x3_task, y_task[i], 0.15, small_box_height, text,
            COLOR_TASK, EDGE_TASK, fontsize=8, fontweight='normal')

# Draw Decoder boxes
decoder_boxes = [
    ('Cross-Attention\nto Encoder', y_decoder[0]),
    ('Masked Self-Attention\n× N layers', y_decoder[1]),
    ('Feed-Forward Networks\n+ Output Projection', y_decoder[2])
]

for text, y in decoder_boxes:
    draw_box(ax, x4, y, box_width, box_height, text, COLOR_DECODER, EDGE_DECODER, fontsize=9)

# Draw arrows between decoder boxes
for i in range(len(decoder_boxes) - 1):
    draw_arrow(ax, x4, y_decoder[i] - box_height/2, x4, y_decoder[i+1] + box_height/2, EDGE_DECODER)

# Arrow from encoder to decoder
draw_arrow(ax, x2, y_encoder[4], x4, y_decoder[0], EDGE_ENCODER, linewidth=2.5)

# Arrow from task tokens to decoder
draw_arrow(ax, x3_task, y_task[3] - small_box_height, x4, y_decoder[0] + box_height/2,
          EDGE_TASK, linewidth=2)

# Draw Output boxes
output_boxes = [
    ('Autoregressive\nDecoding', y_output[0]),
    ('Beam Search /\nGreedy Sampling', y_output[1]),
    ('Text Tokens\ny₁, y₂, ..., yₗ', y_output[2])
]

for text, y in output_boxes:
    draw_box(ax, x5, y, box_width, box_height, text, COLOR_OUTPUT, EDGE_OUTPUT, fontsize=9)

# Draw arrows between output boxes
for i in range(len(output_boxes) - 1):
    draw_arrow(ax, x5, y_output[i] - box_height/2, x5, y_output[i+1] + box_height/2, EDGE_OUTPUT)

# Arrow from decoder to output
draw_arrow(ax, x4 + box_width/2, y_decoder[2], x5 - box_width/2, y_output[0], EDGE_DECODER)

# Add section labels
section_labels = [
    ('Audio Input\nProcessing', x1, 0.92, COLOR_INPUT, EDGE_INPUT),
    ('Transformer\nEncoder', x2, 0.92, COLOR_ENCODER, EDGE_ENCODER),
    ('Task\nSpecification', x3_task, 0.92, COLOR_TASK, EDGE_TASK),
    ('Transformer\nDecoder', x4, 0.48, COLOR_DECODER, EDGE_DECODER),
    ('Text\nGeneration', x5, 0.48, COLOR_OUTPUT, EDGE_OUTPUT)
]

for text, x, y, facecolor, edgecolor in section_labels:
    ax.text(x, y, text, ha='center', va='center',
           fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=facecolor,
                    edgecolor=edgecolor, linewidth=2),
           transform=ax.transAxes)

# Title
ax.text(0.5, 0.98, 'Whisper Architecture: Encoder-Decoder Transformer',
       ha='center', va='top', fontsize=16, fontweight='bold',
       transform=ax.transAxes)

# Clean up
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/whisper_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated whisper_architecture.png")

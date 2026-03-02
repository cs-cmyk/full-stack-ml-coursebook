#!/usr/bin/env python3
"""Generate WER visualization and error type breakdown"""

import numpy as np
import matplotlib.pyplot as plt

# WER benchmark data
benchmarks = [
    ('Whisper large-v3\n(clean English)', 2.0),
    ('Whisper base\n(clean English)', 5.0),
    ('Whisper tiny\n(clean English)', 10.0),
    ('Human transcribers\n(clean audio)', 4.0),
    ('Whisper large-v3\n(spontaneous)', 8.0),
    ('Whisper large-v3\n(teenagers)', 56.0),
    ('Legacy systems\n(2015)', 15.0),
]

systems = [b[0] for b in benchmarks]
wers = [b[1] for b in benchmarks]

# Define colors based on category
colors_map = {
    0: '#4CAF50',  # large-v3 clean - best
    1: '#2196F3',  # base clean
    2: '#FF9800',  # tiny clean
    3: '#607D8B',  # human
    4: '#FFA726',  # spontaneous
    5: '#F44336',  # teenagers - worst
    6: '#9C27B0',  # legacy
}
colors = [colors_map[i] for i in range(len(systems))]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: WER comparison bar chart
bars = ax1.barh(systems, wers, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
ax1.set_xlabel('Word Error Rate (%)', fontsize=13, fontweight='bold')
ax1.set_title('ASR System Performance Comparison', fontsize=14, fontweight='bold')
ax1.grid(True, axis='x', alpha=0.3, linestyle='--')
ax1.set_xlim(0, 60)

# Add value labels
for i, (system, wer) in enumerate(zip(systems, wers)):
    ax1.text(wer + 1.5, i, f'{wer:.1f}%', va='center', fontsize=11, fontweight='bold')

# Add reference line for human performance
ax1.axvline(x=4.0, color='#607D8B', linestyle=':', linewidth=2, alpha=0.7, label='Human baseline')
ax1.legend(fontsize=11, loc='lower right')
ax1.tick_params(labelsize=10)

# Plot 2: Error type breakdown for a sample transcription
# Example error breakdown
error_types = ['Substitutions', 'Insertions', 'Deletions', 'Correct']
clean_audio = [2, 1, 1, 96]  # 4% WER total
noisy_audio = [12, 5, 8, 75]  # 25% WER total
spontaneous = [25, 10, 15, 50]  # 50% WER total

x = np.arange(len(error_types))
width = 0.25

bars1 = ax2.bar(x - width, clean_audio, width, label='Clean audio (4% WER)',
                color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax2.bar(x, noisy_audio, width, label='Noisy audio (25% WER)',
                color='#FF9800', alpha=0.8, edgecolor='black', linewidth=1.2)
bars3 = ax2.bar(x + width, spontaneous, width, label='Spontaneous (50% WER)',
                color='#F44336', alpha=0.8, edgecolor='black', linewidth=1.2)

ax2.set_ylabel('Percentage of Words', fontsize=13, fontweight='bold')
ax2.set_title('Error Type Breakdown by Condition', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(error_types, fontsize=11)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, 105)
ax2.tick_params(labelsize=11)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 3:  # Only show labels for visible bars
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/wer_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated wer_comparison.png")

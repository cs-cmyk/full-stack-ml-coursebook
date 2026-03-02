import matplotlib.pyplot as plt
import numpy as np

# Color palette
COLOR_BLUE = '#2196F3'
COLOR_GREEN = '#4CAF50'
COLOR_ORANGE = '#FF9800'
COLOR_RED = '#F44336'
COLOR_PURPLE = '#9C27B0'

# Model data
models = ['tiny', 'base', 'small', 'medium', 'large']
params = [39, 74, 244, 769, 1550]  # Millions
relative_speed = [32, 16, 6, 2, 1]  # Relative to large
wer_min = [10, 8, 6, 4, 3]  # Minimum WER %
wer_max = [15, 12, 9, 7, 5]  # Maximum WER %

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Color mapping for models
colors = [COLOR_BLUE, COLOR_GREEN, COLOR_ORANGE, COLOR_RED, COLOR_PURPLE]

# Plot 1: Parameter Count
ax = axes[0, 0]
bars = ax.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
ax.set_title('Model Size (Parameters)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar, val in zip(bars, params):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Relative Speed
ax = axes[0, 1]
bars = ax.bar(models, relative_speed, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Speed (relative to large)', fontsize=12, fontweight='bold')
ax.set_title('Processing Speed', fontsize=14, fontweight='bold')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='Real-time threshold')
ax.grid(axis='y', alpha=0.3)
ax.legend()
# Add value labels
for bar, val in zip(bars, relative_speed):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: WER Range
ax = axes[1, 0]
x = np.arange(len(models))
width = 0.6
# Create error bars showing min-max range
wer_mean = [(wer_min[i] + wer_max[i])/2 for i in range(len(models))]
wer_err = [[wer_mean[i] - wer_min[i] for i in range(len(models))],
           [wer_max[i] - wer_mean[i] for i in range(len(models))]]

bars = ax.bar(x, wer_mean, width, yerr=wer_err, color=colors, alpha=0.7,
              edgecolor='black', linewidth=1.5, capsize=5, error_kw={'linewidth': 2})
ax.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy (WER on Clean Audio)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.grid(axis='y', alpha=0.3)
ax.invert_yaxis()  # Lower is better
ax.text(0.02, 0.98, 'Lower is better ↓', transform=ax.transAxes,
        fontsize=10, fontweight='bold', va='top', color=COLOR_GREEN)

# Plot 4: Speed-Accuracy Tradeoff
ax = axes[1, 1]
# Normalize for visualization
rtf = [1/s for s in relative_speed]  # Convert to RTF (lower is faster)
wer_avg = [(wer_min[i] + wer_max[i])/2 for i in range(len(models))]

for i, (model, r, w, c) in enumerate(zip(models, rtf, wer_avg, colors)):
    ax.scatter(w, r, s=params[i]*3, color=c, alpha=0.6, edgecolors='black', linewidth=2)
    ax.annotate(model, (w, r), xytext=(5, 5), textcoords='offset points',
                fontsize=11, fontweight='bold')

ax.set_xlabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Real-Time Factor', fontsize=12, fontweight='bold')
ax.set_title('Speed-Accuracy Tradeoff', fontsize=14, fontweight='bold')
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.text(ax.get_xlim()[1]*0.7, 1.05, 'Real-time threshold', fontsize=9, style='italic')
ax.grid(True, alpha=0.3)
ax.invert_xaxis()  # Lower WER is better
ax.invert_yaxis()  # Lower RTF is better

# Add annotation for ideal region
ax.add_patch(plt.Rectangle((3, 0), 2, 0.3,
                          facecolor=COLOR_GREEN, alpha=0.1, edgecolor=COLOR_GREEN,
                          linewidth=2, linestyle='--'))
ax.text(4, 0.15, 'Ideal\nRegion', fontsize=9, ha='center', fontweight='bold', color=COLOR_GREEN)

# Circle size legend
ax.text(0.02, 0.98, 'Circle size = Parameters', transform=ax.transAxes,
        fontsize=9, va='top', style='italic', bbox=dict(boxstyle='round',
                                                         facecolor='white', alpha=0.7))

plt.suptitle('Whisper Model Comparison', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-17/ch50/diagrams/model_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Generated model_comparison.png")

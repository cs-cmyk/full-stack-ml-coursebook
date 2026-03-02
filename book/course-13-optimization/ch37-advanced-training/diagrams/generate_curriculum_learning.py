"""
Generate curriculum learning visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

# Set style
plt.style.use('default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Color palette
colors = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============= Left: Curriculum Learning Concept =============
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Curriculum Learning Approach', fontsize=13, fontweight='bold', pad=15)

# Title
ax.text(5, 9.5, 'Training Progression: Easy → Hard', fontsize=12, ha='center', fontweight='bold')

# Phase 1: Easy Examples
phase1_y = 7.5
phase1_box = FancyBboxPatch((0.5, phase1_y - 0.5), 2.5, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor=colors['green'],
                           facecolor='#C8E6C9',
                           linewidth=2.5, alpha=0.7)
ax.add_patch(phase1_box)
ax.text(1.75, phase1_y + 0.6, 'Phase 1', fontsize=11, ha='center', fontweight='bold')
ax.text(1.75, phase1_y + 0.2, 'Easy Examples', fontsize=10, ha='center')
ax.text(1.75, phase1_y - 0.2, '• Low loss\n• Clear labels\n• Simple patterns',
       fontsize=8, ha='center', va='center', linespacing=1.4)

# Arrow to Phase 2
arrow1 = FancyArrowPatch((3, phase1_y), (3.5, phase1_y),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color=colors['gray'])
ax.add_patch(arrow1)
ax.text(3.25, phase1_y + 0.5, 'Progress', fontsize=9, ha='center', style='italic')

# Phase 2: Medium Examples
phase2_y = 7.5
phase2_box = FancyBboxPatch((3.5, phase2_y - 0.5), 2.5, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor=colors['orange'],
                           facecolor='#FFE0B2',
                           linewidth=2.5, alpha=0.7)
ax.add_patch(phase2_box)
ax.text(4.75, phase2_y + 0.6, 'Phase 2', fontsize=11, ha='center', fontweight='bold')
ax.text(4.75, phase2_y + 0.2, 'Medium Examples', fontsize=10, ha='center')
ax.text(4.75, phase2_y - 0.2, '• Moderate loss\n• Some noise\n• Complex patterns',
       fontsize=8, ha='center', va='center', linespacing=1.4)

# Arrow to Phase 3
arrow2 = FancyArrowPatch((6, phase2_y), (6.5, phase2_y),
                        arrowstyle='->', mutation_scale=20,
                        linewidth=2.5, color=colors['gray'])
ax.add_patch(arrow2)
ax.text(6.25, phase2_y + 0.5, 'Progress', fontsize=9, ha='center', style='italic')

# Phase 3: Hard Examples
phase3_y = 7.5
phase3_box = FancyBboxPatch((6.5, phase3_y - 0.5), 2.5, 1.5,
                           boxstyle="round,pad=0.1",
                           edgecolor=colors['red'],
                           facecolor='#FFCDD2',
                           linewidth=2.5, alpha=0.7)
ax.add_patch(phase3_box)
ax.text(7.75, phase3_y + 0.6, 'Phase 3', fontsize=11, ha='center', fontweight='bold')
ax.text(7.75, phase3_y + 0.2, 'Hard Examples', fontsize=10, ha='center')
ax.text(7.75, phase3_y - 0.2, '• High loss\n• Label noise\n• Edge cases',
       fontsize=8, ha='center', va='center', linespacing=1.4)

# Data distribution over time
ax.text(5, 5.5, 'Dataset Usage Over Time', fontsize=12, ha='center', fontweight='bold')

epochs = [1, 5, 10, 15, 20]
y_pos = 4.5

# Draw timeline
ax.plot([1, 9], [y_pos, y_pos], color=colors['gray'], linewidth=2)
for i, epoch in enumerate(epochs):
    x = 1 + (i / (len(epochs)-1)) * 8
    ax.plot([x, x], [y_pos - 0.1, y_pos + 0.1], color=colors['gray'], linewidth=2)
    ax.text(x, y_pos - 0.4, f'Epoch {epoch}', fontsize=8, ha='center')

    # Show data composition
    bar_height = 1.5
    bar_y = y_pos + 0.4

    # Easy (green)
    easy_frac = max(0.5 - i*0.1, 0.2)
    ax.add_patch(Rectangle((x - 0.3, bar_y), 0.6, bar_height * easy_frac,
                           facecolor=colors['green'], edgecolor='black', linewidth=0.5))

    # Medium (orange)
    medium_frac = 0.3 + i*0.05
    ax.add_patch(Rectangle((x - 0.3, bar_y + bar_height * easy_frac), 0.6, bar_height * medium_frac,
                           facecolor=colors['orange'], edgecolor='black', linewidth=0.5))

    # Hard (red)
    hard_frac = 0.2 + i*0.05
    ax.add_patch(Rectangle((x - 0.3, bar_y + bar_height * (easy_frac + medium_frac)),
                           0.6, bar_height * hard_frac,
                           facecolor=colors['red'], edgecolor='black', linewidth=0.5))

# Legend
legend_y = 2.5
ax.add_patch(Rectangle((3, legend_y), 0.4, 0.3, facecolor=colors['green'], edgecolor='black', linewidth=0.5))
ax.text(3.6, legend_y + 0.15, 'Easy Examples', fontsize=9, va='center')

ax.add_patch(Rectangle((3, legend_y - 0.5), 0.4, 0.3, facecolor=colors['orange'], edgecolor='black', linewidth=0.5))
ax.text(3.6, legend_y - 0.35, 'Medium Examples', fontsize=9, va='center')

ax.add_patch(Rectangle((3, legend_y - 1), 0.4, 0.3, facecolor=colors['red'], edgecolor='black', linewidth=0.5))
ax.text(3.6, legend_y - 0.85, 'Hard Examples', fontsize=9, va='center')

# Benefits box
benefits_y = 0.5
ax.text(5, benefits_y + 0.3, 'Benefits:', fontsize=10, ha='center', fontweight='bold')
ax.text(5, benefits_y - 0.1, '✓ Better convergence with noisy labels\n✓ More stable training\n✓ Improved final accuracy',
       fontsize=9, ha='center', va='top', linespacing=1.5,
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# ============= Right: Performance Comparison =============
ax = axes[1]
ax.set_title('Performance: Curriculum vs. Random Training', fontsize=13, fontweight='bold', pad=15)

# Generate simulated training curves
np.random.seed(42)
epochs = np.arange(0, 30)

# Random training - more erratic, lower final accuracy
random_acc = 60 + 30 * (1 - np.exp(-epochs / 15)) + np.random.randn(len(epochs)) * 2
random_acc = np.clip(random_acc, 0, 100)

# Curriculum learning - smoother, higher final accuracy
curriculum_acc = 70 + 28 * (1 - np.exp(-epochs / 12)) + np.random.randn(len(epochs)) * 1
curriculum_acc = np.clip(curriculum_acc, 0, 100)

# Plot curves
ax.plot(epochs, random_acc, label='Random Training',
       linewidth=2.5, color=colors['gray'], alpha=0.8, marker='o', markersize=4, markevery=3)
ax.plot(epochs, curriculum_acc, label='Curriculum Learning',
       linewidth=2.5, color=colors['green'], alpha=0.8, marker='s', markersize=4, markevery=3)

# Add annotations
final_random = random_acc[-1]
final_curriculum = curriculum_acc[-1]

ax.annotate(f'Final: {final_random:.1f}%',
           xy=(epochs[-1], final_random), xytext=(epochs[-5], final_random - 8),
           arrowprops=dict(arrowstyle='->', color=colors['gray'], lw=1.5),
           fontsize=10, color=colors['gray'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.annotate(f'Final: {final_curriculum:.1f}%',
           xy=(epochs[-1], final_curriculum), xytext=(epochs[-5], final_curriculum + 3),
           arrowprops=dict(arrowstyle='->', color=colors['green'], lw=1.5),
           fontsize=10, color=colors['green'], fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Highlight improvement
improvement = final_curriculum - final_random
ax.text(0.5, 0.05, f'Improvement: +{improvement:.1f}%',
       transform=ax.transAxes, fontsize=11, ha='center',
       bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8, edgecolor=colors['green'], linewidth=2))

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([50, 105])

plt.tight_layout()
plt.savefig('curriculum_learning_concept.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: curriculum_learning_concept.png")
plt.close()

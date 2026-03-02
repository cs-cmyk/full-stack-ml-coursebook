"""
Create visualization comparing different NLP approaches
"""
import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('default')

# Create figure with subplots
fig = plt.figure(figsize=(14, 10))

# Color palette
colors = {
    'Random': '#607D8B',
    'TF-IDF': '#FF9800',
    'Few-shot': '#9C27B0',
    'Fine-tuned': '#4CAF50'
}

# Data for comparison
methods = ['Random\nGuessing', 'TF-IDF +\nLogistic Reg', 'Few-shot\nPrompting\n(GPT-2)', 'Fine-tuned\nBERT']
accuracy = [0.50, 0.88, 0.75, 0.92]
data_required = [0, 2000, 5, 2000]
training_time = [0, 1, 0, 15]  # in minutes

# Subplot 1: Accuracy Comparison (Top Left)
ax1 = plt.subplot(2, 2, 1)
bars1 = ax1.bar(methods, accuracy, color=[colors['Random'], colors['TF-IDF'],
                                           colors['Few-shot'], colors['Fine-tuned']],
                alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, accuracy)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('A) Model Performance Comparison', fontsize=13, fontweight='bold', pad=10)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random baseline')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=9)

# Subplot 2: Data Requirements (Top Right)
ax2 = plt.subplot(2, 2, 2)
bars2 = ax2.bar(methods, data_required, color=[colors['Random'], colors['TF-IDF'],
                                                colors['Few-shot'], colors['Fine-tuned']],
                alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, data_required)):
    height = bar.get_height()
    if val > 0:
        label = f'{val}' if val < 100 else f'{val:,}'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 100,
                 label, ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Labeled Examples Required', fontsize=12, fontweight='bold')
ax2.set_title('B) Training Data Requirements', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, 2500)
ax2.grid(axis='y', alpha=0.3)

# Subplot 3: Training Time (Bottom Left)
ax3 = plt.subplot(2, 2, 3)
bars3 = ax3.bar(methods, training_time, color=[colors['Random'], colors['TF-IDF'],
                                                colors['Few-shot'], colors['Fine-tuned']],
                alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars3, training_time)):
    height = bar.get_height()
    if val > 0:
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{val} min', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylabel('Training Time (minutes)', fontsize=12, fontweight='bold')
ax3.set_title('C) Training Time Comparison', fontsize=13, fontweight='bold', pad=10)
ax3.set_ylim(0, 20)
ax3.grid(axis='y', alpha=0.3)

# Subplot 4: Performance vs Data Trade-off (Bottom Right)
ax4 = plt.subplot(2, 2, 4)

# Scatter plot with sizes based on training time
sizes = [100 if t == 0 else t * 20 for t in training_time]

for i, (method, x, y, size) in enumerate(zip(methods, data_required, accuracy, sizes)):
    color = list(colors.values())[i]
    ax4.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=2, zorder=3)

    # Add labels
    clean_method = method.replace('\n', ' ')
    offset_x = 200 if i < 2 else -200
    offset_y = 0.02 if i % 2 == 0 else -0.02
    ax4.annotate(clean_method, (x, y), xytext=(x + offset_x, y + offset_y),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, edgecolor='black'))

ax4.set_xlabel('Labeled Examples Required', fontsize=12, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax4.set_title('D) Performance vs Data Trade-off\n(bubble size = training time)',
              fontsize=13, fontweight='bold', pad=10)
ax4.set_xlim(-200, 2500)
ax4.set_ylim(0.45, 0.95)
ax4.grid(True, alpha=0.3)

# Add diagonal line showing efficiency frontier
ax4.plot([0, 2000], [0.5, 0.92], 'k--', alpha=0.3, linewidth=1, label='Efficiency frontier')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-06-nlp/ch28-modern-nlp/diagrams/performance-comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: performance-comparison.png")

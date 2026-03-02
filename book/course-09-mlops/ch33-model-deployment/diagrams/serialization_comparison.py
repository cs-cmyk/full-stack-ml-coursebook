"""
Serialization Format Comparison
Visualizes performance and size tradeoffs between Pickle, Joblib, and ONNX
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the examples in the chapter
formats = ['Pickle', 'Joblib', 'ONNX']
save_times = [0.45, 2.15, 45.32]  # milliseconds
load_times = [0.32, 1.87, 8.76]   # milliseconds
file_sizes = [987, 985, 1245]      # bytes

# Color palette
colors = ['#2196F3', '#4CAF50', '#FF9800']

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1: Save Time
axes[0].bar(formats, save_times, color=colors, edgecolor='white', linewidth=2)
axes[0].set_ylabel('Time (ms)', fontsize=12)
axes[0].set_title('Serialization (Save) Time', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, max(save_times) * 1.2)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
for i, (fmt, val) in enumerate(zip(formats, save_times)):
    axes[0].text(i, val + max(save_times)*0.03, f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Load Time
axes[1].bar(formats, load_times, color=colors, edgecolor='white', linewidth=2)
axes[1].set_ylabel('Time (ms)', fontsize=12)
axes[1].set_title('Deserialization (Load) Time', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, max(load_times) * 1.2)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')
for i, (fmt, val) in enumerate(zip(formats, load_times)):
    axes[1].text(i, val + max(load_times)*0.03, f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 3: File Size
axes[2].bar(formats, file_sizes, color=colors, edgecolor='white', linewidth=2)
axes[2].set_ylabel('Size (bytes)', fontsize=12)
axes[2].set_title('Serialized File Size', fontsize=13, fontweight='bold')
axes[2].set_ylim(0, max(file_sizes) * 1.2)
axes[2].grid(axis='y', alpha=0.3, linestyle='--')
for i, (fmt, val) in enumerate(zip(formats, file_sizes)):
    axes[2].text(i, val + max(file_sizes)*0.03, f'{val}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add annotations
fig.text(0.5, 0.02, 'Format Tradeoffs: Pickle (fast, Python-only) | Joblib (compressed) | ONNX (cross-platform)',
         ha='center', fontsize=11, style='italic', color='#607D8B')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('/home/chirag/ds-book/book/course-09-mlops/ch33-model-deployment/diagrams/serialization_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated: serialization_comparison.png")

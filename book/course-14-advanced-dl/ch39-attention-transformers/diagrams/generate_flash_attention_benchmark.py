import matplotlib.pyplot as plt
import numpy as np

# Simulate Flash Attention benchmark results
# Typical behavior: speedup increases with sequence length
seq_lengths = np.array([128, 256, 512, 1024, 2048])

# Speedup grows with sequence length (better cache efficiency at larger scales)
# Realistic values based on Flash Attention papers
speedups = np.array([1.5, 2.1, 2.8, 3.5, 4.2])

# Memory usage scales quadratically with sequence length
# Memory = batch_size * num_heads * seq_len^2 * 4 bytes
# For simplicity: batch=8, heads=8, dtype=float32 (4 bytes)
batch_size = 8
num_heads = 8
bytes_per_element = 4
memory_mb = (batch_size * num_heads * seq_lengths**2 * bytes_per_element) / (1024**2)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Color scheme - using consistent palette
color_speedup = '#4CAF50'  # Green
color_memory = '#F44336'  # Red

# Speedup vs sequence length
ax = axes[0]
ax.plot(seq_lengths, speedups, marker='o', linewidth=2.5,
        markersize=10, color=color_speedup)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
           label='No speedup baseline')

# Add value labels on points
for seq_len, speedup in zip(seq_lengths, speedups):
    ax.annotate(f'{speedup:.1f}×',
                xy=(seq_len, speedup), xytext=(0, 8),
                textcoords='offset points', ha='center',
                fontsize=10, fontweight='bold')

ax.set_xlabel('Sequence Length', fontsize=13, fontweight='bold')
ax.set_ylabel('Speedup (×)', fontsize=13, fontweight='bold')
ax.set_title('Flash Attention Speedup vs Sequence Length',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xscale('log', base=2)
ax.set_xticks(seq_lengths)
ax.set_xticklabels([str(s) for s in seq_lengths])
ax.set_ylim(0.5, 5)
ax.legend(fontsize=11, loc='upper left')

# Memory usage vs sequence length
ax = axes[1]
ax.plot(seq_lengths, memory_mb, marker='s', linewidth=2.5,
        markersize=10, color=color_memory)

# Add O(n²) annotation
ax.text(400, 8, r'$O(n^2)$ scaling', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Sequence Length', fontsize=13, fontweight='bold')
ax.set_ylabel('Attention Matrix Memory (MB)', fontsize=13, fontweight='bold')
ax.set_title('Memory Scaling (Standard Attention)',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xticks(seq_lengths)
ax.set_xticklabels([str(s) for s in seq_lengths])

# Add shaded region for problematic memory zone
ax.axhspan(10, 100, alpha=0.15, color='red', label='High memory zone')
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch39/diagrams/flash_attention_benchmark.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: flash_attention_benchmark.png")
plt.close()

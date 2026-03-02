import matplotlib.pyplot as plt
import numpy as np

# Simulate the extrapolation behavior based on typical results
# Training length
train_len = 128
test_lengths = np.array([128, 192, 256, 320, 384])

# Simulated perplexity results based on typical behavior
# RoPE: gradual degradation but maintains reasonable performance
rope_perplexities = np.array([845, 891, 948, 1012, 1089])

# ALiBi: more stable extrapolation due to linear biases
alibi_perplexities = np.array([862, 883, 903, 925, 950])

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Color scheme - using consistent palette
color_rope = '#2196F3'  # Blue
color_alibi = '#4CAF50'  # Green

ax.plot(test_lengths, rope_perplexities, marker='o', linewidth=2.5,
        label='RoPE', color=color_rope, markersize=8)
ax.plot(test_lengths, alibi_perplexities, marker='s', linewidth=2.5,
        label='ALiBi', color=color_alibi, markersize=8)

# Mark training length
ax.axvline(train_len, color='#F44336', linestyle='--', alpha=0.7,
           linewidth=2, label='Training length')

# Add shaded region for extrapolation zone
ax.axvspan(train_len, test_lengths[-1], alpha=0.1, color='gray',
           label='Extrapolation zone')

ax.set_xlabel('Sequence Length', fontsize=13, fontweight='bold')
ax.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
ax.set_title('Positional Encoding Extrapolation Performance',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(100, 400)
ax.set_ylim(800, 1150)

# Add annotations
ax.annotate('Better extrapolation →',
            xy=(300, 950), fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-14/ch39/diagrams/positional_encoding_extrapolation.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: positional_encoding_extrapolation.png")
plt.close()

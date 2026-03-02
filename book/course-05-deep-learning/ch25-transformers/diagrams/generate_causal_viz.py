#!/usr/bin/env python3
"""Generate causal vs bidirectional attention visualization"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

print("Generating causal_vs_bidirectional_attention.png...")

def create_causal_mask(seq_len):
    """Create lower-triangular causal mask"""
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    return mask

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention with optional masking"""
    d_k = Q.size(-1)
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)
    output = attention_weights @ V
    return output, attention_weights

# Test with example
seq_len = 8
d_k = 16

Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

# Bidirectional (no mask)
output_bidir, weights_bidir = scaled_dot_product_attention(Q, K, V, mask=None)

# Causal (with mask)
causal_mask = create_causal_mask(seq_len)
output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

# Visualize both
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bidirectional
sns.heatmap(weights_bidir.numpy(), annot=True, fmt='.2f', cmap='Blues',
            ax=axes[0], cbar_kws={'label': 'Weight'},
            xticklabels=[f'{i}' for i in range(seq_len)],
            yticklabels=[f'{i}' for i in range(seq_len)],
            vmin=0, vmax=weights_bidir.max().item())
axes[0].set_title('Bidirectional Attention (No Mask)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Key Position', fontsize=11)
axes[0].set_ylabel('Query Position', fontsize=11)

# Causal
# Replace NaN values (from -inf before softmax) with 0 for visualization
weights_causal_viz = weights_causal.numpy()
weights_causal_viz = np.nan_to_num(weights_causal_viz, 0)

sns.heatmap(weights_causal_viz, annot=True, fmt='.2f', cmap='Oranges',
            ax=axes[1], cbar_kws={'label': 'Weight'},
            xticklabels=[f'{i}' for i in range(seq_len)],
            yticklabels=[f'{i}' for i in range(seq_len)],
            vmin=0, vmax=weights_causal_viz.max())
axes[1].set_title('Causal Attention (Masked)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Key Position', fontsize=11)
axes[1].set_ylabel('Query Position', fontsize=11)

# Add note
fig.text(0.5, 0.02, 'Note: In causal attention, position i can only attend to positions ≤ i (lower triangular)',
         ha='center', fontsize=10, style='italic')

plt.suptitle('Causal vs Bidirectional Attention Patterns', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch25-transformers/diagrams/causal_vs_bidirectional_attention.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated: causal_vs_bidirectional_attention.png")

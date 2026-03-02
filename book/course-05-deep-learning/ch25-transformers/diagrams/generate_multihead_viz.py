#!/usr/bin/env python3
"""Generate multihead attention patterns visualization"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print("Generating multihead_attention_patterns.png...")

# Simple MultiHeadAttention class (minimal version)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)
        attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attended_values)
        return output, attention_weights

# Create sample input
batch_size = 2
seq_len = 6
d_model = 64
num_heads = 4

x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
output, attention_weights = mha(x)

# Visualize attention patterns from different heads
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
token_labels = ['The', 'cat', 'sat', 'on', 'the', 'mat']

for head_idx in range(4):
    ax = axes[head_idx // 2, head_idx % 2]
    attn = attention_weights[0, head_idx].detach().numpy()

    sns.heatmap(attn, annot=True, fmt='.2f', cmap='viridis',
                ax=ax, cbar_kws={'label': 'Weight'},
                xticklabels=token_labels, yticklabels=token_labels,
                vmin=0, vmax=attn.max())
    ax.set_title(f'Head {head_idx}: Attention Pattern', fontsize=12, fontweight='bold')
    ax.set_xlabel('Attending to', fontsize=11)
    ax.set_ylabel('Attending from', fontsize=11)

plt.suptitle('Multi-Head Attention: Different Heads Learn Different Patterns',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch25-transformers/diagrams/multihead_attention_patterns.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated: multihead_attention_patterns.png")

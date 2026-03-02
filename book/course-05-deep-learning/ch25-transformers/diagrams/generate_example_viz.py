#!/usr/bin/env python3
"""Generate example visualizations for the code examples"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Color palette
sns.set_palette(["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#607D8B"])

print("Generating attention_weights_heatmap.png...")

# 1. Generate attention_weights_heatmap.png (from Example 1)
seq_len = 4
d_k = 8

# Generate random Q, K, V matrices
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

# Compute attention
scores = Q @ K.T
scaled_scores = scores / np.sqrt(d_k)
attention_weights = F.softmax(scaled_scores, dim=-1)

# Visualize
plt.figure(figsize=(6, 5))
sns.heatmap(attention_weights.numpy(), annot=True, fmt='.3f',
            cmap='Blues', cbar_kws={'label': 'Attention Weight'},
            xticklabels=[f'Token {i}' for i in range(seq_len)],
            yticklabels=[f'Token {i}' for i in range(seq_len)])
plt.title('Attention Weight Matrix\n(rows = queries, columns = keys)', fontsize=14, fontweight='bold')
plt.xlabel('Attending to (Key)', fontsize=12)
plt.ylabel('Attending from (Query)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-05-deep-learning/ch25-transformers/diagrams/attention_weights_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated: attention_weights_heatmap.png")

plt.close()

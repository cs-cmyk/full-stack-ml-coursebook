"""
Code Review Test Script for Chapter 39: Attention and Transformer Variants
Tests all code blocks in order to verify they execute correctly.
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_block(block_num, description, test_func):
    """Test a code block and report results."""
    print(f"\n{'='*60}")
    print(f"Block {block_num}: {description}")
    print('='*60)
    try:
        test_func()
        print(f"✓ Block {block_num} passed")
        return True
    except Exception as e:
        print(f"✗ Block {block_num} failed")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False

# Track results
results = []

# Block 1: Visualization - MHA, MQA, GQA Comparison
def block1():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # Create figure comparing MHA, MQA, and GQA architectures
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common parameters
    num_heads = 8
    head_height = 0.8
    head_spacing = 1.0

    # Color scheme
    color_q = '#3498db'  # Blue for queries
    color_k = '#e74c3c'  # Red for keys
    color_v = '#2ecc71'  # Green for values

    def draw_projections(ax, title, kv_config):
        """
        Draw Q, K, V projections for different attention variants.
        kv_config: list of (head_idx, kv_group_idx) tuples
        """
        ax.set_xlim(0, 4)
        ax.set_ylim(0, num_heads * head_spacing)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Draw query heads (always independent)
        for i in range(num_heads):
            y = i * head_spacing + 0.1
            rect = mpatches.Rectangle((0.2, y), 0.6, head_height,
                                       facecolor=color_q, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(0.5, y + head_height/2, f'Q{i}', ha='center', va='center',
                    fontsize=10, fontweight='bold')

        # Draw key and value projections based on config
        unique_kv = sorted(set(group_idx for _, group_idx in kv_config))
        kv_colors_k = {group_idx: color_k for group_idx in unique_kv}
        kv_colors_v = {group_idx: color_v for group_idx in unique_kv}

        for i, (head_idx, kv_group_idx) in enumerate(kv_config):
            y = head_idx * head_spacing + 0.1

            # Draw key
            rect_k = mpatches.Rectangle((1.5, y), 0.6, head_height,
                                         facecolor=kv_colors_k[kv_group_idx],
                                         edgecolor='black', linewidth=2,
                                         alpha=0.7)
            ax.add_patch(rect_k)
            ax.text(1.8, y + head_height/2, f'K{kv_group_idx}', ha='center', va='center',
                    fontsize=10, fontweight='bold')

            # Draw value
            rect_v = mpatches.Rectangle((2.8, y), 0.6, head_height,
                                         facecolor=kv_colors_v[kv_group_idx],
                                         edgecolor='black', linewidth=2,
                                         alpha=0.7)
            ax.add_patch(rect_v)
            ax.text(3.1, y + head_height/2, f'V{kv_group_idx}', ha='center', va='center',
                    fontsize=10, fontweight='bold')

            # Draw connection line
            ax.plot([0.8, 1.5], [y + head_height/2, y + head_height/2],
                    'k-', linewidth=1, alpha=0.3)
            ax.plot([2.2, 2.8], [y + head_height/2, y + head_height/2],
                    'k-', linewidth=1, alpha=0.3)

        # Add labels
        ax.text(0.5, -0.5, 'Queries', ha='center', fontsize=11, fontweight='bold')
        ax.text(1.8, -0.5, 'Keys', ha='center', fontsize=11, fontweight='bold')
        ax.text(3.1, -0.5, 'Values', ha='center', fontsize=11, fontweight='bold')

    # Multi-Head Attention: Each head has its own K, V
    mha_config = [(i, i) for i in range(num_heads)]
    draw_projections(axes[0], 'Multi-Head Attention (MHA)', mha_config)

    # Multi-Query Attention: All heads share single K, V
    mqa_config = [(i, 0) for i in range(num_heads)]
    draw_projections(axes[1], 'Multi-Query Attention (MQA)', mqa_config)

    # Grouped-Query Attention: 8 heads, 2 groups
    gqa_config = [(i, i // 4) for i in range(num_heads)]
    draw_projections(axes[2], 'Grouped-Query Attention (GQA)\n8 heads, 2 groups', gqa_config)

    plt.tight_layout()
    plt.savefig('attention_variants_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print parameter comparison
    d_model = 512
    d_k = 64
    num_heads = 8

    print("Parameter Count Comparison:")
    print(f"Model dimension: {d_model}, Head dimension: {d_k}, Num heads: {num_heads}\n")

    mha_params = num_heads * (3 * d_model * d_k) + d_model * (num_heads * d_k)
    mqa_params = num_heads * (d_model * d_k) + 2 * (d_model * d_k) + d_model * (num_heads * d_k)
    gqa_params_2 = num_heads * (d_model * d_k) + 2 * 2 * (d_model * d_k) + d_model * (num_heads * d_k)

    print(f"MHA:       {mha_params:,} parameters")
    print(f"MQA:       {mqa_params:,} parameters ({mqa_params/mha_params:.1%} of MHA)")
    print(f"GQA (g=2): {gqa_params_2:,} parameters ({gqa_params_2/mha_params:.1%} of MHA)")

results.append(test_block(1, "Visualization - MHA/MQA/GQA Comparison", block1))

# Block 2: Multi-Head Attention and Multi-Query Attention Implementation
def block2():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    import numpy as np

    # Set random seed for reproducibility
    torch.manual_seed(42)

    class MultiHeadAttention(nn.Module):
        """Standard Multi-Head Attention with independent K, V per head."""

        def __init__(self, d_model, num_heads):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            # Separate Q, K, V projections for each head
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, query, key, value, mask=None, use_cache=False, past_kv=None):
            batch_size = query.size(0)

            # Project and reshape: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # KV caching for autoregressive inference
            if use_cache:
                if past_kv is not None:
                    past_k, past_v = past_kv
                    K = torch.cat([past_k, K], dim=2)
                    V = torch.cat([past_v, V], dim=2)
                current_kv = (K, V)
            else:
                current_kv = None

            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)

            # Concatenate heads and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(attn_output)

            return output, current_kv

    class MultiQueryAttention(nn.Module):
        """Multi-Query Attention with shared K, V across all query heads."""

        def __init__(self, d_model, num_heads):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            # Multiple query projections, single K and V projection
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, self.d_k)  # Single head dimension
            self.W_v = nn.Linear(d_model, self.d_k)  # Single head dimension
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, query, key, value, mask=None, use_cache=False, past_kv=None):
            batch_size = query.size(0)

            # Project queries: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # Project single K, V: (batch, seq_len, d_k) -> (batch, 1, seq_len, d_k)
            K = self.W_k(key).unsqueeze(1)
            V = self.W_v(value).unsqueeze(1)

            # Expand K, V to match number of query heads
            K = K.expand(-1, self.num_heads, -1, -1)
            V = V.expand(-1, self.num_heads, -1, -1)

            # KV caching
            if use_cache:
                if past_kv is not None:
                    past_k, past_v = past_kv
                    K = torch.cat([past_k, K], dim=2)
                    V = torch.cat([past_v, V], dim=2)
                current_kv = (K, V)
            else:
                current_kv = None

            # Scaled dot-product attention (identical to MHA)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)

            # Concatenate and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(attn_output)

            return output, current_kv

    # Benchmark inference with KV caching
    def benchmark_inference(model, seq_length, d_model, num_steps=100, device='cpu'):
        """Simulate autoregressive generation with KV caching."""
        model.eval()
        model.to(device)

        # Initial prompt
        x = torch.randn(1, seq_length, d_model, device=device)

        times = []
        past_kv = None

        with torch.no_grad():
            for step in range(num_steps):
                # In autoregressive mode, we only process one new token at a time
                if step == 0:
                    # First step: process full prompt
                    input_x = x
                else:
                    # Subsequent steps: process single token
                    input_x = torch.randn(1, 1, d_model, device=device)

                start = time.time()
                output, past_kv = model(input_x, input_x, input_x, use_cache=True, past_kv=past_kv)
                torch.cuda.synchronize() if device == 'cuda' else None
                times.append(time.time() - start)

        return np.mean(times[10:])  # Skip warmup

    # Compare MHA vs MQA
    d_model = 512
    num_heads = 8
    seq_length = 512

    print("=" * 60)
    print("MHA vs MQA Comparison")
    print("=" * 60)

    # Parameter count comparison
    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)

    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())

    print(f"\nParameter Counts:")
    print(f"MHA: {mha_params:,} parameters")
    print(f"MQA: {mqa_params:,} parameters")
    print(f"Reduction: {(1 - mqa_params/mha_params)*100:.1f}%")

    # KV cache size comparison
    # MHA: num_heads * d_k dimensions per token
    # MQA: d_k dimensions per token (shared across heads)
    kv_cache_mha = 2 * seq_length * num_heads * (d_model // num_heads) * 4  # 4 bytes per float32
    kv_cache_mqa = 2 * seq_length * (d_model // num_heads) * 4

    print(f"\nKV Cache Memory (for {seq_length} tokens):")
    print(f"MHA: {kv_cache_mha / 1024:.2f} KB")
    print(f"MQA: {kv_cache_mqa / 1024:.2f} KB")
    print(f"Reduction: {(1 - kv_cache_mqa/kv_cache_mha)*100:.1f}%")

    # Inference speed comparison
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nInference Benchmark ({device}):")

    mha_time = benchmark_inference(mha, seq_length, d_model, num_steps=50, device=device)
    mqa_time = benchmark_inference(mqa, seq_length, d_model, num_steps=50, device=device)

    print(f"MHA: {mha_time*1000:.3f} ms/token")
    print(f"MQA: {mqa_time*1000:.3f} ms/token")
    print(f"Speedup: {mha_time/mqa_time:.2f}x")

results.append(test_block(2, "MHA and MQA Implementation", block2))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Blocks passed: {sum(results)}/{len(results)}")
print(f"Blocks failed: {len(results) - sum(results)}/{len(results)}")

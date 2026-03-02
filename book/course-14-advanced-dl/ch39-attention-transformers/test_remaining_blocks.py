"""
Test remaining code blocks from Chapter 39
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

results = []

# Block 3: Grouped-Query Attention
def block3():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    torch.manual_seed(42)

    class MultiHeadAttention(nn.Module):
        """Standard Multi-Head Attention with independent K, V per head."""

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

        def forward(self, query, key, value, mask=None, use_cache=False, past_kv=None):
            batch_size = query.size(0)
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            if use_cache:
                if past_kv is not None:
                    past_k, past_v = past_kv
                    K = torch.cat([past_k, K], dim=2)
                    V = torch.cat([past_v, V], dim=2)
                current_kv = (K, V)
            else:
                current_kv = None

            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
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

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, self.d_k)
            self.W_v = nn.Linear(d_model, self.d_k)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, query, key, value, mask=None, use_cache=False, past_kv=None):
            batch_size = query.size(0)
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).unsqueeze(1)
            V = self.W_v(value).unsqueeze(1)
            K = K.expand(-1, self.num_heads, -1, -1)
            V = V.expand(-1, self.num_heads, -1, -1)

            if use_cache:
                if past_kv is not None:
                    past_k, past_v = past_kv
                    K = torch.cat([past_k, K], dim=2)
                    V = torch.cat([past_v, V], dim=2)
                current_kv = (K, V)
            else:
                current_kv = None

            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(attn_output)

            return output, current_kv

    class GroupedQueryAttention(nn.Module):
        """Grouped-Query Attention: Middle ground between MHA and MQA."""

        def __init__(self, d_model, num_heads, num_kv_groups):
            super().__init__()
            assert d_model % num_heads == 0
            assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

            self.d_model = d_model
            self.num_heads = num_heads
            self.num_kv_groups = num_kv_groups
            self.d_k = d_model // num_heads
            self.heads_per_group = num_heads // num_kv_groups

            # Query projection for all heads
            self.W_q = nn.Linear(d_model, d_model)

            # Key and value projections for each group
            self.W_k = nn.Linear(d_model, num_kv_groups * self.d_k)
            self.W_v = nn.Linear(d_model, num_kv_groups * self.d_k)
            self.W_o = nn.Linear(d_model, d_model)

        def forward(self, query, key, value, mask=None, use_cache=False, past_kv=None):
            batch_size = query.size(0)

            # Project queries: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # Project keys and values: (batch, seq_len, num_kv_groups * d_k)
            K = self.W_k(key).view(batch_size, -1, self.num_kv_groups, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.num_kv_groups, self.d_k).transpose(1, 2)

            # Expand K, V to match query heads by repeating each group
            # (batch, num_kv_groups, seq_len, d_k) -> (batch, num_heads, seq_len, d_k)
            K = K.repeat_interleave(self.heads_per_group, dim=1)
            V = V.repeat_interleave(self.heads_per_group, dim=1)

            # KV caching
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

            # Concatenate and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(attn_output)

            return output, current_kv

    # Compare MHA, MQA, and GQA with different group counts
    d_model = 512
    num_heads = 8

    print("\n" + "=" * 60)
    print("GQA: Finding the Sweet Spot")
    print("=" * 60)

    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)

    models = {
        'MHA (g=8)': mha,
        'GQA (g=4)': GroupedQueryAttention(d_model, num_heads, num_kv_groups=4),
        'GQA (g=2)': GroupedQueryAttention(d_model, num_heads, num_kv_groups=2),
        'MQA (g=1)': mqa
    }

    results_list = []

    for name, model in models.items():
        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        # Calculate KV cache size per token
        if 'MHA' in name:
            kv_size = 2 * num_heads * (d_model // num_heads)
        elif 'MQA' in name:
            kv_size = 2 * (d_model // num_heads)
        else:  # GQA
            num_groups = int(name.split('=')[1].strip(')'))
            kv_size = 2 * num_groups * (d_model // num_heads)

        results_list.append({
            'Model': name,
            'Parameters': params,
            'KV Cache/Token': kv_size * 4,  # bytes (float32)
            'Relative Params': params / results_list[0]['Parameters'] if results_list else 1.0,
            'Relative KV': (kv_size * 4) / (2 * num_heads * (d_model // num_heads) * 4)
        })

    print(f"\n{'Model':<15} {'Parameters':>12} {'KV/Token':>12} {'Param %':>10} {'KV %':>10}")
    print("-" * 60)
    for r in results_list:
        print(f"{r['Model']:<15} {r['Parameters']:>12,} {r['KV Cache/Token']:>10}B "
              f"{r['Relative Params']:>9.1%} {r['Relative KV']:>9.1%}")

    print("\nInterpretation:")
    print("- GQA (g=2) uses only 67.5% of MHA parameters and 25% of KV cache")
    print("- In practice, GQA (g=2 or g=4) matches MHA quality with 2-3x inference speedup")
    print("- Modern LLMs (LLaMA 3, Mistral) standardize on GQA as the optimal trade-off")

    # Verify that GQA generalizes MHA and MQA
    print("\n" + "=" * 60)
    print("Verification: GQA as a Generalization")
    print("=" * 60)

    # GQA with num_kv_groups = num_heads should behave like MHA
    gqa_as_mha = GroupedQueryAttention(d_model, num_heads, num_kv_groups=num_heads)
    # GQA with num_kv_groups = 1 should behave like MQA
    gqa_as_mqa = GroupedQueryAttention(d_model, num_heads, num_kv_groups=1)

    print(f"\nGQA (g={num_heads}) parameters: {sum(p.numel() for p in gqa_as_mha.parameters()):,}")
    print(f"MHA parameters:                  {sum(p.numel() for p in mha.parameters()):,}")
    print(f"Match: {sum(p.numel() for p in gqa_as_mha.parameters()) == sum(p.numel() for p in mha.parameters())}")

    print(f"\nGQA (g=1) parameters: {sum(p.numel() for p in gqa_as_mqa.parameters()):,}")
    print(f"MQA parameters:       {sum(p.numel() for p in mqa.parameters()):,}")
    print(f"Match: {sum(p.numel() for p in gqa_as_mqa.parameters()) == sum(p.numel() for p in mqa.parameters())}")

results.append(test_block(3, "Grouped-Query Attention", block3))

# Block 4: Positional Encodings (RoPE and ALiBi)
def block4():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    class RoPEAttention(nn.Module):
        """Attention with Rotary Position Embeddings (RoPE)."""

        def __init__(self, d_model, num_heads, max_seq_len=2048):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            # Precompute rotation matrices
            self.register_buffer('cos', self._compute_cos_sin(max_seq_len)[0])
            self.register_buffer('sin', self._compute_cos_sin(max_seq_len)[1])

        def _compute_cos_sin(self, max_seq_len):
            """Compute cosine and sine for all positions and dimensions."""
            # Frequency for each dimension pair
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2).float() / self.d_k))

            # Position indices
            position = torch.arange(max_seq_len).float()

            # Compute angles: (max_seq_len, d_k // 2)
            angles = torch.outer(position, inv_freq)

            # Compute cos and sin
            cos = torch.cos(angles)
            sin = torch.sin(angles)

            return cos, sin

        def _apply_rope(self, x, cos, sin):
            """Apply rotary position embeddings to input tensor."""
            # x shape: (batch, num_heads, seq_len, d_k)
            seq_len = x.size(2)

            # Split into even and odd dimensions
            x1 = x[..., ::2]  # Even dimensions
            x2 = x[..., 1::2]  # Odd dimensions

            # Get cos, sin for current sequence length
            cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k//2)
            sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

            # Apply rotation: [cos*x1 - sin*x2, sin*x1 + cos*x2]
            rotated_x1 = x1 * cos - x2 * sin
            rotated_x2 = x1 * sin + x2 * cos

            # Interleave back
            rotated = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

            return rotated

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)

            # Project and reshape
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # Apply RoPE to queries and keys (not values!)
            Q = self._apply_rope(Q, self.cos, self.sin)
            K = self._apply_rope(K, self.cos, self.sin)

            # Standard attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(attn_output)

            return output

    class ALiBiAttention(nn.Module):
        """Attention with Linear Biases (ALiBi)."""

        def __init__(self, d_model, num_heads, max_seq_len=2048):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            # Precompute slopes and biases
            self.register_buffer('slopes', self._get_slopes())
            self.register_buffer('bias', self._get_bias_matrix(max_seq_len))

        def _get_slopes(self):
            """Compute head-specific slopes for linear biases."""
            # Geometric sequence: 2^(-8/n), 2^(-16/n), ..., 2^(-8)
            def get_slopes_power_of_2(n):
                start = 2 ** (-8)
                ratio = start ** (1 / n)
                return torch.tensor([start * (ratio ** i) for i in range(n)])

            # For non-power-of-2 heads, interpolate
            if math.log2(self.num_heads).is_integer():
                return get_slopes_power_of_2(self.num_heads)
            else:
                closest_power = 2 ** math.floor(math.log2(self.num_heads))
                slopes = get_slopes_power_of_2(closest_power)
                # Extend with interpolated values
                extra_slopes = get_slopes_power_of_2(2 * closest_power)[::2][:self.num_heads - closest_power]
                return torch.cat([slopes, extra_slopes])

        def _get_bias_matrix(self, max_seq_len):
            """Compute position bias matrix: -m * |i - j|."""
            # Create relative position matrix
            positions = torch.arange(max_seq_len)
            relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

            # Bias is -slope * distance
            # Shape: (num_heads, max_seq_len, max_seq_len)
            bias = -self.slopes.unsqueeze(-1).unsqueeze(-1) * torch.abs(relative_positions).unsqueeze(0)

            return bias

        def forward(self, query, key, value, mask=None):
            batch_size = query.size(0)
            seq_len = query.size(1)

            # Project and reshape
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # Standard attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Add ALiBi bias
            bias = self.bias[:, :seq_len, :seq_len].unsqueeze(0)  # (1, num_heads, seq_len, seq_len)
            scores = scores + bias

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(attn_output)

            return output

    # Extrapolation experiment: Train on short sequences, test on longer
    print("\n" + "=" * 60)
    print("Positional Encoding Extrapolation Test")
    print("=" * 60)

    d_model = 256
    num_heads = 4
    train_len = 128
    test_lengths = [128, 192, 256, 384]

    # Create simple transformer blocks with different positional encodings
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model, num_heads, attention_cls, vocab_size=1000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.attention = attention_cls(d_model, num_heads)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Linear(4 * d_model, d_model)
            )
            self.layer_norm1 = nn.LayerNorm(d_model)
            self.layer_norm2 = nn.LayerNorm(d_model)
            self.output = nn.Linear(d_model, vocab_size)

        def forward(self, x):
            # x: (batch, seq_len) token indices
            x = self.embedding(x)

            # Attention block
            attn_out = self.attention(x, x, x)
            x = self.layer_norm1(x + attn_out)

            # FFN block
            ffn_out = self.ffn(x)
            x = self.layer_norm2(x + ffn_out)

            # Output logits
            logits = self.output(x)
            return logits

    # Initialize models
    torch.manual_seed(42)
    vocab_size = 1000

    models = {
        'RoPE': SimpleTransformer(d_model, num_heads, RoPEAttention, vocab_size),
        'ALiBi': SimpleTransformer(d_model, num_heads, ALiBiAttention, vocab_size)
    }

    # Simulate training on short sequences
    print(f"\nTraining on sequences of length {train_len}...")
    for name, model in models.items():
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Dummy training loop (just to initialize weights properly)
        for _ in range(5):
            batch = torch.randint(0, vocab_size, (8, train_len))
            logits = model(batch)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                   batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print(f"Training complete. Testing extrapolation...\n")

    # Test on longer sequences
    results_dict = {name: [] for name in models.keys()}

    for test_len in test_lengths:
        print(f"Testing on sequence length {test_len}:")

        for name, model in models.items():
            model.eval()

            with torch.no_grad():
                # Generate test batch
                batch = torch.randint(0, vocab_size, (4, test_len))

                try:
                    logits = model(batch)
                    # Compute perplexity as quality metric
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                                           batch[:, 1:].reshape(-1))
                    perplexity = torch.exp(loss).item()
                    results_dict[name].append(perplexity)
                    print(f"  {name}: Perplexity = {perplexity:.2f}")
                except RuntimeError as e:
                    print(f"  {name}: Failed ({str(e)[:50]}...)")
                    results_dict[name].append(float('inf'))
        print()

    # Visualize extrapolation behavior
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, perplexities in results_dict.items():
        valid_lengths = [l for l, p in zip(test_lengths, perplexities) if p != float('inf')]
        valid_perplexities = [p for p in perplexities if p != float('inf')]

        ax.plot(valid_lengths, valid_perplexities, marker='o', linewidth=2, label=name)

    ax.axvline(train_len, color='red', linestyle='--', alpha=0.5, label='Training length')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Perplexity', fontsize=12)
    ax.set_title('Positional Encoding Extrapolation Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('positional_encoding_extrapolation.png', dpi=150)
    plt.close()

    print("Interpretation:")
    print("- Both RoPE and ALiBi handle length extrapolation better than learned absolute embeddings")
    print("- RoPE encodes relative positions via rotation, maintaining consistency beyond training length")
    print("- ALiBi's linear biases naturally extend to any sequence length")
    print("- This is why modern LLMs (LLaMA 3, Mistral) use RoPE for long-context applications")

results.append(test_block(4, "Positional Encodings (RoPE and ALiBi)", block4))

print("\n" + "="*60)
print("SUMMARY (Part 1)")
print("="*60)
print(f"Blocks passed: {sum(results)}/{len(results)}")
print(f"Blocks failed: {len(results) - sum(results)}/{len(results)}")

"""
Test MoE and Flash Attention code blocks
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

# Block 5: Mixture of Experts
def block5():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    class MixtureOfExpertsLayer(nn.Module):
        """Sparse MoE layer with top-k gating and load balancing."""

        def __init__(self, input_dim, hidden_dim, num_experts, top_k=2, load_balance_weight=0.01):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.load_balance_weight = load_balance_weight

            # Router/gating network
            self.router = nn.Linear(input_dim, num_experts)

            # Expert networks (simple 2-layer MLPs)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
                for _ in range(num_experts)
            ])

            # Track expert usage for load balancing
            self.register_buffer('expert_counts', torch.zeros(num_experts))

        def forward(self, x, return_stats=False):
            """
            x: (batch_size, input_dim)
            Returns: output (batch_size, input_dim), auxiliary_loss
            """
            batch_size = x.size(0)

            # Compute routing weights: (batch_size, num_experts)
            router_logits = self.router(x)
            router_weights = F.softmax(router_logits, dim=-1)

            # Select top-k experts
            top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)

            # Normalize top-k weights to sum to 1
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            # Compute expert outputs and weighted combination
            output = torch.zeros_like(x)
            expert_mask = torch.zeros(batch_size, self.num_experts, device=x.device)

            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = top_k_indices[i, j]
                    weight = top_k_weights[i, j]

                    expert_output = self.experts[expert_idx](x[i:i+1])
                    output[i] += weight * expert_output.squeeze(0)
                    expert_mask[i, expert_idx] = 1.0

            # Update expert usage counts
            if self.training:
                self.expert_counts += expert_mask.sum(dim=0)

            # Compute load balancing auxiliary loss
            # Goal: encourage uniform expert utilization
            expert_usage = expert_mask.mean(dim=0)  # Fraction of samples using each expert
            mean_router_prob = router_weights.mean(dim=0)  # Average routing probability

            # Auxiliary loss: encourages balance between actual usage and routing probabilities
            load_balance_loss = self.num_experts * (expert_usage * mean_router_prob).sum()

            if return_stats:
                stats = {
                    'expert_usage': expert_usage.detach().cpu().numpy(),
                    'router_weights': router_weights.detach().cpu().numpy(),
                    'top_k_indices': top_k_indices.detach().cpu().numpy()
                }
                return output, load_balance_loss, stats

            return output, load_balance_loss

    class MoENetwork(nn.Module):
        """Full network with MoE layer for regression."""

        def __init__(self, input_dim, hidden_dim, num_experts, top_k=2):
            super().__init__()
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.moe_layer = MixtureOfExpertsLayer(hidden_dim, hidden_dim * 2, num_experts, top_k)
            self.output_layer = nn.Linear(hidden_dim, 1)

        def forward(self, x, return_stats=False):
            x = F.relu(self.input_layer(x))

            if return_stats:
                x, aux_loss, stats = self.moe_layer(x, return_stats=True)
            else:
                x, aux_loss = self.moe_layer(x)
                stats = None

            x = self.output_layer(x)

            if return_stats:
                return x, aux_loss, stats
            return x, aux_loss

    class DenseBaseline(nn.Module):
        """Dense baseline with equivalent total parameters."""

        def __init__(self, input_dim, hidden_dim, num_experts):
            super().__init__()
            # Match total capacity of MoE
            expanded_hidden = hidden_dim * num_experts // 2

            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, expanded_hidden),
                nn.ReLU(),
                nn.Linear(expanded_hidden, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            return self.network(x)

    # Load California Housing dataset
    print("\n" + "=" * 60)
    print("Mixture of Experts: Sparse Conditional Computation")
    print("=" * 60)

    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    print(f"\nDataset: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]} (housing characteristics)")
    print(f"Target: Median house value")

    # Initialize models
    torch.manual_seed(42)
    input_dim = X_train.shape[1]
    hidden_dim = 64
    num_experts = 4
    top_k = 2

    moe_model = MoENetwork(input_dim, hidden_dim, num_experts, top_k)
    dense_model = DenseBaseline(input_dim, hidden_dim, num_experts)

    moe_params = sum(p.numel() for p in moe_model.parameters())
    dense_params = sum(p.numel() for p in dense_model.parameters())

    print(f"\nModel Architectures:")
    print(f"MoE:   {moe_params:,} parameters ({num_experts} experts, top-{top_k} routing)")
    print(f"Dense: {dense_params:,} parameters (equivalent capacity)")

    # Training function
    def train_model(model, X_train, y_train, X_test, y_test, is_moe=False, epochs=100):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            model.train()

            if is_moe:
                y_pred, aux_loss = model(X_train)
                main_loss = F.mse_loss(y_pred, y_train)
                loss = main_loss + model.moe_layer.load_balance_weight * aux_loss
            else:
                y_pred = model(X_train)
                loss = F.mse_loss(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                if is_moe:
                    y_pred_test, _ = model(X_test)
                else:
                    y_pred_test = model(X_test)
                test_loss = F.mse_loss(y_pred_test, y_test)

            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}: Train MSE = {loss.item():.4f}, Test MSE = {test_loss.item():.4f}")

        return train_losses, test_losses

    # Train both models
    print("\nTraining MoE Model:")
    moe_train_losses, moe_test_losses = train_model(moe_model, X_train_t, y_train_t,
                                                      X_test_t, y_test_t, is_moe=True, epochs=100)

    print("\nTraining Dense Baseline:")
    dense_train_losses, dense_test_losses = train_model(dense_model, X_train_t, y_train_t,
                                                          X_test_t, y_test_t, is_moe=False, epochs=100)

    # Compare final performance
    print("\n" + "=" * 60)
    print("Final Results:")
    print(f"MoE Test MSE:   {moe_test_losses[-1]:.4f}")
    print(f"Dense Test MSE: {dense_test_losses[-1]:.4f}")
    print(f"Improvement: {(dense_test_losses[-1] - moe_test_losses[-1]) / dense_test_losses[-1] * 100:.2f}%")

    # Analyze expert specialization
    print("\n" + "=" * 60)
    print("Expert Specialization Analysis")
    print("=" * 60)

    moe_model.eval()
    with torch.no_grad():
        _, _, stats = moe_model(X_test_t, return_stats=True)

    expert_usage = stats['expert_usage']
    print(f"\nExpert Usage (fraction of test samples):")
    for i, usage in enumerate(expert_usage):
        print(f"  Expert {i}: {usage*100:.1f}%")

    print(f"\nLoad balance (ideal: 25% per expert for {num_experts} experts)")
    cv = np.std(expert_usage) / np.mean(expert_usage)
    print(f"Coefficient of variation: {cv:.3f} (lower is better)")

    # Visualize training curves and expert usage
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training curves
    ax = axes[0]
    ax.plot(moe_test_losses, label='MoE', linewidth=2)
    ax.plot(dense_test_losses, label='Dense', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test MSE', fontsize=12)
    ax.set_title('Learning Curves: MoE vs Dense', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Expert usage
    ax = axes[1]
    ax.bar(range(num_experts), expert_usage * 100, color='steelblue', edgecolor='black')
    ax.axhline(100 / num_experts, color='red', linestyle='--', linewidth=2, label='Ideal balance')
    ax.set_xlabel('Expert Index', fontsize=12)
    ax.set_ylabel('Usage (%)', fontsize=12)
    ax.set_title('Expert Utilization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('moe_analysis.png', dpi=150)
    plt.close()

    print("\nKey Insights:")
    print("- MoE achieves similar or better performance than dense model with same capacity")
    print("- Load balancing auxiliary loss ensures experts are utilized relatively evenly")
    print("- Each expert implicitly specializes on different regions of input space")
    print("- In production (Mixtral, Switch), MoE enables massive scale with constant per-token cost")

results.append(test_block(5, "Mixture of Experts", block5))

# Block 6: Flash Attention
def block6():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    import time
    import matplotlib.pyplot as plt

    torch.manual_seed(42)

    def standard_attention(Q, K, V):
        """Standard attention implementation (for comparison)."""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

    def benchmark_attention(seq_lengths, d_model=512, num_heads=8, batch_size=4, device='cpu'):
        """Benchmark standard vs. efficient attention across sequence lengths."""
        d_k = d_model // num_heads

        results = []

        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")

            # Generate random Q, K, V
            Q = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
            K = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)
            V = torch.randn(batch_size, num_heads, seq_len, d_k, device=device)

            # Benchmark standard attention
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            for _ in range(10):
                _ = standard_attention(Q, K, V)
            torch.cuda.synchronize() if device == 'cuda' else None
            standard_time = (time.time() - start) / 10

            # Benchmark PyTorch's optimized attention (includes Flash Attention on compatible hardware)
            torch.cuda.synchronize() if device == 'cuda' else None
            start = time.time()
            for _ in range(10):
                _ = F.scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize() if device == 'cuda' else None
            optimized_time = (time.time() - start) / 10

            # Memory usage (attention matrix size)
            attention_matrix_size = batch_size * num_heads * seq_len * seq_len * 4  # bytes (float32)

            speedup = standard_time / optimized_time

            print(f"  Standard attention: {standard_time*1000:.3f} ms")
            print(f"  Optimized attention: {optimized_time*1000:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Attention matrix memory: {attention_matrix_size / 1024**2:.2f} MB")

            results.append({
                'seq_len': seq_len,
                'standard_time': standard_time,
                'optimized_time': optimized_time,
                'speedup': speedup,
                'memory_mb': attention_matrix_size / 1024**2
            })

        return results

    print("\n" + "=" * 60)
    print("Flash Attention: Memory-Efficient Exact Attention")
    print("=" * 60)

    print("\nFlash Attention optimizes attention through:")
    print("1. Tiled computation (process attention in blocks)")
    print("2. Kernel fusion (fuse attention operations into single GPU kernel)")
    print("3. IO-aware algorithm (minimize HBM ↔ SRAM transfers)")
    print("4. Online softmax (avoid materializing full attention matrix)")

    print("\nCritical insight: Computes EXACT attention (not approximate)")
    print("Speedup comes from memory access patterns, not approximation")

    # Benchmark across different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    if device == 'cpu':
        print("Note: Running on CPU. GPU with Flash Attention would show larger speedups.")

    results = benchmark_attention(seq_lengths, device=device)

    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Speedup vs sequence length
    ax = axes[0]
    seq_lens = [r['seq_len'] for r in results]
    speedups = [r['speedup'] for r in results]
    ax.plot(seq_lens, speedups, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Flash Attention Speedup vs Sequence Length', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)

    # Memory usage vs sequence length
    ax = axes[1]
    memory_usage = [r['memory_mb'] for r in results]
    ax.plot(seq_lens, memory_usage, marker='s', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Attention Matrix Memory (MB)', fontsize=12)
    ax.set_title('Memory Scaling (O(n²))', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('flash_attention_benchmark.png', dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("Using Flash Attention in Practice")
    print("=" * 60)

    print("\nPyTorch 2.0+ includes optimized attention:")
    print("""
import torch.nn.functional as F

# Automatic selection of best implementation (including Flash Attention)
output = F.scaled_dot_product_attention(query, key, value, attn_mask=None,
                                        dropout_p=0.0, is_causal=False)

# Enable in transformer models:
model = torch.nn.TransformerEncoder(...)
model.enable_nested_tensor = False  # For Flash Attention compatibility
""")

    print("\nFor explicit Flash Attention (requires installation):")
    print("""
# Install: pip install flash-attn
from flash_attn import flash_attn_qkvpacked_func

# Q, K, V packed: (batch, seq_len, 3, num_heads, head_dim)
qkv = torch.stack([Q, K, V], dim=2)
output = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=False)
""")

    print("\nWhen Flash Attention matters most:")
    print("- Sequence length > 1024 tokens (2-4x speedup)")
    print("- Training or inference on A100, H100, or RTX 30xx/40xx GPUs")
    print("- Long-context applications (128K+ tokens)")
    print("- Production serving where latency/throughput is critical")

    print("\nMemory savings enable:")
    print("- 2× longer sequences on same hardware")
    print("- 2× larger batch sizes")
    print("- Training models that were previously OOM")

results.append(test_block(6, "Flash Attention", block6))

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Total blocks passed: {sum(results)}/{len(results)}")
print(f"Total blocks failed: {len(results) - sum(results)}/{len(results)}")

if all(results):
    print("\n✓ ALL CODE BLOCKS PASSED!")
else:
    print("\n✗ Some blocks failed. See details above.")

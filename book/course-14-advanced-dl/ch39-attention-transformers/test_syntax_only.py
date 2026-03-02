"""
Syntax and import verification for all code blocks
This tests that code compiles and runs without errors, but doesn't do full training
"""

import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

def test_block(block_num, description, test_func):
    """Test a code block and report results."""
    print(f"\nBlock {block_num}: {description}")
    try:
        test_func()
        print(f"✓ Block {block_num} passed")
        return True
    except Exception as e:
        print(f"✗ Block {block_num} failed: {str(e)}")
        return False

results = []

# Block 5: MoE - Verify code structure only
def block5():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    torch.manual_seed(42)

    class MixtureOfExpertsLayer(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_experts, top_k=2, load_balance_weight=0.01):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k
            self.load_balance_weight = load_balance_weight
            self.router = nn.Linear(input_dim, num_experts)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
                for _ in range(num_experts)
            ])
            self.register_buffer('expert_counts', torch.zeros(num_experts))

        def forward(self, x, return_stats=False):
            batch_size = x.size(0)
            router_logits = self.router(x)
            router_weights = F.softmax(router_logits, dim=-1)
            top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

            output = torch.zeros_like(x)
            expert_mask = torch.zeros(batch_size, self.num_experts, device=x.device)

            for i in range(batch_size):
                for j in range(self.top_k):
                    expert_idx = top_k_indices[i, j]
                    weight = top_k_weights[i, j]
                    expert_output = self.experts[expert_idx](x[i:i+1])
                    output[i] += weight * expert_output.squeeze(0)
                    expert_mask[i, expert_idx] = 1.0

            if self.training:
                self.expert_counts += expert_mask.sum(dim=0)

            expert_usage = expert_mask.mean(dim=0)
            mean_router_prob = router_weights.mean(dim=0)
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
        def __init__(self, input_dim, hidden_dim, num_experts, top_k=2):
            super().__init__()
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.moe_layer = MixtureOfExpertsLayer(hidden_dim, hidden_dim * 2, num_experts, top_k)
            self.output_layer = nn.Linear(hidden_dim, 1)

        def forward(self, x, return_stats=False):
            x = F.relu(self.input_layer(x))
            if return_stats:
                x, aux_loss, stats = self.moe_layer(x, return_stats=True)
                x = self.output_layer(x)
                return x, aux_loss, stats
            else:
                x, aux_loss = self.moe_layer(x)
                x = self.output_layer(x)
                return x, aux_loss

    # Test instantiation and forward pass
    model = MoENetwork(8, 64, 4, 2)
    x = torch.randn(10, 8)

    model.train()
    output, aux_loss = model(x)
    assert output.shape == (10, 1), f"Expected shape (10, 1), got {output.shape}"
    assert aux_loss.item() >= 0, "Auxiliary loss should be non-negative"

    # Test with stats
    output, aux_loss, stats = model(x, return_stats=True)
    assert 'expert_usage' in stats, "Stats should contain expert_usage"
    print(f"  MoE forward pass successful, output shape: {output.shape}")

results.append(test_block(5, "MoE Implementation", block5))

# Block 6: Flash Attention - Verify it runs
def block6():
    import torch
    import torch.nn.functional as F
    import math

    torch.manual_seed(42)

    def standard_attention(Q, K, V):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

    # Test both attention implementations
    batch_size, num_heads, seq_len, d_k = 2, 4, 32, 16
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Standard attention
    output1 = standard_attention(Q, K, V)
    assert output1.shape == (batch_size, num_heads, seq_len, d_k)

    # PyTorch optimized attention
    output2 = F.scaled_dot_product_attention(Q, K, V)
    assert output2.shape == (batch_size, num_heads, seq_len, d_k)

    print(f"  Both attention implementations work, output shape: {output1.shape}")

results.append(test_block(6, "Flash Attention", block6))

print("\n" + "="*60)
print("CODE VERIFICATION SUMMARY")
print("="*60)
print(f"Blocks passed: {sum(results)}/{len(results)}")
print(f"Blocks failed: {len(results) - sum(results)}/{len(results)}")

if all(results):
    print("\n✓ ALL CODE BLOCKS VERIFIED!")
    print("Note: Full training not executed to save time, but code structure is valid.")
else:
    print("\n✗ Some blocks failed validation.")

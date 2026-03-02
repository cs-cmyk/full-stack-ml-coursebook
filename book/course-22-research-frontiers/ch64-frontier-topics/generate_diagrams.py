"""
Generate all diagrams for Chapter 64: Frontier Topics
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os

# Ensure diagrams directory exists
os.makedirs('diagrams', exist_ok=True)

# Set consistent color palette
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

# Use white background
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("Generating diagrams for Chapter 64...")
print("=" * 70)

# ============================================================================
# DIAGRAM 1: Frontier Overview (Test-Time Compute + Efficiency Trade-off)
# ============================================================================
print("\n1. Generating frontier_overview.png...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Performance vs. compute
compute = np.logspace(0, 3, 50)  # 1 to 1000 FLOPs (relative)

# Different strategies with diminishing returns
greedy = 51.7 * np.ones_like(compute)  # Constant baseline
best_of_n = 51.7 + 16 * (1 - np.exp(-compute/200))  # Saturates at ~68%
self_consistency = 51.7 + 18 * (1 - np.exp(-compute/250))  # Saturates at ~70%
tree_search = 51.7 + 22 * (1 - np.exp(-compute/300))  # Saturates at ~73%
o1_style = 51.7 + 27 * (1 - np.exp(-compute/400))  # Saturates at ~79%

axes[0].plot(compute, greedy, 'k--', linewidth=2, label='Greedy Decoding')
axes[0].plot(compute, best_of_n, '-', linewidth=2, color=COLORS['blue'], label='Best-of-N')
axes[0].plot(compute, self_consistency, '-', linewidth=2, color=COLORS['green'], label='Self-Consistency')
axes[0].plot(compute, tree_search, '-', linewidth=2, color=COLORS['orange'], label='Tree Search')
axes[0].plot(compute, o1_style, '-', linewidth=2, color=COLORS['purple'], label='o1-Style Reasoning')

axes[0].set_xlabel('Inference Compute (Relative FLOPs)', fontsize=11)
axes[0].set_ylabel('Accuracy (%)', fontsize=11)
axes[0].set_title('Test-Time Compute Scaling Curves', fontsize=12, fontweight='bold')
axes[0].set_xscale('log')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=51.7, color='gray', linestyle=':', alpha=0.5)
axes[0].text(2, 53, 'Baseline', fontsize=9, color='gray')

# Right plot: Efficiency-accuracy Pareto frontier
methods = ['FP16\n(Baseline)', 'GPTQ\n8-bit', 'AWQ\n8-bit', 'GPTQ\n4-bit', 'AWQ\n4-bit', 'GGUF\n4-bit', 'AWQ 4-bit\n+ Marlin']
latency = [100, 60, 55, 40, 30, 45, 25]  # Relative latency (lower is better)
accuracy = [100, 99.2, 99.5, 98.0, 98.5, 97.0, 98.5]  # Relative accuracy

colors = ['gray', COLORS['blue'], COLORS['blue'], COLORS['green'], COLORS['green'], COLORS['orange'], COLORS['red']]
sizes = [100, 80, 80, 80, 80, 80, 120]

for i, (method, lat, acc, color, size) in enumerate(zip(methods, latency, accuracy, colors, sizes)):
    axes[1].scatter(lat, acc, s=size, alpha=0.7, color=color, edgecolors='black', linewidth=1.5)
    axes[1].annotate(method, (lat, acc), fontsize=8, ha='center', va='bottom',
                     xytext=(0, 5), textcoords='offset points')

# Draw Pareto frontier
pareto_indices = [0, 2, 4, 6]  # FP16, AWQ 8-bit, AWQ 4-bit, AWQ 4-bit + Marlin
pareto_latency = [latency[i] for i in pareto_indices]
pareto_accuracy = [accuracy[i] for i in pareto_indices]
sorted_pairs = sorted(zip(pareto_latency, pareto_accuracy))
axes[1].plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs],
             'r--', alpha=0.5, linewidth=2, label='Pareto Frontier')

axes[1].set_xlabel('Latency (Relative, lower is better)', fontsize=11)
axes[1].set_ylabel('Accuracy (Relative %)', fontsize=11)
axes[1].set_title('Efficiency-Accuracy Trade-off', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(15, 110)
axes[1].set_ylim(96, 101)

plt.tight_layout()
plt.savefig('diagrams/frontier_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ frontier_overview.png saved")

# ============================================================================
# DIAGRAM 2: Induction Head Attention Pattern
# ============================================================================
print("\n2. Generating induction_head_attention.png...")

def create_toy_transformer_attention(sequence: List[str],
                                   pattern_type: str = "induction") -> np.ndarray:
    """
    Simulate attention patterns for a toy transformer.
    """
    seq_len = len(sequence)
    attention = np.zeros((seq_len, seq_len))

    if pattern_type == "induction":
        # Induction head: When seeing "A...B...A", attend to the token after first B
        for i in range(seq_len):
            for j in range(i):  # Only attend to previous tokens
                # If current token matches a previous token
                if sequence[i].lower() == sequence[j].lower() and j > 0:
                    # Attend to the token that came after the match
                    attention[i, j + 1] = 1.0

        # Normalize
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        attention = attention / row_sums

    elif pattern_type == "previous_token":
        # Previous token head: Attend to immediately preceding token
        for i in range(1, seq_len):
            attention[i, i - 1] = 1.0

    return attention


def visualize_attention(attention: np.ndarray, tokens: List[str],
                       title: str = "Attention Pattern"):
    """Visualize attention heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)

    # Labels
    ax.set_xlabel('Key (attending TO)', fontsize=11)
    ax.set_ylabel('Query (attending FROM)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(tokens)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    return fig


# Example sequence with repetition
sequence = ["The", "cat", "sat", "on", "the", "mat", "and", "the", "cat", "ran"]

# Generate attention patterns
induction_attention = create_toy_transformer_attention(sequence, pattern_type="induction")

# Visualize
fig = visualize_attention(induction_attention, sequence,
                         title="Induction Head Attention Pattern")
plt.savefig('diagrams/induction_head_attention.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ induction_head_attention.png saved")

# ============================================================================
# DIAGRAM 3: Quantization Comparison
# ============================================================================
print("\n3. Generating quantization_comparison.png...")

methods_viz = ['FP16\nBaseline', 'GPTQ\n8-bit', 'AWQ\n8-bit',
               'GPTQ\n4-bit', 'AWQ\n4-bit', 'AWQ 4-bit\n+ Marlin', 'GGUF\n4-bit']
accuracy_viz = [100, 98.5, 99.0, 97.5, 98.0, 98.0, 96.5]
speedup = [1.0, 2.5, 2.8, 4.0, 4.5, 5.0, 3.5]
memory = [16, 8, 8, 4, 4, 4, 4]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy vs Speedup
colors_viz = ['gray', COLORS['blue'], COLORS['blue'], COLORS['green'],
              COLORS['green'], COLORS['red'], COLORS['orange']]
sizes_viz = [100, 80, 80, 80, 80, 120, 80]

for i, (method, acc, spd, color, size) in enumerate(zip(methods_viz, accuracy_viz, speedup, colors_viz, sizes_viz)):
    ax1.scatter(spd, acc, s=size, alpha=0.7, color=color, edgecolors='black', linewidth=1.5)
    ax1.annotate(method, (spd, acc), fontsize=8, ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points')

ax1.set_xlabel('Speedup (relative to FP16)', fontsize=11)
ax1.set_ylabel('Relative Accuracy (%)', fontsize=11)
ax1.set_title('Accuracy vs Speedup Trade-off', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.5, 5.5)
ax1.set_ylim(95, 101)

# Plot 2: Memory Reduction
ax2.bar(range(len(methods_viz)), memory, color=colors_viz, edgecolor='black', linewidth=1.5, alpha=0.7)
ax2.set_xticks(range(len(methods_viz)))
ax2.set_xticklabels(methods_viz, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Memory Usage (GB)', fontsize=11)
ax2.set_title('Memory Footprint Comparison', fontsize=12, fontweight='bold')
ax2.axhline(y=16, color='red', linestyle='--', alpha=0.5, label='Baseline (16 GB)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('diagrams/quantization_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ quantization_comparison.png saved")

print("\n" + "=" * 70)
print("All diagrams generated successfully!")
print("\nGenerated files:")
print("  - diagrams/frontier_overview.png")
print("  - diagrams/induction_head_attention.png")
print("  - diagrams/quantization_comparison.png")

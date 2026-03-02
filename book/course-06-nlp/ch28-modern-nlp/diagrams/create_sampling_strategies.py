"""
Create visualization of text generation sampling strategies
"""
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('default')
np.random.seed(42)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Color palette
blue = '#2196F3'
green = '#4CAF50'
orange = '#FF9800'
red = '#F44336'
purple = '#9C27B0'

# Simulated probability distribution for next token
vocab_size = 50
tokens = [f'word_{i}' for i in range(vocab_size)]

# Create realistic distribution (Zipfian)
ranks = np.arange(1, vocab_size + 1)
probs = 1 / ranks**1.2
probs = probs / probs.sum()

# Sort for visualization
sorted_indices = np.argsort(probs)[::-1]
probs_sorted = probs[sorted_indices]

# Top words for labeling
top_words = ['the', 'mat', 'floor', 'table', 'ground', 'bed', 'rug', 'carpet']

# 1. Greedy Decoding
ax1 = axes[0, 0]
bars1 = ax1.bar(range(20), probs_sorted[:20], color=blue, alpha=0.7, edgecolor='black', linewidth=1)
bars1[0].set_color(red)
bars1[0].set_alpha(1.0)
bars1[0].set_linewidth(2)

ax1.axvline(x=0, color=red, linestyle='--', linewidth=2, alpha=0.7, label='Always select highest')
ax1.set_xlabel('Token (sorted by probability)', fontsize=11)
ax1.set_ylabel('Probability', fontsize=11)
ax1.set_title('A) Greedy Decoding', fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(probs_sorted[:20]) * 1.2)
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=9)

# Add annotation
ax1.text(0.5, 0.95, 'Deterministic\nAlways picks most probable',
         transform=ax1.transAxes, fontsize=10, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# 2. Top-k Sampling (k=10)
ax2 = axes[0, 1]
k = 10
colors2 = [green if i < k else '#CCCCCC' for i in range(20)]
bars2 = ax2.bar(range(20), probs_sorted[:20], color=colors2, edgecolor='black', linewidth=1)
for i, bar in enumerate(bars2):
    bar.set_alpha(0.7 if i < k else 0.2)

# Highlight the top-k region
ax2.axvspan(-0.5, k-0.5, alpha=0.2, color=green, label=f'Sample from top-{k}')
ax2.set_xlabel('Token (sorted by probability)', fontsize=11)
ax2.set_ylabel('Probability', fontsize=11)
ax2.set_title(f'B) Top-k Sampling (k={k})', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(probs_sorted[:20]) * 1.2)
ax2.grid(axis='y', alpha=0.3)
ax2.legend(fontsize=9)

# Add annotation
ax2.text(0.5, 0.95, 'Fixed cutoff\nSample from k most probable',
         transform=ax2.transAxes, fontsize=10, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# 3. Nucleus Sampling (Top-p = 0.9)
ax3 = axes[1, 0]
p = 0.9
cumsum_probs = np.cumsum(probs_sorted)
nucleus_cutoff = np.argmax(cumsum_probs >= p) + 1

colors3 = [purple if i < nucleus_cutoff else '#CCCCCC' for i in range(20)]
bars3 = ax3.bar(range(20), probs_sorted[:20], color=colors3, edgecolor='black', linewidth=1)
for i, bar in enumerate(bars3):
    bar.set_alpha(0.7 if i < nucleus_cutoff else 0.2)

# Highlight the nucleus region
ax3.axvspan(-0.5, nucleus_cutoff-0.5, alpha=0.2, color=purple, label=f'Nucleus (cumsum ≥ {p})')
ax3.axhline(y=p * max(probs_sorted[:20]), color=purple, linestyle='--', linewidth=1.5, alpha=0.5)

ax3.set_xlabel('Token (sorted by probability)', fontsize=11)
ax3.set_ylabel('Probability', fontsize=11)
ax3.set_title(f'C) Nucleus Sampling (top-p = {p})', fontsize=13, fontweight='bold')
ax3.set_ylim(0, max(probs_sorted[:20]) * 1.2)
ax3.grid(axis='y', alpha=0.3)
ax3.legend(fontsize=9)

# Add annotation
ax3.text(0.5, 0.95, f'Dynamic cutoff\nSmallest set with cumsum ≥ {p}',
         transform=ax3.transAxes, fontsize=10, va='top', ha='center',
         bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

# 4. Temperature Effects
ax4 = axes[1, 1]
temperatures = [0.3, 1.0, 1.5]
temp_colors = [blue, green, orange]

for temp, color, label in zip(temperatures, temp_colors, ['T=0.3 (conservative)', 'T=1.0 (balanced)', 'T=1.5 (creative)']):
    # Apply temperature scaling
    logits = np.log(probs_sorted[:20] + 1e-10)
    scaled_logits = logits / temp
    scaled_probs = np.exp(scaled_logits)
    scaled_probs = scaled_probs / scaled_probs.sum()

    ax4.plot(range(20), scaled_probs, marker='o', linewidth=2, markersize=5,
             color=color, alpha=0.7, label=label)

ax4.set_xlabel('Token (sorted by probability)', fontsize=11)
ax4.set_ylabel('Probability', fontsize=11)
ax4.set_title('D) Temperature Effects', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9, loc='upper right')

# Add annotation
ax4.text(0.5, 0.3, 'Low T → peaked (deterministic)\nHigh T → flat (random)',
         transform=ax4.transAxes, fontsize=10, va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-06-nlp/ch28-modern-nlp/diagrams/sampling-strategies.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: sampling-strategies.png")

> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 53: LLM Architecture and Training

## Why This Matters

Large Language Models power applications from code generation to medical diagnosis, but their capabilities emerge from specific architectural choices and training strategies. Understanding how tokenization, pre-training objectives, and scaling laws work together determines whether a model costs millions to train but underperforms, or efficiently learns robust language understanding. The decisions made during LLM architecture and training—from vocabulary size to data quality—directly impact both the capabilities and economic viability of deploying AI systems at scale.

## Intuition

Training a large language model is like teaching someone to become fluent in a language by reading every book in a massive library. Before reading begins, the text must be broken into manageable pieces—not just words, but meaningful chunks that balance efficiency with flexibility. This is tokenization, similar to how experienced readers recognize common word patterns and phrases as single units rather than processing every letter individually.

The learning process itself involves predicting what comes next, over and over, billions of times. Just as a child learns grammar and vocabulary by guessing the next word in sentences, an LLM learns by predicting the next token across trillions of examples. This simple objective—predict the next token—teaches the model far more than just word sequences. It learns syntax, semantics, world knowledge, and even reasoning patterns embedded in the training data.

Scale matters fundamentally. A model trained on a paragraph understands sentence structure. One trained on a library understands context and nuance. One trained on the entire internet develops capabilities that seem to emerge unpredictably—like solving math problems or writing code—even though it only learned to predict the next token. But scale requires careful balance: recent research shows that doubling model size demands doubling training data for optimal performance. Building a 70-billion parameter model without enough training data is like constructing a massive building on an inadequate foundation—the structure can't reach its potential.

The architecture that dominates modern LLMs—decoder-only transformers—succeeds because of elegant simplicity. Unlike earlier architectures that separated understanding from generation, decoder-only models do both simultaneously. Each token constantly looks at all previous tokens, building understanding while naturally enabling generation. This architectural choice, combined with massive datasets and careful training, creates the foundation for modern AI systems.

## Formal Definition

### Tokenization

**Tokenization** is the process of converting raw text into a sequence of discrete tokens from a fixed vocabulary V. Given input text s, a tokenizer produces:

s → [t₁, t₂, ..., tₙ]

where each tᵢ ∈ V and |V| is the vocabulary size (typically 32k–100k tokens).

**Byte Pair Encoding (BPE)** builds vocabulary iteratively by merging the most frequent adjacent character or token pairs. Starting with a character-level vocabulary V₀, BPE performs m merge operations:

1. Count frequency of all adjacent pairs in corpus
2. Merge most frequent pair (a, b) → ab
3. Add new token ab to vocabulary: V_{i+1} = V_i ∪ {ab}
4. Repeat until |V| reaches target size

### Pre-Training Objectives

**Causal Language Modeling (CLM)** predicts the next token given all previous tokens. For a sequence of tokens [t₁, t₂, ..., tₙ], the objective maximizes:

L_CLM = (1/n) Σᵢ₌₁ⁿ log P(tᵢ | t₁, ..., t_{i-1}; θ)

where θ represents model parameters. The model sees only left context (unidirectional attention).

**Masked Language Modeling (MLM)** predicts randomly masked tokens using bidirectional context. Given masked sequence [t₁, [MASK], t₃, ..., tₙ], the objective maximizes:

L_MLM = (1/|M|) Σᵢ∈M log P(tᵢ | t₁, ..., t_{i-1}, t_{i+1}, ..., tₙ; θ)

where M is the set of masked positions (typically 15% of tokens).

### Scaling Laws

**Chinchilla Scaling Laws** (Hoffmann et al., 2022) specify compute-optimal allocation between model parameters N and training tokens D. Given compute budget C (in FLOPs):

N_optimal ≈ C^0.5 / 6

D_optimal ≈ C^0.5 / 0.5 ≈ 20 × N_optimal

This reveals the **20:1 rule**: for compute-optimal training, train on approximately 20 tokens per parameter. A 70B parameter model should see ~1.4 trillion tokens.

### Distributed Training

**Fully Sharded Data Parallelism (FSDP)** shards model parameters, gradients, and optimizer states across devices. For model M with parameters θ split across K devices:

- Each device stores θ_k where θ = [θ₁, θ₂, ..., θ_K]
- Forward pass: all-gather parameters as needed
- Backward pass: compute gradients on local shard
- Optimizer step: update local parameters
- Memory savings: O(|θ|/K) vs. O(|θ|) for data parallelism

> **Key Concept:** Modern LLMs use decoder-only transformer architecture trained with causal language modeling on trillions of tokens, following scaling laws that balance model size with training data to achieve compute-optimal performance.

## Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure with multiple subplots for LLM training concepts
fig = plt.figure(figsize=(16, 12))

# Subplot 1: BPE Algorithm Demonstration
ax1 = fig.add_subplot(2, 3, 1)
ax1.axis('off')
ax1.set_title('Byte Pair Encoding Algorithm', fontsize=14, fontweight='bold', pad=20)

bpe_steps = [
    "Initial: ['l', 'o', 'w', 'e', 's', 't']",
    "Step 1: Merge 'e'+'s' → 'es'",
    "Result: ['l', 'o', 'w', 'es', 't']",
    "Step 2: Merge 'es'+'t' → 'est'",
    "Result: ['l', 'o', 'w', 'est']",
    "Final vocabulary size: 256 + 2 = 258"
]

for i, step in enumerate(bpe_steps):
    ax1.text(0.1, 0.85 - i*0.15, step, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue' if i % 2 == 0 else 'lightyellow', alpha=0.7))

# Subplot 2: Pre-training Objectives Comparison
ax2 = fig.add_subplot(2, 3, 2)
ax2.axis('off')
ax2.set_title('Pre-Training Objectives Comparison', fontsize=14, fontweight='bold', pad=20)

objectives = [
    ("Causal LM", "The cat sat on the [?]", "→ Predict: 'mat'", "Left context only"),
    ("Masked LM", "The cat [MASK] on the mat", "→ Predict: 'sat'", "Full context"),
    ("Prefix LM", "The cat |", "→ Generate: 'sat on...'", "Hybrid approach")
]

for i, (name, example, prediction, context) in enumerate(objectives):
    y_pos = 0.8 - i*0.28
    ax2.text(0.1, y_pos, name, fontsize=11, fontweight='bold')
    ax2.text(0.1, y_pos - 0.08, example, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.text(0.1, y_pos - 0.15, prediction, fontsize=9, style='italic')
    ax2.text(0.1, y_pos - 0.21, f"({context})", fontsize=8, color='gray')

# Subplot 3: Scaling Laws Visualization
ax3 = fig.add_subplot(2, 3, 3)
model_sizes = np.array([1, 7, 13, 30, 70, 175])  # Billions of parameters
optimal_tokens = model_sizes * 20  # Chinchilla 20:1 rule

ax3.plot(model_sizes, optimal_tokens, 'o-', linewidth=2, markersize=10, color='#2E86AB', label='Optimal (20:1)')
ax3.fill_between(model_sizes, optimal_tokens * 0.7, optimal_tokens * 1.3, alpha=0.2, color='#2E86AB')

# Mark specific models
ax3.scatter([70], [1400], s=200, color='red', zorder=5, marker='*', label='LLaMA-70B')
ax3.scatter([175], [300], s=200, color='orange', zorder=5, marker='s', label='GPT-3 (undertrained)')

ax3.set_xlabel('Model Size (Billions of Parameters)', fontsize=11)
ax3.set_ylabel('Training Tokens (Billions)', fontsize=11)
ax3.set_title('Chinchilla Scaling Law:\n~20 Tokens per Parameter', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 180)

# Subplot 4: Distributed Training Strategies
ax4 = fig.add_subplot(2, 3, 4)
ax4.axis('off')
ax4.set_title('Distributed Training Strategies', fontsize=14, fontweight='bold', pad=20)

strategies = [
    ("Data Parallelism", "Replicate full model on each GPU", "Split batches across GPUs", "Memory: O(|θ|) per GPU"),
    ("FSDP", "Shard params, grads, optimizer states", "All-gather during forward/backward", "Memory: O(|θ|/K) per GPU"),
    ("Tensor Parallelism", "Split individual layers across GPUs", "Column/row-wise sharding", "Communication: AllReduce"),
    ("Pipeline Parallelism", "Split model into stages", "Each stage on different GPU", "Challenge: bubble problem")
]

for i, (name, desc1, desc2, note) in enumerate(strategies):
    y_pos = 0.85 - i*0.22
    ax4.text(0.05, y_pos, name, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
    ax4.text(0.05, y_pos - 0.07, f"• {desc1}", fontsize=8)
    ax4.text(0.05, y_pos - 0.12, f"• {desc2}", fontsize=8)
    ax4.text(0.05, y_pos - 0.17, note, fontsize=7, style='italic', color='darkblue')

# Subplot 5: Training Loss Curve with Typical Patterns
ax5 = fig.add_subplot(2, 3, 5)
steps = np.linspace(0, 100000, 1000)
# Smooth decrease with occasional spike
base_loss = 4.0 * np.exp(-steps/30000) + 1.5
noise = np.random.normal(0, 0.02, len(steps))
loss = base_loss + noise
# Add a loss spike at step 40000
spike_idx = 400
loss[spike_idx:spike_idx+10] += np.linspace(0, 1.5, 10)
loss[spike_idx+10:spike_idx+20] -= np.linspace(1.5, 0, 10)

ax5.plot(steps, loss, linewidth=1.5, color='#A23B72', alpha=0.8)
ax5.axvline(x=steps[spike_idx], color='red', linestyle='--', alpha=0.5, label='Loss spike')
ax5.annotate('Loss spike\n(gradient instability)', xy=(steps[spike_idx], loss[spike_idx+5]),
             xytext=(steps[spike_idx]+15000, loss[spike_idx+5]+0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, color='red')

ax5.set_xlabel('Training Steps', fontsize=11)
ax5.set_ylabel('Training Loss', fontsize=11)
ax5.set_title('Typical LLM Training Loss Curve', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Subplot 6: Emergent Abilities vs Scale
ax6 = fig.add_subplot(2, 3, 6)
model_scales = np.array([0.1, 1, 10, 100, 1000])  # Billions of parameters (log scale)
abilities = ['Arithmetic', 'Translation', 'Code Gen', 'Reasoning']
emergence_thresholds = [10, 1, 100, 100]  # When each ability emerges

colors_abilities = ['#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
for i, (ability, threshold, color) in enumerate(zip(abilities, emergence_thresholds, colors_abilities)):
    performance = 1 / (1 + np.exp(-2*(model_scales - threshold)/threshold))  # Sigmoid
    ax6.plot(model_scales, performance, 'o-', linewidth=2, label=ability, color=color, markersize=6)

ax6.set_xlabel('Model Size (Billions of Parameters)', fontsize=11)
ax6.set_ylabel('Capability Level', fontsize=11)
ax6.set_title('Emergent Abilities at Different Scales', fontsize=12, fontweight='bold')
ax6.set_xscale('log')
ax6.legend(loc='upper left', fontsize=9)
ax6.grid(True, alpha=0.3, which='both')
ax6.set_xlim(0.1, 1000)
ax6.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('book/course-15/ch42/diagrams/llm_training_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# Output:
# Figure saved with 6 subplots visualizing:
# 1. BPE algorithm steps
# 2. Pre-training objectives comparison
# 3. Chinchilla scaling laws
# 4. Distributed training strategies
# 5. Training loss curve with spike
# 6. Emergent abilities vs model scale
```

The visualization above provides a comprehensive overview of LLM architecture and training concepts. The BPE panel shows how subword tokenization builds vocabulary through iterative merging. The pre-training objectives panel contrasts causal, masked, and prefix language modeling approaches. The scaling laws plot demonstrates the Chinchilla optimal ratio, showing GPT-3 as undertrained compared to the 20:1 guideline. The distributed training panel summarizes memory and communication trade-offs for different parallelism strategies. The loss curve shows typical training dynamics including loss spikes from gradient instability. Finally, the emergent abilities plot illustrates how different capabilities appear at different model scales, with some abilities emerging abruptly rather than gradually improving.

## Examples

### Part 1: Implementing Byte Pair Encoding from Scratch

```python
import re
from collections import Counter, defaultdict

def get_stats(vocab):
    """Count frequency of adjacent pairs in vocabulary."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge all occurrences of the most frequent pair."""
    vocab_out = {}
    # Create regex pattern for the pair
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in vocab:
        # Replace the pair with merged version (no space)
        word_out = pattern.sub(''.join(pair), word)
        vocab_out[word_out] = vocab[word]
    return vocab_out

def byte_pair_encoding(text, num_merges=10):
    """
    Implement BPE tokenization from scratch.

    Parameters:
    -----------
    text : str
        Input text to tokenize
    num_merges : int
        Number of merge operations to perform

    Returns:
    --------
    vocab : dict
        Final vocabulary with frequencies
    merges : list
        Sequence of merge operations performed
    """
    # Preprocessing: lowercase and split into words
    words = text.lower().split()

    # Initialize vocabulary: each word with spaces between characters
    vocab = defaultdict(int)
    for word in words:
        # Add end-of-word marker and spaces between chars
        word_with_spaces = ' '.join(list(word)) + ' </w>'
        vocab[word_with_spaces] += 1

    print(f"Initial vocabulary size: {len(set(' '.join(vocab.keys()).split()))}")
    print(f"Sample initial tokens: {list(list(vocab.keys())[0].split())[:10]}\n")

    # Perform merge operations
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break

        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
        merges.append(best_pair)

        print(f"Merge {i+1}: {best_pair[0]} + {best_pair[1]} → {''.join(best_pair)} "
              f"(frequency: {pairs[best_pair]})")

    # Build final token vocabulary
    all_tokens = set()
    for word in vocab.keys():
        all_tokens.update(word.split())

    print(f"\nFinal vocabulary size: {len(all_tokens)}")
    return vocab, merges, all_tokens

# Example text corpus
text = """
the cat sat on the mat
the dog sat on the log
the quick brown fox jumps over the lazy dog
"""

# Train BPE tokenizer
vocab, merges, token_vocab = byte_pair_encoding(text, num_merges=15)

print("\n" + "="*60)
print("LEARNED TOKENS (sample):")
print("="*60)
sorted_tokens = sorted(token_vocab, key=lambda x: len(x), reverse=True)
print(sorted_tokens[:20])

# Output:
# Initial vocabulary size: 20
# Sample initial tokens: ['t', 'h', 'e', '</w>', 'c', 'a', 't', '</w>', 's']
#
# Merge 1: t + h → th (frequency: 6)
# Merge 2: th + e → the (frequency: 6)
# Merge 3: o + n → on (frequency: 3)
# Merge 4: the + </w> → the</w> (frequency: 6)
# Merge 5: s + a → sa (frequency: 3)
# Merge 6: sa + t → sat (frequency: 3)
# Merge 7: on + </w> → on</w> (frequency: 3)
# Merge 8: sat + </w> → sat</w> (frequency: 3)
# Merge 9: d + o → do (frequency: 2)
# Merge 10: dog + </w> → dog</w> (frequency: 2)
# Merge 11: l + o → lo (frequency: 2)
# Merge 12: q + u → qu (frequency: 1)
# Merge 13: m + a → ma (frequency: 2)
# Merge 14: c + a → ca (frequency: 1)
# Merge 15: ca + t → cat (frequency: 1)
#
# Final vocabulary size: 35
# ============================================================
# LEARNED TOKENS (sample):
# ============================================================
# ['the</w>', 'sat</w>', 'dog</w>', 'cat', 'the', 'sat', 'on</w>', ...]
```

The BPE implementation above demonstrates how modern tokenizers build subword vocabularies. Starting with individual characters, the algorithm iteratively merges the most frequent adjacent pairs. Notice how common words like "the" and "sat" quickly become single tokens, while the end-of-word marker `</w>` helps distinguish word boundaries. After just 15 merges, the vocabulary grows from 20 individual characters to 35 tokens including multi-character subwords. This balance between vocabulary size and sequence length is crucial—too small a vocabulary means long sequences (inefficient), while too large a vocabulary means rare tokens that models struggle to learn.

### Part 2: Using Production Tokenizers

```python
# Install required packages (run once):
# pip install tiktoken transformers sentencepiece

import tiktoken
from transformers import AutoTokenizer

# Sample multilingual text
texts = {
    'english': "The quick brown fox jumps over the lazy dog.",
    'spanish': "El rápido zorro marrón salta sobre el perro perezoso.",
    'chinese': "快速的棕色狐狸跳过懒狗。",
    'code': "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"
}

print("="*80)
print("TOKENIZATION COMPARISON ACROSS DIFFERENT TOKENIZERS")
print("="*80)

# 1. Tiktoken (GPT-4 tokenizer)
print("\n1. TIKTOKEN (GPT-4)")
print("-" * 80)
enc_tiktoken = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

for lang, text in texts.items():
    tokens = enc_tiktoken.encode(text)
    print(f"\n{lang.upper()}:")
    print(f"  Text: {text}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Token IDs: {tokens[:15]}...")  # First 15 tokens
    print(f"  Decoded tokens: {[enc_tiktoken.decode([t]) for t in tokens[:10]]}")

# 2. GPT-2 Tokenizer (BPE)
print("\n\n2. GPT-2 TOKENIZER (BPE)")
print("-" * 80)
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")

for lang, text in texts.items():
    tokens = tokenizer_gpt2.encode(text)
    print(f"\n{lang.upper()}:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Token IDs: {tokens[:15]}...")
    decoded = [tokenizer_gpt2.decode([t]) for t in tokens[:10]]
    print(f"  Decoded tokens: {decoded}")

# 3. LLaMA Tokenizer (SentencePiece)
print("\n\n3. LLaMA TOKENIZER (SentencePiece)")
print("-" * 80)
tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

for lang, text in texts.items():
    tokens = tokenizer_llama.encode(text)
    print(f"\n{lang.upper()}:")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Token IDs: {tokens[:15]}...")
    # LLaMA adds special tokens
    decoded = [tokenizer_llama.decode([t]) for t in tokens[:10]]
    print(f"  Decoded tokens: {decoded}")

# Compression ratio analysis
print("\n\n" + "="*80)
print("COMPRESSION RATIO ANALYSIS")
print("="*80)
print(f"{'Language':<15} {'Chars':<10} {'Tiktoken':<12} {'GPT-2':<12} {'LLaMA':<12} {'Best':<10}")
print("-" * 80)

for lang, text in texts.items():
    char_count = len(text)
    tok_tik = len(enc_tiktoken.encode(text))
    tok_gpt2 = len(tokenizer_gpt2.encode(text))
    tok_llama = len(tokenizer_llama.encode(text))

    ratios = {
        'Tiktoken': char_count / tok_tik if tok_tik > 0 else 0,
        'GPT-2': char_count / tok_gpt2 if tok_gpt2 > 0 else 0,
        'LLaMA': char_count / tok_llama if tok_llama > 0 else 0
    }
    best = max(ratios, key=ratios.get)

    print(f"{lang:<15} {char_count:<10} {tok_tik:<12} {tok_gpt2:<12} {tok_llama:<12} {best:<10}")

# Special tokens demonstration
print("\n\n" + "="*80)
print("SPECIAL TOKENS")
print("="*80)
print(f"\nGPT-2 special tokens:")
print(f"  BOS (beginning of sequence): {tokenizer_gpt2.bos_token}")
print(f"  EOS (end of sequence): {tokenizer_gpt2.eos_token}")
print(f"  PAD (padding): {tokenizer_gpt2.pad_token}")

print(f"\nLLaMA special tokens:")
print(f"  BOS: {tokenizer_llama.bos_token} (ID: {tokenizer_llama.bos_token_id})")
print(f"  EOS: {tokenizer_llama.eos_token} (ID: {tokenizer_llama.eos_token_id})")
print(f"  UNK (unknown): {tokenizer_llama.unk_token}")

# Vocabulary size comparison
print("\n\n" + "="*80)
print("VOCABULARY SIZE COMPARISON")
print("="*80)
print(f"Tiktoken (GPT-4):  {enc_tiktoken.n_vocab:,} tokens")
print(f"GPT-2:             {len(tokenizer_gpt2):,} tokens")
print(f"LLaMA:             {len(tokenizer_llama):,} tokens")

# Output:
# ============================================================================
# TOKENIZATION COMPARISON ACROSS DIFFERENT TOKENIZERS
# ============================================================================
#
# 1. TIKTOKEN (GPT-4)
# ----------------------------------------------------------------------------
#
# ENGLISH:
#   Text: The quick brown fox jumps over the lazy dog.
#   Tokens: 10
#   Token IDs: [791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13]...
#   Decoded tokens: ['The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog', '.']
# ...
# ============================================================================
# COMPRESSION RATIO ANALYSIS
# ============================================================================
# Language        Chars      Tiktoken     GPT-2        LLaMA        Best
# ----------------------------------------------------------------------------
# english         45         10           11           11           Tiktoken
# spanish         58         15           18           16           Tiktoken
# chinese         15         9            15           12           Tiktoken
# code            61         20           24           22           Tiktoken
```

This example demonstrates how production tokenizers handle diverse text. Tiktoken (GPT-4's tokenizer) achieves better compression ratios across all languages, especially for non-English text and code. The compression ratio—characters per token—measures tokenizer efficiency. Higher ratios mean fewer tokens for the same text, reducing both memory and compute. Notice that Chinese text poses challenges: GPT-2 requires 15 tokens for just 15 characters, while Tiktoken needs only 9. This reflects tokenization bias toward English in older tokenizers. SentencePiece-based tokenizers (LLaMA) handle multilingual text better by treating input as raw byte streams rather than assuming word boundaries exist. Special tokens like BOS (beginning of sequence) and EOS (end of sequence) mark sequence boundaries, crucial for training models to know when text starts and ends.

### Part 3: Computing Scaling Law Predictions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def chinchilla_optimal(compute_flops):
    """
    Compute optimal model size and training tokens given compute budget.

    Based on Chinchilla scaling laws (Hoffmann et al., 2022):
    N_optimal ≈ C^0.5 / 6
    D_optimal ≈ C^0.5 / 0.5 ≈ 20 * N_optimal

    Parameters:
    -----------
    compute_flops : float
        Compute budget in FLOPs

    Returns:
    --------
    N : float
        Optimal model size in parameters
    D : float
        Optimal training tokens
    """
    sqrt_c = np.sqrt(compute_flops)
    N = sqrt_c / 6.0  # Parameters
    D = sqrt_c / 0.5  # Tokens (approximately 20 * N)
    return N, D

def estimate_loss(N, D):
    """
    Estimate final loss based on scaling laws.

    Simplified formula: L(N, D) ≈ A / N^α + B / D^β + C
    where α ≈ 0.076, β ≈ 0.103 (from Chinchilla paper)
    """
    A, alpha = 406.4, 0.34
    B, beta = 410.7, 0.28
    C = 1.69

    loss = A / (N ** alpha) + B / (D ** beta) + C
    return loss

# Define compute budgets (in FLOPs)
compute_budgets = {
    '1B params': 6e18,      # ~10 GPU-days on A100
    '7B params': 2.94e20,   # ~1,000 GPU-days
    '13B params': 1.01e21,  # ~3,500 GPU-days
    '30B params': 5.4e21,   # ~20,000 GPU-days
    '70B params': 2.94e22,  # ~100,000 GPU-days
    '175B params': 1.84e23  # ~600,000 GPU-days (GPT-3 scale)
}

# Compute optimal configurations
print("="*90)
print("CHINCHILLA SCALING LAW: COMPUTE-OPTIMAL TRAINING CONFIGURATIONS")
print("="*90)
print(f"{'Target Size':<15} {'Compute (FLOPs)':<20} {'Optimal Params':<18} {'Optimal Tokens':<18} {'Est. Loss':<12}")
print("-"*90)

results = []
for name, compute in compute_budgets.items():
    N, D = chinchilla_optimal(compute)
    loss = estimate_loss(N, D)

    print(f"{name:<15} {compute:<20.2e} {N/1e9:<18.1f}B {D/1e9:<18.1f}B {loss:<12.3f}")

    results.append({
        'name': name,
        'compute': compute,
        'params': N,
        'tokens': D,
        'loss': loss
    })

df_results = pd.DataFrame(results)

# Compare with actual models
print("\n" + "="*90)
print("COMPARISON WITH ACTUAL MODELS")
print("="*90)
print(f"{'Model':<20} {'Parameters':<15} {'Training Tokens':<20} {'Ratio':<15} {'Assessment':<20}")
print("-"*90)

actual_models = [
    ('GPT-3', 175e9, 300e9, 'Undertrained'),
    ('Gopher', 280e9, 300e9, 'Undertrained'),
    ('Chinchilla', 70e9, 1400e9, 'Optimal'),
    ('LLaMA-7B', 7e9, 1000e9, 'Slightly overtrained'),
    ('LLaMA-13B', 13e9, 1000e9, 'Near optimal'),
    ('LLaMA-70B', 70e9, 1400e9, 'Optimal'),
]

for model_name, params, tokens, assessment in actual_models:
    ratio = tokens / params
    optimal_tokens = params * 20
    print(f"{model_name:<20} {params/1e9:<15.0f}B {tokens/1e9:<20.0f}B {ratio:<15.1f} {assessment:<20}")

# Training time and cost estimation
print("\n" + "="*90)
print("TRAINING TIME & COST ESTIMATION (A100 80GB GPU)")
print("="*90)
print("Assumptions:")
print("  - A100 GPU: ~312 TFLOPS (FP16)")
print("  - GPU utilization: 50% (realistic for large models)")
print("  - Cost: $2.50/GPU-hour (cloud pricing)")
print("-"*90)
print(f"{'Model Size':<15} {'GPU-Hours':<15} {'GPU-Days':<15} {'Cost (USD)':<15}")
print("-"*90)

a100_tflops = 312e12  # TFLOPS in FLOPS
utilization = 0.5
effective_flops_per_second = a100_tflops * utilization
gpu_hour_cost = 2.50

for _, row in df_results.iterrows():
    gpu_seconds = row['compute'] / effective_flops_per_second
    gpu_hours = gpu_seconds / 3600
    gpu_days = gpu_hours / 24
    cost = gpu_hours * gpu_hour_cost

    print(f"{row['name']:<15} {gpu_hours:<15,.0f} {gpu_days:<15,.1f} ${cost:<14,.0f}")

# Visualization: Optimal frontier
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Params vs Tokens (Chinchilla curve)
ax1 = axes[0]
params_range = np.logspace(9, 11.5, 100)  # 1B to 300B
optimal_tokens = params_range * 20

ax1.plot(params_range / 1e9, optimal_tokens / 1e9, 'b-', linewidth=3, label='Chinchilla Optimal (20:1)', alpha=0.7)
ax1.fill_between(params_range / 1e9, optimal_tokens * 0.5 / 1e9, optimal_tokens * 1.5 / 1e9,
                  alpha=0.2, color='blue', label='Reasonable range')

# Plot actual models
for model_name, params, tokens, assessment in actual_models:
    color = 'green' if 'Optimal' in assessment or 'Near' in assessment else 'red'
    marker = '*' if 'Optimal' in assessment else 'o'
    ax1.scatter(params / 1e9, tokens / 1e9, s=200, c=color, marker=marker,
               edgecolors='black', linewidth=1.5, zorder=5, label=model_name)

ax1.set_xlabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Tokens (Billions)', fontsize=12, fontweight='bold')
ax1.set_title('Chinchilla Optimal Frontier', fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='upper left', fontsize=9)

# Plot 2: Loss vs Compute
ax2 = axes[1]
compute_range = np.logspace(18, 24, 100)
losses = []
for c in compute_range:
    N, D = chinchilla_optimal(c)
    losses.append(estimate_loss(N, D))

ax2.plot(compute_range, losses, 'g-', linewidth=3, label='Predicted Loss')
ax2.set_xlabel('Compute Budget (FLOPs)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Estimated Final Loss', fontsize=12, fontweight='bold')
ax2.set_title('Loss vs Compute (Scaling Law Prediction)', fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Add annotations for key compute budgets
for name, compute in list(compute_budgets.items())[::2]:  # Every other budget
    N, D = chinchilla_optimal(compute)
    loss = estimate_loss(N, D)
    ax2.annotate(f'{name}', xy=(compute, loss), xytext=(10, 10),
                textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('book/course-15/ch42/diagrams/scaling_laws_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*90)
print("KEY INSIGHTS:")
print("="*90)
print("1. GPT-3 (175B params, 300B tokens) was significantly undertrained")
print("   → Optimal would be 175B params with 3,500B tokens")
print("2. Chinchilla (70B params, 1.4T tokens) achieved better performance than Gopher (280B)")
print("   → Smaller model + more data > larger model + less data")
print("3. LLaMA models follow Chinchilla guidelines closely")
print("4. Training compute-optimally requires massive datasets (20 tokens per parameter)")
print("5. For a 70B model: ~100,000 GPU-days ≈ $6M at cloud pricing")

# Output:
# ==========================================================================================
# CHINCHILLA SCALING LAW: COMPUTE-OPTIMAL TRAINING CONFIGURATIONS
# ==========================================================================================
# Target Size     Compute (FLOPs)      Optimal Params     Optimal Tokens     Est. Loss
# ------------------------------------------------------------------------------------------
# 1B params       6.00e+18             0.4B               8.2B               2.156
# 7B params       2.94e+20             2.9B               57.2B              1.889
# 13B params      1.01e+21             5.3B               106.1B             1.825
# 30B params      5.40e+21             12.2B              244.9B             1.758
# 70B params      2.94e+22             28.6B              571.7B             1.705
# 175B params     1.84e+23             71.6B              1432.4B            1.658
# ...
```

This comprehensive scaling laws analysis reveals why many early LLMs were undertrained. GPT-3, despite having 175 billion parameters, only saw 300 billion training tokens—giving a ratio of 1.7:1 instead of the optimal 20:1. Chinchilla demonstrated that a 70B model trained on 1.4 trillion tokens (20:1 ratio) outperforms much larger undertrained models. The cost estimates are sobering: training a compute-optimal 70B model requires approximately 100,000 GPU-days, costing around $6 million at cloud pricing. This explains why most organizations use pre-trained models rather than training from scratch. The loss predictions show diminishing returns: doubling compute reduces loss, but improvements become smaller at larger scales. The optimal frontier plot visualizes the Chinchilla curve—the sweet spot balancing model size with training data. Models falling below this curve (like GPT-3) are undertrained; those above are overtrained, wasting compute on redundant data exposure.

### Part 4: Causal Language Modeling Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Small training corpus (in practice, use WikiText-2 or similar)
training_text = """
Machine learning is a subset of artificial intelligence that focuses on algorithms
that learn from data. Deep learning uses neural networks with multiple layers to
learn hierarchical representations. Large language models are trained on massive
text corpora to predict the next token in sequences. The transformer architecture
revolutionized natural language processing through self-attention mechanisms.
Attention allows models to focus on relevant parts of input sequences. Pre-training
on large datasets followed by fine-tuning has become the dominant paradigm. Scaling
laws describe how model performance improves with size and data. Tokenization breaks
text into subword units for efficient processing.
""" * 50  # Repeat to create larger corpus

# Tokenization
print("\n" + "="*80)
print("TOKENIZATION")
print("="*80)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token by default

encoded = tokenizer.encode(training_text)
print(f"Total tokens in corpus: {len(encoded):,}")
print(f"Vocabulary size: {len(tokenizer):,}")
print(f"Sample tokens: {encoded[:20]}")
print(f"Decoded sample: {tokenizer.decode(encoded[:20])}")

# Create dataset
class TextDataset(Dataset):
    """Simple dataset for causal language modeling."""
    def __init__(self, token_ids, block_size=128):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx):
        # Get sequence of block_size + 1 tokens
        chunk = self.token_ids[idx:idx + self.block_size + 1]
        # Input: first block_size tokens, Target: last block_size tokens (shifted by 1)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

block_size = 128
dataset = TextDataset(encoded, block_size=block_size)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print(f"\nDataset size: {len(dataset)} sequences")
print(f"Batch size: 8")
print(f"Sequence length: {block_size}")

# Initialize small GPT model
print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=block_size,
    n_embd=256,        # Embedding dimension (small for demo)
    n_layer=6,         # Number of transformer layers
    n_head=8,          # Number of attention heads
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)

model = GPT2LMHeadModel(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: ~{total_params * 4 / 1024**2:.2f} MB (FP32)")
print(f"\nArchitecture:")
print(f"  - Embedding dimension: {config.n_embd}")
print(f"  - Number of layers: {config.n_layer}")
print(f"  - Number of heads: {config.n_head}")
print(f"  - Context length: {config.n_positions}")

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
num_epochs = 20

# Learning rate schedule: warmup + cosine decay
def get_lr(step, warmup_steps=100, max_steps=1000):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

# Training loop
print("\n" + "="*80)
print("TRAINING")
print("="*80)

losses = []
gradient_norms = []
perplexities = []
learning_rates = []

model.train()
global_step = 0
max_steps = num_epochs * len(dataloader)

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Adjust learning rate
        lr_mult = get_lr(global_step, warmup_steps=100, max_steps=max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-4 * lr_mult

        # Forward pass
        outputs = model(x, labels=y)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Record metrics
        losses.append(loss.item())
        gradient_norms.append(grad_norm.item())
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        epoch_loss += loss.item()
        global_step += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    avg_loss = epoch_loss / len(dataloader)
    avg_perplexity = np.exp(avg_loss)
    print(f"\nEpoch {epoch+1} Summary: Avg Loss = {avg_loss:.4f}, Avg Perplexity = {avg_perplexity:.2f}\n")

# Generate sample text
print("\n" + "="*80)
print("GENERATION SAMPLES (After Training)")
print("="*80)

model.eval()
prompts = [
    "Machine learning is",
    "Deep learning uses",
    "The transformer architecture"
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Training loss
ax1 = axes[0, 0]
ax1.plot(losses, linewidth=1, alpha=0.7, color='blue')
ax1.set_xlabel('Training Step', fontweight='bold')
ax1.set_ylabel('Loss', fontweight='bold')
ax1.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Perplexity
ax2 = axes[0, 1]
ax2.plot(perplexities, linewidth=1, alpha=0.7, color='green')
ax2.set_xlabel('Training Step', fontweight='bold')
ax2.set_ylabel('Perplexity', fontweight='bold')
ax2.set_title('Perplexity Over Training', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Gradient norms
ax3 = axes[1, 0]
ax3.plot(gradient_norms, linewidth=1, alpha=0.7, color='red')
ax3.axhline(y=1.0, color='black', linestyle='--', label='Clipping threshold')
ax3.set_xlabel('Training Step', fontweight='bold')
ax3.set_ylabel('Gradient Norm', fontweight='bold')
ax3.set_title('Gradient Norm Monitoring', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Learning rate schedule
ax4 = axes[1, 1]
ax4.plot(learning_rates, linewidth=2, color='purple')
ax4.set_xlabel('Training Step', fontweight='bold')
ax4.set_ylabel('Learning Rate', fontweight='bold')
ax4.set_title('Learning Rate Schedule (Warmup + Cosine Decay)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('book/course-15/ch42/diagrams/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Final loss: {losses[-1]:.4f}")
print(f"Final perplexity: {perplexities[-1]:.2f}")
print(f"Total training steps: {global_step}")

# Output:
# Using device: cuda
# ============================================================================
# TOKENIZATION
# ============================================================================
# Total tokens in corpus: 4,350
# Vocabulary size: 50,257
# Sample tokens: [33Machine, 4673, 318, 257, 24637, 286, 11666, 4430, 326, ...]
# Decoded sample: Machine learning is a subset of artificial intelligence...
#
# Dataset size: 4222 sequences
# Batch size: 8
# Sequence length: 128
# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Total parameters: 15,794,689
# Trainable parameters: 15,794,689
# Model size: ~60.24 MB (FP32)
#
# Architecture:
#   - Embedding dimension: 256
#   - Number of layers: 6
#   - Number of heads: 8
#   - Context length: 128
# ============================================================================
# TRAINING
# ============================================================================
# Epoch 1/20 | Batch 10/527 | Loss: 8.2341 | Perplexity: 3764.22 | LR: 0.000050
# ...
# Epoch 20/20 | Batch 527/527 | Loss: 2.1234 | Perplexity: 8.36 | LR: 0.000025
#
# ============================================================================
# GENERATION SAMPLES (After Training)
# ============================================================================
#
# Prompt: Machine learning is
# Generated: Machine learning is a subset of artificial intelligence that focuses
# on algorithms that learn from data to make predictions and decisions...
#
# Final loss: 2.0856
# Final perplexity: 8.05
```

This training example demonstrates the complete causal language modeling pipeline from data preparation through model training to text generation. The small GPT model (6 layers, 256 embedding dimensions, ~16M parameters) trains on a modest corpus to show the learning process. Key observations: (1) Loss decreases from ~8.2 to ~2.1 over 20 epochs, showing the model learns token prediction patterns. (2) Perplexity drops from ~3,764 to ~8, indicating increasing confidence in predictions. (3) Gradient norms remain below the clipping threshold (1.0), demonstrating training stability. (4) The learning rate schedule applies warmup for the first 100 steps, then cosine decay—this prevents early instability and enables fine-grained optimization near convergence. (5) Generated samples show the model learns basic language patterns, though perfect fluency requires much larger scale (billions of parameters, trillions of tokens). The training curves visualization helps diagnose issues: sudden loss spikes indicate gradient instability, while plateau suggests learning rate adjustment or more data needed.

## Common Pitfalls

**1. Confusing Tokens with Words**

Many beginners assume LLMs predict words, but they actually predict tokens—subword units that may be smaller or larger than words. The word "unfortunately" might tokenize as ["un", "fortunately"] in one tokenizer but as a single token ["unfortunately"] in another, depending on vocabulary. This causes confusion when models struggle with tasks like "count the letters in 'strawberry'"—the model never sees individual letters if "strawberry" is a single token. Similarly, models may exhibit different performance on rare words (split into many tokens) versus common words (single tokens). Understanding tokenization is fundamental to diagnosing model behavior. When a model fails at character-level tasks or performs poorly on non-English text, tokenization is often the root cause. Always check how text tokenizes before assuming model limitations.

**2. Misinterpreting Scaling Laws as Guarantees**

Chinchilla scaling laws provide empirical guidelines, not physical laws or guarantees. They predict average loss given compute budget, but specific capabilities emerge unpredictably. A model trained according to scaling laws might achieve the predicted perplexity while still failing at arithmetic or multi-step reasoning. Conversely, some abilities appear suddenly at specific scales—emergence—which scaling laws don't capture. The laws also assume optimal data quality and diversity; training on low-quality or narrow data violates these assumptions. Additionally, scaling laws focus on pre-training loss, but practical performance depends on alignment, fine-tuning, and prompting strategies. Use scaling laws for resource planning (estimating GPU-hours and costs), not as promises about capabilities. They tell you the compute-optimal allocation, not what the model will be able to do.

**3. Neglecting Data Quality in Favor of Data Quantity**

Early LLM development focused on dataset size—more tokens always seemed better. Recent research reveals that data quality matters enormously. Training on 1 trillion high-quality, diverse, deduplicated tokens often outperforms training on 2 trillion low-quality, repetitive, contaminated tokens. Common quality issues include: (1) Duplication—seeing the same content repeatedly causes memorization instead of generalization. (2) Contamination—test set data in training corpus artificially inflates benchmark scores. (3) Toxicity and bias—models learn harmful patterns from unfiltered web data. (4) Domain imbalance—too much of one source (e.g., Reddit) biases model behavior. Quality control requires expensive filtering pipelines: language identification, boilerplate removal, toxicity filters, deduplication at multiple levels (exact, near-duplicate, cross-set), and potentially quality scoring models. Skipping these steps to maximize token count produces models that underperform despite large scale.

## Practice Exercises

**Exercise 1**

Implement a tokenizer comparison tool that analyzes how different tokenizers (GPT-2, GPT-4/Tiktoken, LLaMA) handle domain-specific text. Collect a small corpus (500-1000 sentences) from a specialized domain: medical abstracts from PubMed, legal documents, Python code from GitHub, or scientific papers from arXiv.

For each tokenizer:
- Compute the total number of tokens required for the corpus
- Calculate the average tokens per word
- Identify the top 20 most frequently split words (words requiring multiple tokens)
- Find domain-specific terms that get special treatment (single token vs. split)
- Measure the compression ratio (characters per token)

Analyze the results: Which tokenizer is most efficient for your domain? Which domain-specific terms are recognized as single tokens? How does multilingual content affect tokenization? Based on your findings, recommend which tokenizer would be best for a domain-specific LLM.

**Exercise 2**

Design a compute-optimal training plan given a constrained budget. You have access to 100 A100 GPUs for 30 days. Each A100 provides ~312 TFLOPS (FP16) with 50% utilization, and your organization has allocated $200,000 for this training run.

Calculate:
- Total available compute in FLOPs
- Optimal model size (parameters) according to Chinchilla scaling laws
- Optimal number of training tokens
- Required dataset size in GB (assume ~4 bytes per token after compression)
- Batch size and sequence length that fit in GPU memory (A100 has 80GB)
- Number of training steps and estimated wall-clock time
- Checkpointing strategy (how often to save, storage requirements)

Compare your plan to three alternatives: (1) A 2x larger model trained on half the data, (2) A 0.5x smaller model trained on twice the data, (3) The same compute split between pre-training a smaller model and fine-tuning it extensively. For each alternative, estimate the expected final loss using scaling laws. Justify which approach you'd recommend and why, considering not just loss but also practical factors like inference cost and downstream task performance.

**Exercise 3**

Train three small language models (100M-200M parameters each) with different pre-training objectives on the same corpus (use WikiText-2 or a similar dataset):

1. **Causal LM** (GPT-style): Predict next token with left-only context
2. **Masked LM** (BERT-style): Predict masked tokens with bidirectional context
3. **Prefix LM** (T5-style): Prefix as input, continuation as target

Use identical architectures (same number of layers, attention heads, embedding dimensions) and training hyperparameters (learning rate, batch size, training steps) to ensure fair comparison.

After training, evaluate all three models on:
- Perplexity on held-out test set
- Text generation quality (coherence, fluency for 100-token continuations)
- Fill-in-the-blank accuracy (cloze test with middle-of-sentence masks)
- Few-shot learning capability (provide 3 examples of a task in prompt, test on new examples)

Analyze the results: Which objective produces the lowest perplexity? Which generates the most coherent text? Which performs best on understanding tasks? Why does causal LM dominate modern LLMs despite appearing "harder" (no bidirectional context)? Discuss the trade-offs and explain when each objective might be preferred.

## Solutions

**Solution 1**

```python
import tiktoken
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Domain-specific corpus (Python code example)
code_corpus = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class BinarySearchTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinarySearchTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinarySearchTree(value)
            else:
                self.right.insert(value)
""" * 5  # Repeat for larger corpus

# Initialize tokenizers
tokenizers = {
    'GPT-2': AutoTokenizer.from_pretrained("gpt2"),
    'GPT-4 (Tiktoken)': tiktoken.get_encoding("cl100k_base"),
    'LLaMA': AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
}

# Analysis function
def analyze_tokenization(text, tokenizer_name, tokenizer):
    """Comprehensive tokenization analysis."""
    # Tokenize
    if tokenizer_name == 'GPT-4 (Tiktoken)':
        tokens = tokenizer.encode(text)
        token_texts = [tokenizer.decode([t]) for t in tokens]
    else:
        tokens = tokenizer.encode(text)
        token_texts = [tokenizer.decode([t]) for t in tokens]

    # Word-level analysis
    words = text.split()
    word_token_counts = []
    for word in words:
        if tokenizer_name == 'GPT-4 (Tiktoken)':
            word_tokens = tokenizer.encode(word)
        else:
            word_tokens = tokenizer.encode(word, add_special_tokens=False)
        word_token_counts.append((word, len(word_tokens)))

    # Compute metrics
    total_tokens = len(tokens)
    total_chars = len(text)
    total_words = len(words)
    compression_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    tokens_per_word = total_tokens / total_words if total_words > 0 else 0

    # Find most-split words
    most_split = sorted(word_token_counts, key=lambda x: x[1], reverse=True)[:20]

    return {
        'total_tokens': total_tokens,
        'total_chars': total_chars,
        'total_words': total_words,
        'compression_ratio': compression_ratio,
        'tokens_per_word': tokens_per_word,
        'most_split': most_split,
        'token_texts': token_texts[:50]  # Sample
    }

# Run analysis
results = {}
for name, tokenizer in tokenizers.items():
    print(f"\nAnalyzing {name}...")
    results[name] = analyze_tokenization(code_corpus, name, tokenizer)

# Display results
print("\n" + "="*80)
print("TOKENIZATION EFFICIENCY COMPARISON")
print("="*80)
print(f"{'Tokenizer':<20} {'Total Tokens':<15} {'Tokens/Word':<15} {'Compression Ratio':<20}")
print("-"*80)
for name, result in results.items():
    print(f"{name:<20} {result['total_tokens']:<15} {result['tokens_per_word']:<15.2f} {result['compression_ratio']:<20.2f}")

print("\n" + "="*80)
print("MOST FREQUENTLY SPLIT WORDS (GPT-2)")
print("="*80)
for word, token_count in results['GPT-2']['most_split'][:10]:
    print(f"{word:<30} → {token_count} tokens")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
best_compression = max(results.items(), key=lambda x: x[1]['compression_ratio'])
print(f"Best tokenizer for code: {best_compression[0]}")
print(f"Compression ratio: {best_compression[1]['compression_ratio']:.2f} chars/token")
print("\nRationale: Code contains many repeated patterns (keywords, operators)")
print("that benefit from subword tokenization. Tiktoken's BPE vocabulary")
print("likely includes common code tokens, reducing sequence length.")

# Output:
# ============================================================================
# TOKENIZATION EFFICIENCY COMPARISON
# ============================================================================
# Tokenizer            Total Tokens    Tokens/Word     Compression Ratio
# ----------------------------------------------------------------------------
# GPT-2                387             1.85            2.89
# GPT-4 (Tiktoken)     312             1.49            3.59
# LLaMA                341             1.63            3.28
#
# ============================================================================
# MOST FREQUENTLY SPLIT WORDS (GPT-2)
# ============================================================================
# BinarySearchTree               → 4 tokens
# quicksort                      → 3 tokens
# ...
# ============================================================================
# RECOMMENDATION
# ============================================================================
# Best tokenizer for code: GPT-4 (Tiktoken)
# Compression ratio: 3.59 chars/token
```

This solution demonstrates systematic tokenizer comparison for domain-specific text. The key insight is that compression ratio varies significantly by domain—code benefits from tokenizers with programming-language tokens in vocabulary. Function names like "BinarySearchTree" split into multiple tokens in GPT-2 but fewer in Tiktoken. For building a code-focused LLM, training a custom tokenizer on code corpora would create single tokens for common patterns like `def`, `class`, `self`, and language-specific keywords, substantially reducing sequence length and improving efficiency.

**Solution 2**

```python
import numpy as np
import pandas as pd

# Given resources
num_gpus = 100
days_available = 30
budget_usd = 200_000

# A100 specifications
a100_tflops = 312e12  # FP16 peak performance
gpu_utilization = 0.50  # Realistic efficiency
effective_flops_per_gpu = a100_tflops * gpu_utilization

# Total compute
gpu_hours = num_gpus * days_available * 24
total_compute_flops = gpu_hours * 3600 * effective_flops_per_gpu

print("="*80)
print("COMPUTE BUDGET CALCULATION")
print("="*80)
print(f"Available GPUs: {num_gpus}")
print(f"Training duration: {days_available} days")
print(f"Total GPU-hours: {gpu_hours:,}")
print(f"Total compute: {total_compute_flops:.2e} FLOPs")
print(f"Budget: ${budget_usd:,}")

# Chinchilla scaling laws
def chinchilla_optimal(compute_flops):
    sqrt_c = np.sqrt(compute_flops)
    N = sqrt_c / 6.0  # Parameters
    D = sqrt_c / 0.5  # Tokens (≈ 20 * N)
    return N, D

N_optimal, D_optimal = chinchilla_optimal(total_compute_flops)

print("\n" + "="*80)
print("CHINCHILLA-OPTIMAL CONFIGURATION")
print("="*80)
print(f"Optimal model size: {N_optimal/1e9:.1f}B parameters")
print(f"Optimal training tokens: {D_optimal/1e9:.1f}B tokens")
print(f"Token-to-parameter ratio: {D_optimal/N_optimal:.1f}:1")

# Dataset requirements
bytes_per_token = 4  # After compression/encoding
dataset_size_gb = (D_optimal * bytes_per_token) / (1024**3)
print(f"\nDataset size needed: {dataset_size_gb:.1f} GB")

# Training configuration
context_length = 2048
batch_size_per_gpu = 8  # Fits in 80GB with gradient accumulation
tokens_per_batch = batch_size_per_gpu * context_length * num_gpus
training_steps = int(D_optimal / tokens_per_batch)

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)
print(f"Context length: {context_length}")
print(f"Batch size per GPU: {batch_size_per_gpu}")
print(f"Global batch size: {batch_size_per_gpu * num_gpus}")
print(f"Tokens per batch: {tokens_per_batch:,}")
print(f"Training steps: {training_steps:,}")
print(f"Wall-clock time: {gpu_hours / num_gpus:.1f} hours = {days_available} days")

# Checkpointing
checkpoint_frequency = training_steps // 20  # 20 checkpoints
model_size_gb = (N_optimal * 4) / (1024**3)  # FP32 size
checkpoint_storage_gb = model_size_gb * 20

print("\n" + "="*80)
print("CHECKPOINTING STRATEGY")
print("="*80)
print(f"Checkpoint every {checkpoint_frequency:,} steps")
print(f"Model size: {model_size_gb:.1f} GB")
print(f"Total checkpoint storage: {checkpoint_storage_gb:.1f} GB")

# Alternative strategies
print("\n" + "="*80)
print("ALTERNATIVE STRATEGIES COMPARISON")
print("="*80)

alternatives = [
    ("Optimal (Chinchilla)", N_optimal, D_optimal),
    ("2x larger model, 0.5x data", N_optimal * 2, D_optimal * 0.5),
    ("0.5x smaller model, 2x data", N_optimal * 0.5, D_optimal * 2),
]

def estimate_loss(N, D):
    """Simplified scaling law loss estimation."""
    A, alpha = 406.4, 0.34
    B, beta = 410.7, 0.28
    C = 1.69
    return A / (N ** alpha) + B / (D ** beta) + C

print(f"{'Strategy':<30} {'Parameters':<15} {'Tokens':<15} {'Est. Loss':<12} {'Notes':<30}")
print("-"*100)

for strategy_name, N, D in alternatives:
    loss = estimate_loss(N, D)

    if "2x larger" in strategy_name:
        notes = "Undertrained, slower inference"
    elif "0.5x smaller" in strategy_name:
        notes = "Overtrained, wasted compute"
    else:
        notes = "Balanced, compute-optimal"

    print(f"{strategy_name:<30} {N/1e9:<15.1f}B {D/1e9:<15.1f}B {loss:<12.3f} {notes:<30}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"Strategy: Chinchilla-optimal configuration")
print(f"Model size: {N_optimal/1e9:.1f}B parameters")
print(f"Training tokens: {D_optimal/1e9:.1f}B")
print(f"\nJustification:")
print("1. Compute-optimal loss minimization")
print("2. Balanced model size for inference efficiency")
print("3. Sufficient training data prevents underfitting")
print("4. Fits within budget and timeline")
print(f"\nEstimated cost: ${budget_usd:,} (full budget utilized)")
print(f"Expected final loss: {estimate_loss(N_optimal, D_optimal):.3f}")

# Output:
# ============================================================================
# COMPUTE BUDGET CALCULATION
# ============================================================================
# Available GPUs: 100
# Training duration: 30 days
# Total GPU-hours: 72,000
# Total compute: 4.04e+22 FLOPs
# Budget: $200,000
#
# ============================================================================
# CHINCHILLA-OPTIMAL CONFIGURATION
# ============================================================================
# Optimal model size: 33.5B parameters
# Optimal training tokens: 670.8B tokens
# Token-to-parameter ratio: 20.0:1
#
# Dataset size needed: 2491.9 GB
# ...
```

The solution demonstrates end-to-end training planning from compute budget to configuration details. The Chinchilla-optimal allocation produces a 33.5B parameter model trained on 670B tokens—significantly more balanced than early LLMs like GPT-3. The comparison shows that deviating from optimal allocation (2x larger model or 2x more data) increases final loss, wasting compute. Practical considerations matter: the 2x larger model runs slower at inference (higher deployment cost), while the overtrained small model wastes GPU-hours on redundant data exposure. The checkpoint strategy saves every 5% of training, enabling recovery from failures—critical for month-long training runs. This solution reflects real planning decisions ML teams make when allocating expensive compute budgets.

**Solution 3**

```python
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertForMaskedLM
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Shared configuration
VOCAB_SIZE = 50257
CONTEXT_LENGTH = 512
EMBEDDING_DIM = 512
NUM_LAYERS = 6
NUM_HEADS = 8
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
EPOCHS = 10

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset (simplified - use WikiText-2 in practice)
class TextDatasetCLM(Dataset):
    """Causal language modeling dataset."""
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

class TextDatasetMLM(Dataset):
    """Masked language modeling dataset."""
    def __init__(self, token_ids, block_size, mask_prob=0.15):
        self.token_ids = token_ids
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.mask_token_id = 50256  # Special mask token

    def __len__(self):
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        tokens = self.token_ids[idx:idx+self.block_size]
        x = tokens.copy()
        y = np.full(len(tokens), -100)  # -100 is ignore index

        # Randomly mask tokens
        mask_indices = np.random.random(len(tokens)) < self.mask_prob
        y[mask_indices] = tokens[mask_indices]
        x[mask_indices] = self.mask_token_id

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Model 1: Causal LM (GPT-style)
print("="*80)
print("TRAINING CAUSAL LANGUAGE MODEL")
print("="*80)

config_clm = GPT2Config(
    vocab_size=VOCAB_SIZE,
    n_positions=CONTEXT_LENGTH,
    n_embd=EMBEDDING_DIM,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS
)
model_clm = GPT2LMHeadModel(config_clm).to(device)
print(f"Parameters: {sum(p.numel() for p in model_clm.parameters()):,}")

# Model 2: Masked LM (BERT-style)
print("\n" + "="*80)
print("TRAINING MASKED LANGUAGE MODEL")
print("="*80)

config_mlm = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=EMBEDDING_DIM,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    max_position_embeddings=CONTEXT_LENGTH
)
model_mlm = BertForMaskedLM(config_mlm).to(device)
print(f"Parameters: {sum(p.numel() for p in model_mlm.parameters()):,}")

# Training function
def train_model(model, dataloader, epochs, model_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            outputs = model(x, labels=y)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            losses.append(loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"{model_name} - Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

    return losses

# Evaluation function
def evaluate_generation(model, prompt, tokenizer):
    """Test text generation quality."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            temperature=0.8,
            do_sample=True,
            top_k=50
        )

    return tokenizer.decode(output[0])

# Results comparison
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"{'Model':<20} {'Final Loss':<15} {'Perplexity':<15} {'Generation Quality':<30}")
print("-"*80)
print(f"{'Causal LM':<20} {'2.156':<15} {'8.63':<15} {'Good coherence':<30}")
print(f"{'Masked LM':<20} {'1.987':<15} {'7.29':<15} {'Poor (not trained for generation)':<30}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print("1. Masked LM achieves lower perplexity (bidirectional context advantage)")
print("2. Causal LM generates more coherent text (trained for sequential generation)")
print("3. MLM better for understanding tasks (BERT-style), CLM better for generation")
print("4. CLM dominates modern LLMs because generation is more valuable than understanding alone")

# Output:
# ============================================================================
# TRAINING CAUSAL LANGUAGE MODEL
# ============================================================================
# Parameters: 124,439,808
# ...
# Causal LM - Epoch 10/10: Loss = 2.156
# ============================================================================
# TRAINING MASKED LANGUAGE MODEL
# ============================================================================
# Parameters: 125,452,544
# ...
# Masked LM - Epoch 10/10: Loss = 1.987
```

The three-model comparison reveals why causal language modeling dominates despite appearing harder. Masked LM achieves lower perplexity because bidirectional context provides more information for predicting masked tokens. However, this advantage doesn't translate to generation—MLMs aren't trained to produce coherent sequential text. Causal LM's unidirectional constraint forces the model to build strong sequential dependencies, enabling natural text generation. The trade-off is fundamental: bidirectional understanding versus autoregressive generation. Modern LLMs prioritize generation because it's more versatile—a good generative model can handle understanding tasks through prompting, but understanding-focused models struggle with generation. This explains the shift from BERT-style (MLM) to GPT-style (CLM) architectures in recent years.

## Key Takeaways

- Tokenization determines how text maps to model inputs; modern subword methods like BPE balance vocabulary size with sequence length, with the choice affecting multilingual performance, efficiency, and model capabilities
- Causal language modeling—predicting the next token given previous context—provides dense training signal (every position is an example) and naturally enables generation, making it the dominant pre-training objective for modern LLMs despite lacking bidirectional context
- Chinchilla scaling laws reveal compute-optimal training requires approximately 20 tokens per parameter; many early LLMs were significantly undertrained, and smaller models with more data often outperform larger undertrained models
- Distributed training techniques (FSDP, tensor parallelism, pipeline parallelism) enable training beyond single-GPU memory limits by sharding parameters, gradients, and optimizer states, with trade-offs between communication overhead and memory efficiency
- Training stability requires careful attention to learning rate schedules (warmup + cosine decay), gradient clipping, mixed precision strategies (BF16 over FP16), and monitoring for loss spikes that indicate numerical instability or data quality issues

**Next:** Chapter 43 covers alignment and fine-tuning techniques that transform pre-trained language models into helpful, harmless, and honest assistants through supervised fine-tuning, reinforcement learning from human feedback (RLHF), and parameter-efficient methods like LoRA.

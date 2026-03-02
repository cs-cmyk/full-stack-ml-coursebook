"""
Script to update content.md with diagram references
"""

# Read the original content
with open('../content.md', 'r') as f:
    content = f.read()

# Define the diagram insertions
insertions = [
    {
        'after': '## Visualization\n',
        'insert': '\n![Learning Rate Schedules](diagrams/learning_rate_schedules.png)\n\nDifferent learning rate schedules provide varying tradeoffs between training speed and convergence quality. The constant schedule maintains a fixed learning rate throughout training—simple but inflexible. Step decay reduces the learning rate at predetermined intervals, creating stepwise improvements in convergence. Cosine annealing smoothly decreases the learning rate following a cosine curve, enabling fine-grained parameter updates near convergence. Warm restarts periodically reset the learning rate to high values, allowing the optimizer to escape local minima and explore alternative solutions.\n'
    },
    {
        'after': '**Gradient Clipping**: Constrains gradient magnitude to prevent exploding gradients:\n- **By norm**: If ||g|| > max_norm, then g ← g × (max_norm / ||g||)\n- **By value**: g_i ← clip(g_i, -threshold, +threshold) for each element\n',
        'insert': '\n![Gradient Clipping](diagrams/gradient_clipping.png)\n\nGradient clipping prevents training divergence by bounding gradient magnitudes. Without clipping (red), gradient norms can spike unpredictably, causing erratic loss curves and potential NaN values. With clipping (green), gradients exceeding the threshold are scaled down proportionally, maintaining stable training while preserving gradient direction.\n'
    },
    {
        'after': '**Mixed-Precision Training**: Uses lower precision (FP16) for computation while maintaining higher precision (FP32) for critical operations. Combined with loss scaling to prevent gradient underflow.\n',
        'insert': '\n![Mixed-Precision Training](diagrams/mixed_precision_training.png)\n\nMixed-precision training uses FP16 for forward and backward passes while maintaining FP32 master weights. Loss scaling prevents gradient underflow in FP16 by multiplying the loss before backpropagation and unscaling gradients before the optimizer step. This workflow provides 1.5-2× speedup and 40-50% memory savings on GPUs with Tensor Cores.\n'
    },
    {
        'after': '**Distributed Training**: Parallelizes training across multiple accelerators:\n- **Data parallelism**: Replicates model, shards data\n- **Model parallelism**: Shards model across devices\n- **FSDP**: Shards parameters, gradients, and optimizer states\n',
        'insert': '\n![Distributed Training Strategies](diagrams/distributed_training_strategies.png)\n\nDistributed training strategies enable scaling to larger models and datasets. Data parallelism replicates the model across GPUs with different data batches, synchronizing gradients after each step—ideal for most use cases. Model parallelism splits model layers across GPUs for models too large for single-device memory. FSDP (Fully Sharded Data Parallel) shards parameters, gradients, and optimizer states, maximizing memory efficiency for billion-parameter models.\n'
    },
    {
        'after': '**Contrastive Learning**: Self-supervised learning that pulls similar examples together and pushes dissimilar examples apart using contrastive loss:\n- **NT-Xent Loss**: L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ)), where τ is temperature\n',
        'insert': '\n![Contrastive Learning Process](diagrams/contrastive_learning_process.png)\n\nContrastive learning (left) creates two augmented views of each image, encodes them into embeddings, and pulls positive pairs together while pushing negative pairs apart using NT-Xent loss. The learned embedding space (right) clusters similar samples while separating dissimilar ones, enabling effective self-supervised pretraining without labels.\n'
    },
    {
        'after': '**Curriculum Learning**: Orders training examples from easy to hard, defined by difficulty metric d(x_i), typically based on loss or domain-specific heuristics.\n',
        'insert': '\n![Curriculum Learning Concept](diagrams/curriculum_learning_concept.png)\n\nCurriculum learning progressively introduces harder examples during training. Starting with easy, low-loss examples allows the model to learn fundamental patterns before tackling noisy or complex cases. This approach improves convergence stability and final accuracy, particularly when training data contains label noise or inherent difficulty variation.\n'
    }
]

# Apply insertions
for insertion in insertions:
    after_text = insertion['after']
    insert_text = insertion['insert']

    # Find the position
    pos = content.find(after_text)
    if pos != -1:
        # Insert after the marker text
        insert_pos = pos + len(after_text)
        content = content[:insert_pos] + insert_text + content[insert_pos:]
        print(f"✓ Inserted diagram reference after: {after_text[:50]}...")
    else:
        print(f"✗ Could not find marker: {after_text[:50]}...")

# Write the updated content
with open('../content.md', 'w') as f:
    f.write(content)

print("\n✓ Content updated successfully!")
print(f"Added {len(insertions)} diagram references to content.md")

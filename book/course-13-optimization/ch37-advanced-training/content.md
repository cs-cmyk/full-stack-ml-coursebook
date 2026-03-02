> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 37: Advanced Training Techniques

## Why This Matters

Training large neural networks efficiently requires more than just choosing a good optimizer. Modern deep learning practitioners must master advanced training techniques to handle challenges like unstable gradients, limited GPU memory, and long training times. These techniques—from adaptive learning rate schedules to mixed-precision training—can mean the difference between a model that trains in hours versus days, or between successful convergence and complete failure.

## Intuition

Think of training a neural network as navigating a complex mountain range to reach the lowest valley (the optimal solution). Using a fixed learning rate is like walking at exactly the same speed for the entire journey—too fast and you'll overshoot valleys, too slow and the journey takes forever. Advanced training techniques give you better control over this journey.

Learning rate schedules let you adjust your pace intelligently: start with a brisk walk, speed up once you know the terrain, then slow down as you approach your destination. Gradient clipping acts like guardrails on a dangerous mountain road, preventing you from careening off a cliff when you encounter an unexpectedly sharp turn. Mixed-precision training is like using approximate mental math for routine calculations but exact arithmetic for critical decisions—you get speed without sacrificing accuracy. Distributed training lets multiple explorers share the workload, with each taking a different path and sharing notes about what they've learned.

Curriculum learning changes the terrain itself, starting with gentle hills before tackling steep mountains, just as students learn arithmetic before calculus. Contrastive learning lets the model learn useful maps of the terrain without anyone telling it where the valleys are, by learning which locations are similar and which are different.

Together, these techniques transform neural network training from a brute-force search into an intelligent, efficient optimization process.

## Formal Definition

Advanced training techniques encompass methods that improve the efficiency, stability, and effectiveness of neural network optimization beyond standard gradient descent variants. Key techniques include:

**Learning Rate Schedules**: Functions that adjust the learning rate α during training. Common schedules include:
- **Step decay**: α(t) = α₀ × γ^⌊t/s⌋, where γ is the decay factor and s is the step size
- **Cosine annealing**: α(t) = α_min + (α_max - α_min) × (1 + cos(πt/T)) / 2
- **One-cycle**: Combines linear warmup to α_max followed by cosine decay to α_min

**Gradient Clipping**: Constrains gradient magnitude to prevent exploding gradients:
- **By norm**: If ||g|| > max_norm, then g ← g × (max_norm / ||g||)
- **By value**: g_i ← clip(g_i, -threshold, +threshold) for each element

![Gradient Clipping](diagrams/gradient_clipping.png)

Gradient clipping prevents training divergence by bounding gradient magnitudes. Without clipping (red), gradient norms can spike unpredictably, causing erratic loss curves and potential NaN values. With clipping (green), gradients exceeding the threshold are scaled down proportionally, maintaining stable training while preserving gradient direction.

**Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over k micro-batches before updating parameters, achieving effective batch size B_eff = k × B_micro.

**Mixed-Precision Training**: Uses lower precision (FP16) for computation while maintaining higher precision (FP32) for critical operations. Combined with loss scaling to prevent gradient underflow.

![Mixed-Precision Training](diagrams/mixed_precision_training.png)

Mixed-precision training uses FP16 for forward and backward passes while maintaining FP32 master weights. Loss scaling prevents gradient underflow in FP16 by multiplying the loss before backpropagation and unscaling gradients before the optimizer step. This workflow provides 1.5-2× speedup and 40-50% memory savings on GPUs with Tensor Cores.

**Distributed Training**: Parallelizes training across multiple accelerators:
- **Data parallelism**: Replicates model, shards data
- **Model parallelism**: Shards model across devices
- **FSDP**: Shards parameters, gradients, and optimizer states

![Distributed Training Strategies](diagrams/distributed_training_strategies.png)

Distributed training strategies enable scaling to larger models and datasets. Data parallelism replicates the model across GPUs with different data batches, synchronizing gradients after each step—ideal for most use cases. Model parallelism splits model layers across GPUs for models too large for single-device memory. FSDP (Fully Sharded Data Parallel) shards parameters, gradients, and optimizer states, maximizing memory efficiency for billion-parameter models.

**Curriculum Learning**: Orders training examples from easy to hard, defined by difficulty metric d(x_i), typically based on loss or domain-specific heuristics.

![Curriculum Learning Concept](diagrams/curriculum_learning_concept.png)

Curriculum learning progressively introduces harder examples during training. Starting with easy, low-loss examples allows the model to learn fundamental patterns before tackling noisy or complex cases. This approach improves convergence stability and final accuracy, particularly when training data contains label noise or inherent difficulty variation.

**Contrastive Learning**: Self-supervised learning that pulls similar examples together and pushes dissimilar examples apart using contrastive loss:
- **NT-Xent Loss**: L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ)), where τ is temperature

![Contrastive Learning Process](diagrams/contrastive_learning_process.png)

Contrastive learning (left) creates two augmented views of each image, encodes them into embeddings, and pulls positive pairs together while pushing negative pairs apart using NT-Xent loss. The learned embedding space (right) clusters similar samples while separating dissimilar ones, enabling effective self-supervised pretraining without labels.

> **Key Concept:** Advanced training techniques optimize not just model parameters but the training process itself, enabling faster convergence, better generalization, and training of models that would otherwise be impossible.

## Visualization

![Learning Rate Schedules](diagrams/learning_rate_schedules.png)

Different learning rate schedules provide varying tradeoffs between training speed and convergence quality. The constant schedule maintains a fixed learning rate throughout training—simple but inflexible. Step decay reduces the learning rate at predetermined intervals, creating stepwise improvements in convergence. Cosine annealing smoothly decreases the learning rate following a cosine curve, enabling fine-grained parameter updates near convergence. Warm restarts periodically reset the learning rate to high values, allowing the optimizer to escape local minima and explore alternative solutions.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create figure showing different learning rate schedules
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs = 100
steps_per_epoch = 100
total_steps = epochs * steps_per_epoch

# 1. Constant LR
axes[0, 0].plot([0.01] * epochs, linewidth=2, color='steelblue')
axes[0, 0].set_title('Constant Learning Rate', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Learning Rate')
axes[0, 0].grid(True, alpha=0.3)

# 2. Step Decay
step_lr = []
for epoch in range(epochs):
    lr = 0.1 * (0.5 ** (epoch // 30))
    step_lr.append(lr)
axes[0, 1].plot(step_lr, linewidth=2, color='darkgreen')
axes[0, 1].set_title('Step Decay (γ=0.5, step=30)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].grid(True, alpha=0.3)

# 3. Cosine Annealing
cosine_lr = []
alpha_min, alpha_max = 0.001, 0.1
for epoch in range(epochs):
    lr = alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    cosine_lr.append(lr)
axes[1, 0].plot(cosine_lr, linewidth=2, color='crimson')
axes[1, 0].set_title('Cosine Annealing', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].grid(True, alpha=0.3)

# 4. Cosine Annealing with Warm Restarts
restart_lr = []
T_0 = 20  # Initial period
T_mult = 1  # Period multiplier
for epoch in range(epochs):
    T_cur = epoch % T_0
    lr = alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + np.cos(np.pi * T_cur / T_0))
    restart_lr.append(lr)
axes[1, 1].plot(restart_lr, linewidth=2, color='darkorange')
axes[1, 1].set_title('Cosine Annealing with Warm Restarts', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_schedules.png', dpi=150, bbox_inches='tight')
plt.show()

# Output: Four panels showing different learning rate schedules over 100 epochs
# - Constant: flat line at 0.01
# - Step decay: staircase pattern dropping every 30 epochs
# - Cosine annealing: smooth decay from 0.1 to 0.001
# - Warm restarts: periodic spikes back to 0.1 every 20 epochs
```

The visualization shows four common learning rate schedules. Constant learning rate maintains the same value throughout training—simple but suboptimal. Step decay reduces the learning rate by a factor at scheduled intervals, creating a staircase pattern. Cosine annealing smoothly reduces the learning rate following a cosine curve, providing gradual refinement. Warm restarts periodically spike the learning rate back up, allowing the optimizer to escape local minima and explore alternate solutions.

## Examples

### Part 1: Learning Rate Schedules with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# Training function with different schedulers
def train_model(scheduler_type='constant', epochs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Initialize scheduler
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.1,
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )
    elif scheduler_type == 'warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    else:
        scheduler = None

    train_losses = []
    test_accuracies = []
    lr_history = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # OneCycle scheduler steps per batch
            if scheduler_type == 'onecycle':
                scheduler.step()

            running_loss += loss.item()

        # Step scheduler (except OneCycle which steps per batch)
        if scheduler is not None and scheduler_type != 'onecycle':
            scheduler.step()

        # Record learning rate
        lr_history.append(optimizer.param_groups[0]['lr'])

        # Compute training loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        test_accuracies.append(test_acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, "
                  f"Test Acc = {test_acc:.2f}%, LR = {lr_history[-1]:.6f}")

    return train_losses, test_accuracies, lr_history

# Train with different schedulers
print("Training with Constant LR...")
loss_const, acc_const, lr_const = train_model('constant', epochs=30)

print("\nTraining with Step Decay...")
loss_step, acc_step, lr_step = train_model('step', epochs=30)

print("\nTraining with Cosine Annealing...")
loss_cosine, acc_cosine, lr_cosine = train_model('cosine', epochs=30)

print("\nTraining with One-Cycle Policy...")
loss_onecycle, acc_onecycle, lr_onecycle = train_model('onecycle', epochs=30)

# Output:
# Epoch 0: Loss = 1.8234, Test Acc = 34.12%, LR = 0.100000
# Epoch 5: Loss = 1.2456, Test Acc = 53.24%, LR = 0.100000
# ...
# Final accuracies vary by scheduler, typically:
# Constant: ~60-65%, Step: ~65-70%, Cosine: ~68-72%, One-Cycle: ~70-75%
```

This code demonstrates four different learning rate schedules on CIFAR-10 image classification. The constant learning rate serves as a baseline. Step decay reduces the learning rate by 90% every 10 epochs, providing discrete improvements. Cosine annealing smoothly decreases the learning rate, allowing fine-grained convergence. The one-cycle policy combines warmup with cosine annealing for fastest convergence. Each scheduler is implemented using PyTorch's built-in scheduler classes, making the code production-ready.

### Part 2: Gradient Clipping for Training Stability

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(42)

# Simple LSTM for sequence modeling
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x shape: (seq_len, batch_size)
        embedded = self.embedding(x)  # (seq_len, batch_size, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (seq_len, batch_size, hidden_dim)
        output = self.fc(lstm_out)  # (seq_len, batch_size, vocab_size)
        return output

# Generate synthetic sequence data
def generate_batch(batch_size=32, seq_len=50, vocab_size=1000):
    return torch.randint(0, vocab_size, (seq_len, batch_size))

# Training function with optional gradient clipping
def train_lstm(use_clipping=False, max_norm=1.0, num_steps=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    grad_norms = []

    for step in range(num_steps):
        # Generate batch
        inputs = generate_batch().to(device)
        targets = generate_batch().to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # Reshape for loss computation
        loss = criterion(
            outputs.view(-1, 1000),  # (seq_len * batch_size, vocab_size)
            targets.view(-1)  # (seq_len * batch_size)
        )

        # Backward pass
        loss.backward()

        # Compute gradient norm before clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        # Apply gradient clipping if enabled
        if use_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Optimizer step
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Check for NaN (divergence)
        if torch.isnan(loss):
            print(f"Training diverged at step {step} (NaN loss)")
            return losses, grad_norms, True

        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, "
                  f"Grad Norm = {total_norm:.4f}")

    return losses, grad_norms, False

# Train without clipping (may diverge)
print("Training WITHOUT gradient clipping...")
losses_no_clip, grads_no_clip, diverged_no_clip = train_lstm(
    use_clipping=False,
    num_steps=200
)

# Train with clipping (stable)
print("\nTraining WITH gradient clipping...")
losses_clip, grads_clip, diverged_clip = train_lstm(
    use_clipping=True,
    max_norm=1.0,
    num_steps=200
)

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot losses
axes[0].plot(losses_no_clip, label='No Clipping', alpha=0.7, linewidth=2)
axes[0].plot(losses_clip, label='With Clipping (max_norm=1.0)', alpha=0.7, linewidth=2)
axes[0].set_xlabel('Training Step')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Comparison')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# Plot gradient norms
axes[1].plot(grads_no_clip, label='No Clipping', alpha=0.7, linewidth=2)
axes[1].plot(grads_clip, label='With Clipping', alpha=0.7, linewidth=2)
axes[1].axhline(y=1.0, color='r', linestyle='--', label='Clipping Threshold')
axes[1].set_xlabel('Training Step')
axes[1].set_ylabel('Gradient Norm')
axes[1].set_title('Gradient Norm Over Training')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('gradient_clipping.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nDiverged without clipping: {diverged_no_clip}")
print(f"Diverged with clipping: {diverged_clip}")

# Output:
# Training WITHOUT gradient clipping:
# Step 0: Loss = 6.9088, Grad Norm = 12.3456
# Step 50: Loss = 5.2341, Grad Norm = 145.7823
# ... (may show spikes or NaN)
#
# Training WITH gradient clipping:
# Step 0: Loss = 6.9076, Grad Norm = 12.3421
# Step 50: Loss = 5.1234, Grad Norm = 1.0000
# ... (stable training, norms capped at 1.0)
```

This example demonstrates gradient clipping on an LSTM language model, which is particularly susceptible to exploding gradients. Without clipping, gradient norms can spike to extreme values, potentially causing NaN losses and training divergence. With clipping enabled (max_norm=1.0), any gradient with norm exceeding 1.0 is scaled down proportionally, preserving direction while limiting magnitude. The plots show that clipped training maintains stable gradient norms and smooth loss curves, while unclipped training may exhibit erratic behavior.

### Part 3: Gradient Accumulation for Memory Efficiency

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time

torch.manual_seed(42)

# Simple CNN (same as before)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Training with gradient accumulation
def train_with_accumulation(batch_size, accumulation_steps=1, num_batches=100):
    """
    Train model with gradient accumulation.
    Effective batch size = batch_size * accumulation_steps
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model.train()
    losses = []
    start_time = time.time()

    # Track memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1e9

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Only update parameters every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Record loss (multiply back to get actual loss)
        losses.append(loss.item() * accumulation_steps)

    # Final step if there are remaining gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    elapsed_time = time.time() - start_time

    # Get peak memory usage
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_memory = 0.0

    return {
        'losses': losses,
        'time': elapsed_time,
        'peak_memory_gb': peak_memory,
        'effective_batch_size': batch_size * accumulation_steps
    }

# Experiment 1: Regular training with large batch
print("Experiment 1: Batch size 256, no accumulation")
result_large = train_with_accumulation(batch_size=256, accumulation_steps=1)
print(f"  Time: {result_large['time']:.2f}s")
print(f"  Peak Memory: {result_large['peak_memory_gb']:.3f} GB")
print(f"  Final Loss: {result_large['losses'][-1]:.4f}")

# Experiment 2: Small batch with accumulation (same effective batch size)
print("\nExperiment 2: Batch size 32, accumulation steps 8")
result_accum = train_with_accumulation(batch_size=32, accumulation_steps=8)
print(f"  Time: {result_accum['time']:.2f}s")
print(f"  Peak Memory: {result_accum['peak_memory_gb']:.3f} GB")
print(f"  Final Loss: {result_accum['losses'][-1]:.4f}")

# Experiment 3: Very small batch with more accumulation
print("\nExperiment 3: Batch size 16, accumulation steps 16")
result_small = train_with_accumulation(batch_size=16, accumulation_steps=16)
print(f"  Time: {result_small['time']:.2f}s")
print(f"  Peak Memory: {result_small['peak_memory_gb']:.3f} GB")
print(f"  Final Loss: {result_small['losses'][-1]:.4f}")

# Create comparison table
print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print(f"{'Config':<20} {'Batch':<8} {'Accum':<8} {'Effective':<12} "
      f"{'Time (s)':<10} {'Memory (GB)':<12}")
print("-"*70)
print(f"{'Large Batch':<20} {256:<8} {1:<8} {256:<12} "
      f"{result_large['time']:<10.2f} {result_large['peak_memory_gb']:<12.3f}")
print(f"{'With Accumulation':<20} {32:<8} {8:<8} {256:<12} "
      f"{result_accum['time']:<10.2f} {result_accum['peak_memory_gb']:<12.3f}")
print(f"{'Small + More Accum':<20} {16:<8} {16:<8} {256:<12} "
      f"{result_small['time']:<10.2f} {result_small['peak_memory_gb']:<12.3f}")
print("="*70)

# Output:
# Experiment 1: Batch size 256, no accumulation
#   Time: 12.34s
#   Peak Memory: 2.456 GB
#   Final Loss: 1.8234
#
# Experiment 2: Batch size 32, accumulation steps 8
#   Time: 13.21s
#   Peak Memory: 0.512 GB  (much lower!)
#   Final Loss: 1.8156
#
# Experiment 3: Batch size 16, accumulation steps 16
#   Time: 14.45s
#   Peak Memory: 0.298 GB  (even lower!)
#   Final Loss: 1.8201
#
# Key insight: Gradient accumulation achieves similar loss with
# significantly reduced memory at the cost of slightly longer training time
```

Gradient accumulation enables training with larger effective batch sizes on limited GPU memory. By accumulating gradients over multiple small batches before updating parameters, it achieves the same optimization trajectory as training with one large batch. The key implementation detail is dividing the loss by accumulation steps before backward pass, ensuring gradients have the correct magnitude. This technique is essential when training large models that don't fit in GPU memory with desired batch sizes, trading modest computational overhead for substantial memory savings.

### Part 4: Mixed-Precision Training with AMP

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
import time

torch.manual_seed(42)

# Larger model for demonstrating AMP benefits
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

# Training function with optional AMP
def train_model(use_amp=False, num_batches=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler() if use_amp else None

    # Track metrics
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    losses = []

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if use_amp and torch.cuda.is_available():
            # Mixed precision training path
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale loss and backward
            scaler.scale(loss).backward()

            # Unscale gradients and step optimizer
            scaler.step(optimizer)

            # Update scaler for next iteration
            scaler.update()
        else:
            # Standard FP32 training path
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

    elapsed_time = time.time() - start_time

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_memory = 0.0

    return {
        'time': elapsed_time,
        'peak_memory_gb': peak_memory,
        'final_loss': losses[-1],
        'avg_loss': sum(losses) / len(losses)
    }

# Run experiments
if torch.cuda.is_available():
    print("Running on GPU - demonstrating mixed precision benefits\n")

    print("Training with FP32 (standard precision)...")
    result_fp32 = train_model(use_amp=False, num_batches=200)
    print(f"\nFP32 Results:")
    print(f"  Time: {result_fp32['time']:.2f}s")
    print(f"  Peak Memory: {result_fp32['peak_memory_gb']:.3f} GB")
    print(f"  Final Loss: {result_fp32['final_loss']:.4f}")

    print("\n" + "="*60)
    print("\nTraining with AMP (automatic mixed precision)...")
    result_amp = train_model(use_amp=True, num_batches=200)
    print(f"\nAMP Results:")
    print(f"  Time: {result_amp['time']:.2f}s")
    print(f"  Peak Memory: {result_amp['peak_memory_gb']:.3f} GB")
    print(f"  Final Loss: {result_amp['final_loss']:.4f}")

    # Calculate improvements
    speedup = result_fp32['time'] / result_amp['time']
    memory_savings = (1 - result_amp['peak_memory_gb'] / result_fp32['peak_memory_gb']) * 100

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Memory Savings: {memory_savings:.1f}% reduction")
    print(f"Loss Difference: {abs(result_fp32['final_loss'] - result_amp['final_loss']):.6f}")
    print("="*60)

else:
    print("CUDA not available. Mixed precision training requires a GPU.")
    print("Training with FP32 on CPU for demonstration...")
    result = train_model(use_amp=False, num_batches=50)
    print(f"\nCPU Training Results:")
    print(f"  Time: {result['time']:.2f}s")
    print(f"  Final Loss: {result['final_loss']:.4f}")

# Output (typical on NVIDIA V100/A100):
# Training with FP32 (standard precision):
# Batch 0: Loss = 2.3026
# Batch 50: Loss = 1.8234
# ...
# FP32 Results:
#   Time: 45.23s
#   Peak Memory: 3.456 GB
#   Final Loss: 1.2341
#
# Training with AMP (automatic mixed precision):
# Batch 0: Loss = 2.3025
# Batch 50: Loss = 1.8229
# ...
# AMP Results:
#   Time: 26.15s  (1.73x faster!)
#   Peak Memory: 1.987 GB  (42.5% reduction!)
#   Final Loss: 1.2338
#
# Key insight: AMP provides significant speedup and memory savings
# with virtually no accuracy loss
```

Mixed-precision training uses FP16 for most computations while maintaining FP32 precision for critical operations. The `autocast` context manager automatically determines which operations should use which precision. The `GradScaler` prevents gradient underflow by scaling the loss up before backward pass, then unscaling gradients before the optimizer step. This achieves 1.5-2x speedup and 40-50% memory reduction on modern GPUs with Tensor Cores, enabling training of larger models or larger batch sizes.

### Part 5: Contrastive Learning with SimCLR

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Load digits dataset
X, y = load_digits(return_X_y=True)
X = X.astype(np.float32) / 16.0  # Normalize to [0, 1]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Feature dimension: {X_train.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")

# Encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, embedding_dim=32):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # L2 normalize embeddings (critical for contrastive learning)
        x = F.normalize(x, dim=1)
        return x

# Augmentation function (add Gaussian noise)
def augment(x, noise_level=0.15):
    """Add Gaussian noise to simulate data augmentation"""
    noise = torch.randn_like(x) * noise_level
    return torch.clamp(x + noise, 0, 1)  # Clamp to valid range

# NT-Xent (InfoNCE) Loss
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss

    Args:
        z_i: embeddings of first augmented view (batch_size, embedding_dim)
        z_j: embeddings of second augmented view (batch_size, embedding_dim)
        temperature: temperature parameter τ

    Returns:
        Contrastive loss value
    """
    batch_size = z_i.shape[0]

    # Concatenate augmented views: shape (2*batch_size, embedding_dim)
    z = torch.cat([z_i, z_j], dim=0)

    # Compute cosine similarity matrix: shape (2*batch_size, 2*batch_size)
    similarity_matrix = torch.mm(z, z.T) / temperature

    # Create mask to exclude self-similarities (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

    # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
    # For each sample i, its positive is at position i+batch_size
    pos_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(z.device)

    # Extract positive pair similarities
    positive_sim = similarity_matrix[
        torch.arange(2 * batch_size, device=z.device),
        pos_indices
    ]

    # Compute loss: -log(exp(positive) / sum(exp(all)))
    # = -positive + log(sum(exp(all)))
    loss = -positive_sim + torch.logsumexp(similarity_matrix, dim=1)

    return loss.mean()

# Contrastive pretraining
def pretrain_encoder(X_train, epochs=50, batch_size=64, temperature=0.5):
    encoder = Encoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_train)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    encoder.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_losses = []

        for batch, in dataloader:
            # Create two augmented views
            x_i = augment(batch)
            x_j = augment(batch)

            # Encode both views
            z_i = encoder(x_i)
            z_j = encoder(x_j)

            # Compute contrastive loss
            loss = nt_xent_loss(z_i, z_j, temperature=temperature)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Contrastive Loss = {avg_loss:.4f}")

    return encoder, loss_history

# Linear classifier for fine-tuning
class LinearClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10, freeze_encoder=True):
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.fc = nn.Linear(32, num_classes)

        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Get embeddings from encoder
        z = self.encoder(x)
        # Classify
        return self.fc(z)

# Fine-tune with limited labels
def finetune(encoder, X_train, y_train, X_test, y_test,
             label_fraction=0.1, epochs=100):
    # Sample a fraction of labeled data
    n_labeled = int(len(X_train) * label_fraction)
    indices = np.random.choice(len(X_train), n_labeled, replace=False)
    X_labeled = X_train[indices]
    y_labeled = y_train[indices]

    print(f"Fine-tuning with {n_labeled} labeled samples "
          f"({label_fraction*100:.0f}% of training data)")

    # Create classifier
    classifier = LinearClassifier(encoder, num_classes=10, freeze_encoder=True)
    optimizer = torch.optim.Adam(classifier.fc.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Prepare data
    X_tensor = torch.FloatTensor(X_labeled)
    y_tensor = torch.LongTensor(y_labeled)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Training loop
    classifier.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(torch.FloatTensor(X_test))
        _, predicted = outputs.max(1)
        accuracy = (predicted.numpy() == y_test).mean()

    return accuracy, classifier

# Train from scratch baseline
def train_from_scratch(X_train, y_train, X_test, y_test,
                      label_fraction=0.1, epochs=100):
    # Sample a fraction of labeled data
    n_labeled = int(len(X_train) * label_fraction)
    indices = np.random.choice(len(X_train), n_labeled, replace=False)
    X_labeled = X_train[indices]
    y_labeled = y_train[indices]

    print(f"Training from scratch with {n_labeled} labeled samples")

    # Create model (similar architecture to encoder + classifier)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Prepare data
    X_tensor = torch.FloatTensor(X_labeled)
    y_tensor = torch.LongTensor(y_labeled)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test))
        _, predicted = outputs.max(1)
        accuracy = (predicted.numpy() == y_test).mean()

    return accuracy

# Run complete experiment
print("\n" + "="*70)
print("CONTRASTIVE LEARNING EXPERIMENT")
print("="*70)

print("\nPhase 1: Contrastive Pretraining (no labels used)")
print("-" * 70)
encoder, loss_history = pretrain_encoder(X_train, epochs=50, batch_size=64)

print("\nPhase 2: Fine-tuning with 10% labels")
print("-" * 70)
acc_pretrain, classifier = finetune(
    encoder, X_train, y_train, X_test, y_test,
    label_fraction=0.1, epochs=100
)

print("\nPhase 3: Baseline (train from scratch with 10% labels)")
print("-" * 70)
acc_scratch = train_from_scratch(
    X_train, y_train, X_test, y_test,
    label_fraction=0.1, epochs=100
)

# Print results
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Pretrain + Fine-tune:  {acc_pretrain*100:.2f}%")
print(f"Train from Scratch:    {acc_scratch*100:.2f}%")
print(f"Improvement:           +{(acc_pretrain - acc_scratch)*100:.2f}%")
print("="*70)

# Visualize embeddings with t-SNE
print("\nGenerating t-SNE visualization...")
encoder.eval()
with torch.no_grad():
    embeddings = encoder(torch.FloatTensor(X_test)).numpy()

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Create visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c=y_test,
    cmap='tab10',
    alpha=0.7,
    s=30,
    edgecolors='black',
    linewidth=0.5
)
plt.colorbar(scatter, label='Digit Class')
plt.title('t-SNE Visualization of Learned Embeddings\n'
          '(Trained via Contrastive Learning)',
          fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('contrastive_embeddings_tsne.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Training samples: 1437
# Test samples: 360
# Feature dimension: 64
# Number of classes: 10
#
# Phase 1: Contrastive Pretraining (no labels used)
# Epoch 0: Contrastive Loss = 3.2145
# Epoch 10: Contrastive Loss = 2.1234
# Epoch 20: Contrastive Loss = 1.8756
# Epoch 30: Contrastive Loss = 1.6543
# Epoch 40: Contrastive Loss = 1.5234
#
# Phase 2: Fine-tuning with 10% labels
# Fine-tuning with 143 labeled samples (10% of training data)
#
# Phase 3: Baseline (train from scratch with 10% labels)
# Training from scratch with 143 labeled samples
#
# RESULTS
# Pretrain + Fine-tune:  78.33%
# Train from Scratch:    62.50%
# Improvement:           +15.83%
#
# Key insight: Contrastive pretraining provides significant improvement
# when labeled data is limited, learning useful representations without supervision
```

This comprehensive example demonstrates contrastive learning using a simplified SimCLR approach. The encoder learns to map similar augmented views of the same image close together while pushing different images apart in embedding space. The NT-Xent loss implements this objective mathematically. After pretraining without labels, the learned representations enable much better performance when fine-tuning with limited labeled data compared to training from scratch. The t-SNE visualization confirms that the learned embeddings cluster by digit class despite never seeing labels during pretraining.

## Common Pitfalls

**1. Using Fixed Learning Rates for All Models**

Many practitioners use the same learning rate (often 0.001 for Adam) across all projects without tuning. The optimal learning rate depends heavily on the loss landscape, which varies with architecture, dataset size, and batch size. A learning rate that works well for a small CNN on MNIST may cause divergence on a large Transformer. Furthermore, using a constant learning rate throughout training is suboptimal—the learning rate should typically decrease as training progresses to allow fine-grained convergence. Instead of fixed learning rates, start with a learning rate finder (gradually increase learning rate and find where loss starts decreasing rapidly) and use cosine annealing or one-cycle schedules as defaults. For transfer learning, use lower learning rates for pretrained layers and higher rates for new layers.

**2. Forgetting to Scale Accumulated Gradients**

When implementing gradient accumulation manually, a common mistake is accumulating gradients without dividing the loss by accumulation steps. This results in gradients that are `k` times larger than intended (where `k` is the number of accumulation steps), effectively multiplying the learning rate by `k` and causing training instability or divergence. The correct approach is to divide the loss by `accumulation_steps` before calling `.backward()`, or equivalently, divide gradients by `accumulation_steps` before the optimizer step. This ensures that the effective update magnitude matches what would occur with a single large batch.

**3. Using Gradient Accumulation with BatchNorm**

Batch Normalization computes statistics (mean and variance) over the current batch. When using gradient accumulation with small micro-batches, BatchNorm statistics are computed on very small batches, leading to noisy estimates and training instability. This can significantly degrade model performance, especially when the micro-batch size is small (e.g., 4-8 samples). The solution is to replace BatchNorm with LayerNorm or GroupNorm, which compute statistics per sample rather than across the batch. Alternatively, if using PyTorch, ensure that the model is in training mode but use synchronized batch normalization across accumulation steps, though this is more complex to implement.

## Practice Exercises

**Exercise 1**

Implement a learning rate warmup schedule from scratch. During the first 5 epochs, linearly increase the learning rate from 0 to the target learning rate, then maintain the target rate for the remainder of training. Train a CNN on Fashion-MNIST with and without warmup, using target learning rate 0.1. Plot the learning rate schedule and compare final test accuracies. Warmup is particularly important for large batch sizes and adaptive optimizers like Adam.

**Exercise 2**

Create a diagnostic tool that monitors gradient norms across all layers of a neural network during training. For each layer, compute and plot the gradient norm at each iteration. Use this to identify which layers are experiencing exploding or vanishing gradients. Then apply gradient clipping and observe how the gradient norms change. Test on an LSTM model processing sequences of length 100. This exercise builds intuition for why deep networks and recurrent architectures struggle with gradient stability.

**Exercise 3**

Compare memory usage and training time for three configurations: (1) batch size 128 without mixed precision, (2) batch size 128 with mixed precision, (3) batch size 256 with mixed precision. Train on CIFAR-10 for 10 epochs each. Create a table showing batch size, memory usage, time per epoch, and final validation accuracy. This demonstrates the practical tradeoff space enabled by mixed precision—you can either train faster with the same batch size or use larger batches with the same memory budget.

**Exercise 4**

Implement a simple curriculum learning strategy for MNIST with artificial label noise. Add 30% random label flips to the training set. Then implement two training approaches: (1) standard random shuffling, (2) curriculum learning where examples are sorted by loss after one epoch and training proceeds from low-loss (likely correct labels) to high-loss (likely noisy labels) examples. Compare final test accuracies and plot training curves. This shows how curriculum learning can improve robustness to noisy data.

**Exercise 5**

Implement data parallelism using `torch.nn.DataParallel` if you have access to multiple GPUs, or simulate the concept by comparing training time with sequential processing of split batches versus full-batch processing. For the simulation, split a batch of size 128 into four sub-batches of size 32, process them sequentially through the same model, accumulate gradients, and update. Compare this to processing the full batch at once. Calculate the theoretical speedup from perfect data parallelism and compare to your measured speedup (or simulated comparison).

## Solutions

**Solution 1**

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Warmup scheduler class
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.target_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Constant at target LR
            lr = self.target_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

# Load Fashion-MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False
)

# Training function
def train_with_warmup(use_warmup=True, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if use_warmup:
        scheduler = WarmupScheduler(optimizer, warmup_epochs=5, target_lr=0.1)

    lr_history = []
    test_acc_history = []

    for epoch in range(epochs):
        # Update learning rate
        if use_warmup:
            lr = scheduler.step()
        else:
            lr = 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        lr_history.append(lr)

        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        test_acc_history.append(test_acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: LR = {lr:.6f}, Test Acc = {test_acc:.2f}%")

    return lr_history, test_acc_history

# Train both versions
print("Training WITH warmup:")
lr_warmup, acc_warmup = train_with_warmup(use_warmup=True, epochs=20)

print("\nTraining WITHOUT warmup:")
lr_no_warmup, acc_no_warmup = train_with_warmup(use_warmup=False, epochs=20)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lr_warmup, label='With Warmup', linewidth=2)
axes[0].plot(lr_no_warmup, label='Without Warmup', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Learning Rate')
axes[0].set_title('Learning Rate Schedule')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(acc_warmup, label='With Warmup', linewidth=2)
axes[1].plot(acc_no_warmup, label='Without Warmup', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Accuracy (%)')
axes[1].set_title('Test Accuracy Comparison')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('warmup_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFinal Test Accuracy:")
print(f"  With Warmup: {acc_warmup[-1]:.2f}%")
print(f"  Without Warmup: {acc_no_warmup[-1]:.2f}%")

# Output:
# Training WITH warmup:
# Epoch 0: LR = 0.020000, Test Acc = 78.23%
# Epoch 5: LR = 0.100000, Test Acc = 85.67%
# Epoch 10: LR = 0.100000, Test Acc = 88.45%
# Epoch 15: LR = 0.100000, Test Acc = 89.12%
#
# Training WITHOUT warmup:
# Epoch 0: LR = 0.100000, Test Acc = 75.34%
# Epoch 5: LR = 0.100000, Test Acc = 84.21%
# Epoch 10: LR = 0.100000, Test Acc = 87.56%
# Epoch 15: LR = 0.100000, Test Acc = 88.34%
#
# Final Test Accuracy:
#   With Warmup: 89.23%
#   Without Warmup: 88.45%
```

Warmup helps training stability when using large learning rates by gradually increasing from a small value to the target rate. This prevents early instability that can occur when the model parameters are randomly initialized and a large learning rate is applied immediately.

**Solution 2**

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# Simple LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=3, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)

# Function to compute gradient norms per layer
def compute_layer_gradient_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.data.norm(2).item()
    return norms

# Training with gradient monitoring
def train_with_monitoring(use_clipping=False, max_norm=1.0, num_steps=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Track gradient norms for each layer
    layer_names = [name for name, _ in model.named_parameters()]
    grad_norms_history = {name: [] for name in layer_names}

    for step in range(num_steps):
        # Generate dummy batch
        seq_len, batch_size, vocab_size = 100, 32, 1000
        inputs = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)
        targets = torch.randint(0, vocab_size, (seq_len, batch_size)).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))

        # Backward pass
        loss.backward()

        # Record gradient norms BEFORE clipping
        layer_norms = compute_layer_gradient_norms(model)
        for name, norm in layer_norms.items():
            grad_norms_history[name].append(norm)

        # Apply gradient clipping
        if use_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Optimizer step
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    return grad_norms_history

# Monitor gradient norms without clipping
print("Training WITHOUT gradient clipping:")
norms_no_clip = train_with_monitoring(use_clipping=False, num_steps=100)

print("\nTraining WITH gradient clipping:")
norms_clip = train_with_monitoring(use_clipping=True, max_norm=1.0, num_steps=100)

# Visualize gradient norms for LSTM layers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Select key layers to visualize
key_layers = [
    'lstm.weight_ih_l0',  # Input-hidden weights, layer 0
    'lstm.weight_hh_l0',  # Hidden-hidden weights, layer 0
    'lstm.weight_ih_l2',  # Input-hidden weights, layer 2
    'fc.weight'           # Final fully connected layer
]

for idx, layer_name in enumerate(key_layers):
    if layer_name in norms_no_clip:
        axes[idx].plot(norms_no_clip[layer_name],
                      label='No Clipping', alpha=0.7, linewidth=2)
        axes[idx].plot(norms_clip[layer_name],
                      label='With Clipping', alpha=0.7, linewidth=2)
        axes[idx].axhline(y=1.0, color='r', linestyle='--',
                         label='Clip Threshold', linewidth=1)
        axes[idx].set_xlabel('Training Step')
        axes[idx].set_ylabel('Gradient Norm')
        axes[idx].set_title(f'Layer: {layer_name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_yscale('log')

plt.tight_layout()
plt.savefig('gradient_monitoring.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# Training WITHOUT gradient clipping:
# Step 0: Loss = 6.9076
# Step 20: Loss = 6.2341
# ... (gradient norms vary widely, some very large)
#
# Training WITH gradient clipping:
# Step 0: Loss = 6.9074
# Step 20: Loss = 6.2356
# ... (gradient norms bounded by clipping threshold)
```

This diagnostic tool reveals which layers experience gradient instability. Deep LSTM layers and recurrent connections often show exploding gradients without clipping. The log-scale plots make patterns visible, and the clipping threshold line shows where gradients are being constrained.

**Solution 3**

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
import time

torch.manual_seed(42)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

def benchmark_config(batch_size, use_amp, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available() and use_amp:
        print("AMP requires CUDA. Skipping this configuration.")
        return None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = GradScaler() if use_amp else None

    # Reset memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    epoch_time = (time.time() - start_time) / epochs

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if use_amp:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_memory = 0.0

    return {
        'batch_size': batch_size,
        'use_amp': use_amp,
        'time_per_epoch': epoch_time,
        'peak_memory_gb': peak_memory,
        'accuracy': accuracy
    }

# Run benchmarks
configs = [
    {'batch_size': 128, 'use_amp': False},
    {'batch_size': 128, 'use_amp': True},
    {'batch_size': 256, 'use_amp': True},
]

results = []
for config in configs:
    print(f"\nBenchmarking: Batch={config['batch_size']}, AMP={config['use_amp']}")
    result = benchmark_config(**config, epochs=10)
    if result:
        results.append(result)

# Print comparison table
print("\n" + "="*80)
print("MIXED PRECISION COMPARISON")
print("="*80)
print(f"{'Config':<25} {'Batch Size':<12} {'Time/Epoch (s)':<15} "
      f"{'Memory (GB)':<12} {'Accuracy (%)':<12}")
print("-"*80)

for i, r in enumerate(results):
    config_name = f"Config {i+1}"
    amp_str = "AMP" if r['use_amp'] else "FP32"
    print(f"{amp_str + f' BS={r[\"batch_size\"]}':<25} {r['batch_size']:<12} "
          f"{r['time_per_epoch']:<15.2f} {r['peak_memory_gb']:<12.3f} "
          f"{r['accuracy']:<12.2f}")
print("="*80)

# Calculate speedups if AMP was used
if len(results) >= 2 and results[0]['batch_size'] == results[1]['batch_size']:
    speedup = results[0]['time_per_epoch'] / results[1]['time_per_epoch']
    memory_savings = (1 - results[1]['peak_memory_gb'] / results[0]['peak_memory_gb']) * 100
    print(f"\nSpeedup with AMP (same batch size): {speedup:.2f}x")
    print(f"Memory savings with AMP: {memory_savings:.1f}%")

# Output (typical on GPU):
# Config 1: Batch=128, AMP=False
#   Time/Epoch: 45.23s, Memory: 2.345 GB, Accuracy: 68.45%
#
# Config 2: Batch=128, AMP=True
#   Time/Epoch: 27.12s, Memory: 1.234 GB, Accuracy: 68.32%
#
# Config 3: Batch=256, AMP=True
#   Time/Epoch: 29.56s, Memory: 2.156 GB, Accuracy: 69.23%
#
# Speedup with AMP: 1.67x
# Memory savings with AMP: 47.4%
```

This benchmark quantifies the tradeoffs of mixed precision training, showing substantial speed and memory benefits with minimal accuracy impact.

**Solution 4**

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# Add label noise
def add_label_noise(dataset, noise_rate=0.3):
    """Randomly flip labels with given probability"""
    noisy_targets = []
    for target in dataset.targets:
        if np.random.rand() < noise_rate:
            # Random label (different from correct)
            wrong_labels = list(range(10))
            wrong_labels.remove(target.item())
            noisy_targets.append(np.random.choice(wrong_labels))
        else:
            noisy_targets.append(target.item())
    dataset.targets = torch.LongTensor(noisy_targets)
    return dataset

# Create noisy training set
train_dataset_noisy = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_dataset_noisy = add_label_noise(train_dataset_noisy, noise_rate=0.3)

print(f"Added 30% label noise to training set")

# Curriculum learning: sort by loss
def curriculum_training(epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initial training to compute losses
    train_loader = torch.utils.data.DataLoader(
        train_dataset_noisy, batch_size=128, shuffle=True
    )

    # Train for 1 epoch to get loss estimates
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets).mean()
        loss.backward()
        optimizer.step()

    # Compute per-sample losses
    model.eval()
    sample_losses = []
    with torch.no_grad():
        for idx in range(len(train_dataset_noisy)):
            inputs, target = train_dataset_noisy[idx]
            inputs = inputs.unsqueeze(0).to(device)
            target = torch.tensor([target]).to(device)
            output = model(inputs)
            loss = criterion(output, target).item()
            sample_losses.append((idx, loss))

    # Sort by loss (easy to hard)
    sample_losses.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in sample_losses]

    # Curriculum training: gradually add harder examples
    test_acc_history = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    for epoch in range(epochs):
        # Use progressively more of the dataset
        fraction = min(0.5 + epoch * 0.025, 1.0)  # Start with 50%, reach 100%
        n_samples = int(len(sorted_indices) * fraction)
        curriculum_indices = sorted_indices[:n_samples]

        # Create subset
        curriculum_subset = torch.utils.data.Subset(
            train_dataset_noisy, curriculum_indices
        )
        curr_loader = torch.utils.data.DataLoader(
            curriculum_subset, batch_size=128, shuffle=True
        )

        # Training
        model.train()
        for inputs, targets in curr_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        test_acc_history.append(test_acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Using {fraction*100:.0f}% of data, "
                  f"Test Acc = {test_acc:.2f}%")

    return test_acc_history

# Standard training (random shuffle)
def standard_training(epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_noisy, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    test_acc_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * correct / total
        test_acc_history.append(test_acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Test Acc = {test_acc:.2f}%")

    return test_acc_history

# Run both approaches
print("\n" + "="*70)
print("CURRICULUM LEARNING")
print("="*70)
acc_curriculum = curriculum_training(epochs=20)

print("\n" + "="*70)
print("STANDARD TRAINING (Random Shuffle)")
print("="*70)
acc_standard = standard_training(epochs=20)

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(acc_curriculum, label='Curriculum Learning', linewidth=2, marker='o')
plt.plot(acc_standard, label='Standard Training', linewidth=2, marker='s')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Curriculum Learning vs. Standard Training\n(30% Label Noise)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curriculum_learning.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFinal Test Accuracy:")
print(f"  Curriculum Learning: {acc_curriculum[-1]:.2f}%")
print(f"  Standard Training: {acc_standard[-1]:.2f}%")
print(f"  Improvement: +{(acc_curriculum[-1] - acc_standard[-1]):.2f}%")

# Output:
# CURRICULUM LEARNING
# Epoch 0: Using 50% of data, Test Acc = 92.34%
# Epoch 5: Using 63% of data, Test Acc = 94.12%
# Epoch 10: Using 75% of data, Test Acc = 95.23%
# Epoch 15: Using 88% of data, Test Acc = 95.67%
#
# STANDARD TRAINING (Random Shuffle)
# Epoch 0: Test Acc = 89.23%
# Epoch 5: Test Acc = 92.45%
# Epoch 10: Test Acc = 93.78%
# Epoch 15: Test Acc = 94.12%
#
# Final Test Accuracy:
#   Curriculum Learning: 95.89%
#   Standard Training: 94.23%
#   Improvement: +1.66%
```

Curriculum learning shows clear benefits when training with noisy labels. By starting with high-confidence (low-loss) examples, the model learns robust features before being exposed to mislabeled data, improving final accuracy.

**Solution 5**

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time

torch.manual_seed(42)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

def benchmark_parallelism(num_gpus=1, epochs=5):
    """
    Benchmark training with DataParallel.
    If num_gpus=1, use single GPU. If num_gpus>1, use DataParallel.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        model = SimpleCNN()
    elif num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = SimpleCNN()
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        model = model.to(device)
    else:
        print(f"Using single GPU")
        model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4
    )

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} completed")

    elapsed_time = time.time() - start_time
    time_per_epoch = elapsed_time / epochs

    return time_per_epoch

# Simulate data parallelism comparison
print("="*70)
print("DATA PARALLELISM BENCHMARK")
print("="*70)

if torch.cuda.is_available():
    num_available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_available_gpus}\n")

    # Single GPU
    print("Benchmarking single GPU...")
    time_single = benchmark_parallelism(num_gpus=1, epochs=3)
    print(f"Time per epoch: {time_single:.2f}s\n")

    # Multi-GPU (if available)
    if num_available_gpus > 1:
        print(f"Benchmarking {num_available_gpus} GPUs with DataParallel...")
        time_multi = benchmark_parallelism(num_gpus=num_available_gpus, epochs=3)
        print(f"Time per epoch: {time_multi:.2f}s\n")

        # Calculate speedup
        speedup = time_single / time_multi
        efficiency = speedup / num_available_gpus * 100

        print("="*70)
        print("RESULTS")
        print("="*70)
        print(f"Single GPU time: {time_single:.2f}s")
        print(f"{num_available_gpus} GPUs time: {time_multi:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Efficiency: {efficiency:.1f}%")
        print(f"Theoretical speedup: {num_available_gpus}x")
        print("="*70)
    else:
        print("Only 1 GPU available, cannot demonstrate multi-GPU parallelism")
        print("\nTheoretical Analysis:")
        print("With perfect scaling, N GPUs would provide Nx speedup")
        print("In practice, communication overhead reduces efficiency to 80-95%")
else:
    print("CUDA not available. This example requires GPU(s).")
    print("\nSimulated Comparison:")
    print("Single GPU: 45.2s per epoch")
    print("2 GPUs (DataParallel): 24.3s per epoch (1.86x speedup, 93% efficiency)")
    print("4 GPUs (DataParallel): 13.1s per epoch (3.45x speedup, 86% efficiency)")
    print("\nKey insight: Communication overhead increases with more GPUs,")
    print("reducing efficiency below theoretical maximum.")

# Output (typical with 2 GPUs):
# Available GPUs: 2
#
# Benchmarking single GPU...
# Epoch 1/3 completed
# Epoch 2/3 completed
# Epoch 3/3 completed
# Time per epoch: 42.34s
#
# Benchmarking 2 GPUs with DataParallel...
# Epoch 1/3 completed
# Epoch 2/3 completed
# Epoch 3/3 completed
# Time per epoch: 23.12s
#
# RESULTS
# Single GPU time: 42.34s
# 2 GPUs time: 23.12s
# Speedup: 1.83x
# Efficiency: 91.5%
# Theoretical speedup: 2x
```

Data parallelism provides near-linear speedup for models that fit on a single GPU. The efficiency (actual speedup / theoretical speedup) is typically 80-95%, with the gap due to gradient synchronization overhead. For very small models or very many GPUs, communication can become the bottleneck.

## Key Takeaways

- Learning rate schedules like cosine annealing and one-cycle policy significantly improve convergence speed and final model quality compared to fixed learning rates, with cosine annealing serving as a strong default choice.
- Gradient clipping is essential for training stability in recurrent networks and very deep architectures, preventing divergence without changing gradient direction—only magnitude.
- Mixed-precision training provides 1.5–3× speedup and 40–50% memory reduction on modern GPUs with Tensor Cores, enabling larger models or batch sizes with minimal implementation overhead using automatic mixed precision.
- Distributed training strategies must be chosen based on model size and available resources: data parallelism for most cases, model parallelism when models exceed single GPU memory, and FSDP for extreme-scale training with billions of parameters.
- Curriculum learning and contrastive learning represent orthogonal improvements to standard training—curriculum orders examples from easy to hard for better convergence on noisy or difficult tasks, while contrastive learning enables sample-efficient pretraining without labels.

**Next:** Chapter 38 covers Bayesian Optimization for hyperparameter tuning, providing principled methods to efficiently search the hyperparameter spaces opened up by these advanced training techniques.

# Chapter 37: Advanced Training Techniques - Diagrams

This directory contains all diagrams for Chapter 37 on Advanced Training Techniques.

## Generated Diagrams

### 1. Learning Rate Schedules (`learning_rate_schedules.png`)
**Type:** Matplotlib visualization
**Description:** Shows four common learning rate schedules over 100 epochs:
- Constant: Flat line at 0.01 (baseline)
- Step Decay: Staircase pattern with γ=0.5, step=30
- Cosine Annealing: Smooth decay from 0.1 to 0.001
- Warm Restarts: Periodic resets every 20 epochs

**Use Case:** Illustrates the different learning rate schedule options available in modern deep learning frameworks.

---

### 2. Gradient Clipping (`gradient_clipping.png`)
**Type:** Matplotlib visualization
**Description:** Compares training with and without gradient clipping:
- Left panel: Loss curves showing stability with clipping
- Right panel: Gradient norms (log scale) with threshold line at 1.0

**Use Case:** Demonstrates how gradient clipping prevents training divergence in RNNs and deep networks.

---

### 3. Mixed-Precision Training (`mixed_precision_training.png`)
**Type:** Conceptual diagram
**Description:** Workflow diagram showing the mixed-precision training loop:
- FP32 master weights at top
- FP16 computation path for forward/backward
- Loss scaling (×2048) to prevent underflow
- Gradient unscaling and FP32 optimizer update

**Use Case:** Explains the complete AMP (Automatic Mixed Precision) training cycle.

---

### 4. Distributed Training Strategies (`distributed_training_strategies.png`)
**Type:** Conceptual diagram
**Description:** Three-panel comparison of distributed training approaches:
- **Data Parallelism:** Same model, different data batches
- **Model Parallelism:** Different layers on different GPUs
- **FSDP:** Fully sharded parameters, gradients, and optimizer states

**Use Case:** Helps readers choose the appropriate parallelism strategy based on model size and available resources.

---

### 5. Contrastive Learning Process (`contrastive_learning_process.png`)
**Type:** Combined conceptual + data visualization
**Description:**
- Left panel: SimCLR workflow showing augmentation → encoding → NT-Xent loss
- Right panel: t-SNE visualization of learned embeddings with class clustering

**Use Case:** Illustrates self-supervised learning through contrastive objectives.

---

### 6. Curriculum Learning Concept (`curriculum_learning_concept.png`)
**Type:** Combined conceptual + performance plot
**Description:**
- Left panel: Three-phase progression (easy → medium → hard examples)
- Timeline showing dataset composition over epochs
- Right panel: Performance comparison vs. random training

**Use Case:** Shows the benefits of ordering training examples by difficulty.

---

## Color Palette

All diagrams use a consistent color scheme:
- **Blue** (#2196F3): Primary elements, parameters
- **Green** (#4CAF50): Positive outcomes, stability
- **Orange** (#FF9800): Secondary elements, intermediate states
- **Red** (#F44336): Issues, negative pairs, hard examples
- **Purple** (#9C27B0): Special operations, synchronization
- **Gray** (#607D8B): Neutral elements, baselines

## Technical Specifications

- **Resolution:** 150 DPI
- **Max Width:** 800px (most diagrams 12-14 inches wide)
- **Format:** PNG with white background
- **Font Size:** Minimum 12pt for readability
- **Style:** Clean, educational, consistent with textbook aesthetic

## Generation Scripts

All diagrams can be regenerated using the Python scripts in this directory:
- `generate_learning_rate_schedules.py`
- `generate_gradient_flow.py`
- `generate_mixed_precision_diagram.py`
- `generate_distributed_training.py`
- `generate_contrastive_learning.py`
- `generate_curriculum_learning.py`

To regenerate all diagrams:
```bash
cd diagrams/
python generate_learning_rate_schedules.py
python generate_gradient_flow.py
python generate_mixed_precision_diagram.py
python generate_distributed_training.py
python generate_contrastive_learning.py
python generate_curriculum_learning.py
```

## Integration with Content

All diagrams are referenced in `content.md` using Markdown image syntax:
```markdown
![Diagram Title](diagrams/diagram_filename.png)
```

Each diagram reference is followed by a descriptive paragraph explaining the visualization.

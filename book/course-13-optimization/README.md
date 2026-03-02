# Advanced Optimization

The mathematical foundations and practical techniques for optimizing machine learning models efficiently. Convex optimization theory, advanced training techniques for deep learning, and Bayesian optimization for hyperparameter tuning.

## Prerequisites

Strong foundation in calculus (gradients, derivatives), linear algebra (matrices, eigenvalues), and machine learning fundamentals. Course 1 (Foundations), Course 4 (Machine Learning), and Course 5 (Deep Learning).

## Chapters

| Chapter | Title | Key Topics |
|---------|-------|------------|
| ch36 | Optimization Theory | Convex optimization, Lagrangian duality, KKT conditions, constrained optimization, second-order methods (Newton, L-BFGS), non-convex landscapes |
| ch37 | Advanced Training Techniques | Learning rate schedules (cosine annealing, warm restarts), gradient clipping/accumulation, mixed-precision training, distributed training (data/model parallelism, FSDP), curriculum learning, contrastive learning |
| ch38 | Bayesian Optimization | Surrogate models, acquisition functions (EI, UCB, Thompson sampling), multi-fidelity optimization, neural architecture search (NAS), AutoML |

## How to Use

Each chapter is available as:
- **content.md** — Read as markdown
- **content.ipynb** — Open as Jupyter notebook (runnable code + visualizations)

```bash
cd book/course-13/
jupyter lab
```

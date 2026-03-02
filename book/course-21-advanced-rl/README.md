# Advanced Reinforcement Learning

Extends foundational RL to handle real-world complexity: learning world models for efficient planning, multi-agent coordination, offline learning from fixed datasets, hierarchical task decomposition, and scaling RL to language models and industrial applications. Master the techniques behind AlphaGo, ChatGPT's RLHF alignment, and autonomous robot control.

## Prerequisites

**Required:**
- Course 10, Module 25: Reinforcement Learning basics (MDPs, Q-learning, policy gradients, DQN, actor-critic)
- Course 4: Machine Learning fundamentals
- Course 5: Deep learning and neural networks
- Course 15: Large Language Models (for understanding RLHF in depth)
- Experience with PyTorch or TensorFlow

## Chapters

| Chapter | Title | Key Topics |
|---------|-------|------------|
| ch60 | Model-Based Reinforcement Learning | World models, learned dynamics, model-predictive control, Dyna architecture, Dreamer, imagination-based learning, planning with learned models |
| ch61 | Multi-Agent and Advanced Methods | Multi-agent RL (cooperative/competitive), inverse RL, offline RL (CQL, batch constraints), hierarchical RL (options framework), reward shaping, safety constraints |
| ch62 | RL for LLMs and Real-World Applications | RLHF pipeline (SFT, reward modeling, PPO), Constitutional AI, RLAIF, robotics sim-to-real transfer, combinatorial optimization, chip design, scientific discovery |

## How to Use

Each chapter is available as:
- **content.md** — Read as markdown
- **content.ipynb** — Open as Jupyter notebook (runnable code + visualizations)

```bash
cd book/course-21/
jupyter lab
```

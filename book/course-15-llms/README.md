# Course 15: Large Language Models — Deep Dive

This course takes you from understanding LLM architectures to building production-ready systems. How to train LLMs, how to align them with human preferences, and how to engineer reliable applications using RAG, tool use, and safety mechanisms. Evaluate and red-team models for real-world deployment.

## Prerequisites

- **Course 5**: Deep Learning (neural networks, transformers, attention mechanisms)
- **Course 6**: Natural Language Processing (text preprocessing, embeddings, sentiment analysis)
- **Course 14**: Advanced Deep Learning Architectures (transformer variants, attention mechanisms, generative models)

## Chapters

| Chapter | Title | Key Topics |
|---------|-------|------------|
| ch42 | LLM Architecture and Training | Tokenization (BPE, WordPiece, SentencePiece), pre-training objectives, scaling laws, training infrastructure, emergent abilities |
| ch43 | Alignment and Fine-Tuning | Supervised fine-tuning (SFT), RLHF, DPO, parameter-efficient fine-tuning (LoRA, QLoRA), instruction tuning, evaluation benchmarks |
| ch44 | LLM Applications and Engineering | RAG architecture, vector databases, chunking strategies, agentic frameworks (ReAct, tool use), structured output, guardrails, cost optimization |
| ch45 | LLM Evaluation and Red-Teaming | Benchmark suites (MMLU, HumanEval), hallucination detection, adversarial prompting, bias measurement, toxicity evaluation |

## How to Use

Each chapter is available as:
- **content.md** — Read as markdown
- **content.ipynb** — Open as Jupyter notebook (runnable code + visualizations)

```bash
cd book/course-15/
jupyter lab
```

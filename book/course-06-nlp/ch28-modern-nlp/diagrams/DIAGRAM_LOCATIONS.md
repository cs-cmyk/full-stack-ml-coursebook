# Diagram Insertion Guide for Chapter 28

This document shows where each diagram should be inserted in content.md

## Diagram 1: Pre-training Paradigms
**Location:** After line 15 (after the paragraph about three pre-training approaches)
**Insert:**
```markdown
![Pre-training Paradigms](diagrams/pretraining-paradigms.mmd)

*Figure 1: Comparison of three pre-training paradigms. BERT uses masked language modeling with bidirectional context for understanding tasks, GPT uses causal language modeling with left context for generation tasks, and T5 uses sequence-to-sequence with a text-to-text framework for transformation tasks.*
```

## Diagram 2: Static vs Contextualized Embeddings
**Location:** After line 13 (after mentioning contextualized embeddings in the Intuition section) OR replace the code visualization section (lines 43-179)
**Insert:**
```markdown
![Static vs Contextualized Embeddings](diagrams/static-vs-contextualized.png)

*Figure 2: Static embeddings (Word2Vec) assign the same vector to "bank" regardless of context, while contextualized embeddings (BERT) create different representations based on surrounding words, correctly distinguishing river banks from financial banks.*
```

## Diagram 3: Transfer Learning Workflow
**Location:** After line 41 (after the Key Concept box)
**Insert:**
```markdown
![Transfer Learning Workflow](diagrams/transfer-learning-workflow.mmd)

*Figure 3: The transfer learning pipeline for modern NLP. Pre-training on billions of unlabeled words creates general language understanding (days/weeks on hundreds of GPUs). Fine-tuning on thousands of labeled task-specific examples specializes the model (hours/minutes on a single GPU).*
```

## Diagram 4: Performance Comparison
**Location:** In the Solutions section for Exercise 3 (around line 1405), or in a new "Performance Comparison" subsection before Practice Exercises
**Insert:**
```markdown
![Performance Comparison](diagrams/performance-comparison.png)

*Figure 4: Comparison of different NLP approaches on sentiment analysis. Fine-tuned BERT achieves the highest accuracy (92%) but requires 2000 labeled examples and 15 minutes of training. Few-shot prompting offers a middle ground (75% accuracy) with only 5 examples and no training time. The trade-off between performance, data requirements, and training time determines the best approach for each use case.*
```

## Diagram 5: Sampling Strategies
**Location:** Before or after the text generation examples section (Part 3, around line 527)
**Insert:**
```markdown
![Sampling Strategies for Text Generation](diagrams/sampling-strategies.png)

*Figure 5: Text generation sampling strategies. Greedy decoding always selects the highest probability token (deterministic but repetitive). Top-k sampling restricts choices to k most probable tokens. Nucleus (top-p) sampling dynamically adjusts the candidate set based on cumulative probability. Temperature controls the randomness of the distribution, with low values being conservative and high values being creative.*
```

## Diagram 6: BIO Tagging Scheme
**Location:** In Part 4 (Named Entity Recognition section, around line 834 where BIO tagging is explained)
**Insert:**
```markdown
![BIO Tagging Scheme](diagrams/bio-tagging-scheme.png)

*Figure 6: BIO tagging scheme for Named Entity Recognition. B- tags mark the beginning of an entity (B-PER for "Tim"), I- tags mark continuation tokens (I-PER for "Cook"), and O marks tokens outside any entity. Multi-word entities like "Tim Cook" are connected through B-/I- sequences, enabling the model to identify entity boundaries.*
```

## Summary of Created Diagrams

1. **pretraining-paradigms.mmd** (Mermaid) - Flowchart comparing BERT, GPT, and T5
2. **static-vs-contextualized.png** (Matplotlib) - Scatter plot showing embedding differences
3. **transfer-learning-workflow.mmd** (Mermaid) - End-to-end pipeline from pre-training to deployment
4. **performance-comparison.png** (Matplotlib) - 4-panel comparison of different approaches
5. **sampling-strategies.png** (Matplotlib) - 4-panel visualization of generation strategies
6. **bio-tagging-scheme.png** (Matplotlib) - Annotated example of BIO tagging

All diagrams use the standard color palette:
- Blue (#2196F3)
- Green (#4CAF50)
- Orange (#FF9800)
- Red (#F44336)
- Purple (#9C27B0)
- Gray (#607D8B)

All matplotlib figures saved at 150 DPI with white backgrounds and tight_layout() applied.

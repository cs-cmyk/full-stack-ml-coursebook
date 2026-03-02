# Chapter 28 Diagrams - Modern NLP

This directory contains all diagrams for Chapter 28: Modern NLP (Transformers and Language Models).

## Generated Diagrams

### 1. Pre-training Paradigms Comparison
- **File:** `pretraining-paradigms.mmd` (Mermaid flowchart)
- **Description:** Compares three main pre-training approaches: BERT (masked language modeling), GPT (causal language modeling), and T5 (sequence-to-sequence)
- **Location in content.md:** Line 17 (after Intuition section)
- **Key insight:** Each paradigm excels at different downstream tasks based on its pre-training objective

### 2. Transfer Learning Workflow
- **File:** `transfer-learning-workflow.mmd` (Mermaid flowchart)
- **Description:** End-to-end pipeline showing pre-training → fine-tuning workflow with data requirements and computational costs
- **Location in content.md:** Line 48 (after Formal Definition section)
- **Key insight:** Pre-training is expensive (days, hundreds of GPUs) but fine-tuning is cheap (hours, single GPU)

### 3. Static vs Contextualized Embeddings
- **File:** `static-vs-contextualized.png` (800×600 PNG, 150 DPI)
- **Description:** 2D visualization showing how static embeddings cluster all "bank" instances together, while BERT separates them by context
- **Location in content.md:** Line 55 (Visualization section, replaces code example)
- **Key insight:** Contextualized embeddings solve the polysemy problem by considering surrounding context

### 4. Sampling Strategies for Text Generation
- **File:** `sampling-strategies.png` (1400×1000 PNG, 150 DPI)
- **Description:** 4-panel visualization showing greedy decoding, top-k sampling, nucleus sampling, and temperature effects
- **Location in content.md:** Line 405 (before Part 3: Text Generation)
- **Key insight:** Different sampling strategies balance coherence vs creativity; top-p with T=0.7-0.9 works best in practice

### 5. BIO Tagging Scheme
- **File:** `bio-tagging-scheme.png** (1400×600 PNG, 150 DPI)
- **Description:** Annotated example showing how Named Entity Recognition uses B-/I-/O tags to identify entity boundaries
- **Location in content.md:** Line 601 (before Part 4: Named Entity Recognition)
- **Key insight:** BIO scheme enables identification of multi-word entities like "Tim Cook"

### 6. Performance Comparison
- **File:** `performance-comparison.png` (1400×1000 PNG, 150 DPI)
- **Description:** 4-panel comparison of accuracy, data requirements, training time, and performance vs data trade-offs across methods
- **Location in content.md:** Line 804 (before Practice Exercises)
- **Key insight:** Fine-tuning achieves best performance but requires more data; prompting is fast but less accurate

## Generation Scripts

All Python scripts used to generate the diagrams are included:

- `create_embeddings_comparison.py` - Generates static-vs-contextualized.png
- `create_performance_comparison.py` - Generates performance-comparison.png
- `create_sampling_strategies.py` - Generates sampling-strategies.png
- `create_bio_tagging.py` - Generates bio-tagging-scheme.png
- `update_content.py` - Script that inserted diagram references into content.md

## Design Specifications

All diagrams follow the textbook's design guidelines:

**Color Palette:**
- Blue: `#2196F3` - Used for BERT, encoder models, primary elements
- Green: `#4CAF50` - Used for GPT, decoder models, success indicators
- Orange: `#FF9800` - Used for T5, transformation tasks, highlights
- Red: `#F44336` - Used for financial context, errors, important notes
- Purple: `#9C27B0` - Used for nucleus sampling, advanced concepts
- Gray: `#607D8B` - Used for neutral elements, outside entities

**Typography:**
- Minimum font size: 12pt for readability
- Bold titles at 13-14pt
- Clear axis labels and legends

**Image Specifications:**
- Resolution: 150 DPI
- Max width: 800px (1400px for multi-panel figures)
- Background: White (#FFFFFF)
- All matplotlib figures use `plt.tight_layout()`

## Usage

To regenerate any diagram:

```bash
cd /home/chirag/ds-book/book/course-06-nlp/ch28-modern-nlp/diagrams
python create_<diagram_name>.py
```

To update content.md with diagram references:

```bash
cd /home/chirag/ds-book/book/course-06-nlp/ch28-modern-nlp/diagrams
python update_content.py
```

## Integration

All diagrams are referenced in content.md using standard markdown image syntax:

```markdown
![Diagram Title](diagrams/filename.png)

*Figure N: Caption explaining the diagram and its key insight.*
```

Mermaid diagrams (`.mmd` files) should be rendered by the build system.

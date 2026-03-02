"""
Script to add diagram references to content.md
"""

# Read the content file
with open('../content.md', 'r') as f:
    lines = f.readlines()

# Define insertions (line_number, text_to_insert_after)
insertions = []

# Find insertion points and prepare new content
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    new_lines.append(line)

    # Insertion 1: After three pre-training approaches paragraph
    if i < len(lines) - 1 and "Each approach excels at different downstream tasks." in line:
        new_lines.append('\n')
        new_lines.append('![Pre-training Paradigms](diagrams/pretraining-paradigms.mmd)\n')
        new_lines.append('\n')
        new_lines.append('*Figure 1: Comparison of three pre-training paradigms. BERT uses masked language modeling with bidirectional context for understanding tasks, GPT uses causal language modeling with left context for generation tasks, and T5 uses sequence-to-sequence with a text-to-text framework for transformation tasks.*\n')
        new_lines.append('\n')

    # Insertion 2: After the key concept box
    elif i < len(lines) - 1 and "> **Key Concept:**" in line and "Pre-training on massive unlabeled text" in line:
        new_lines.append('\n')
        new_lines.append('![Transfer Learning Workflow](diagrams/transfer-learning-workflow.mmd)\n')
        new_lines.append('\n')
        new_lines.append('*Figure 2: The transfer learning pipeline for modern NLP. Pre-training on billions of unlabeled words creates general language understanding (days/weeks on hundreds of GPUs). Fine-tuning on thousands of labeled task-specific examples specializes the model (hours/minutes on a single GPU).*\n')
        new_lines.append('\n')

    # Insertion 3: Replace visualization section with static figure
    elif "## Visualization" in line:
        # Skip ahead to find the end of the code block and explanation
        new_lines.append('\n')
        new_lines.append('![Static vs Contextualized Embeddings](diagrams/static-vs-contextualized.png)\n')
        new_lines.append('\n')
        new_lines.append('*Figure 3: Static embeddings (Word2Vec) assign the same vector to "bank" regardless of context, while contextualized embeddings (BERT) create different representations based on surrounding words, correctly distinguishing river banks from financial banks.*\n')
        new_lines.append('\n')
        # Skip the code block (find where it ends)
        i += 1
        while i < len(lines) and not lines[i].startswith('This visualization demonstrates'):
            i += 1
        # Skip the explanation paragraph too
        while i < len(lines) and lines[i].strip() != '' and not lines[i].startswith('##'):
            i += 1
        i -= 1  # Back up one so the main loop will process the next section header

    # Insertion 4: Before text generation section
    elif "### Part 3: Text Generation with Sampling Strategies" in line:
        new_lines.append('\n')
        new_lines.append('![Sampling Strategies for Text Generation](diagrams/sampling-strategies.png)\n')
        new_lines.append('\n')
        new_lines.append('*Figure 4: Text generation sampling strategies. Greedy decoding always selects the highest probability token (deterministic but repetitive). Top-k sampling restricts choices to k most probable tokens. Nucleus (top-p) sampling dynamically adjusts the candidate set based on cumulative probability. Temperature controls the randomness of the distribution, with low values being conservative and high values being creative.*\n')
        new_lines.append('\n')

    # Insertion 5: Before NER section
    elif "### Part 4: Named Entity Recognition with BERT" in line:
        new_lines.append('\n')
        new_lines.append('![BIO Tagging Scheme](diagrams/bio-tagging-scheme.png)\n')
        new_lines.append('\n')
        new_lines.append('*Figure 5: BIO tagging scheme for Named Entity Recognition. B- tags mark the beginning of an entity (B-PER for "Tim"), I- tags mark continuation tokens (I-PER for "Cook"), and O marks tokens outside any entity. Multi-word entities like "Tim Cook" are connected through B-/I- sequences, enabling the model to identify entity boundaries.*\n')
        new_lines.append('\n')

    # Insertion 6: At the comparison section (Solution 3 or before Practice Exercises)
    elif "**Solution 3**" in line or "## Practice Exercises" in line:
        # Add before Practice Exercises or in Solution 3
        if "## Practice Exercises" in line:
            new_lines.insert(-1, '\n')
            new_lines.insert(-1, '![Performance Comparison](diagrams/performance-comparison.png)\n')
            new_lines.insert(-1, '\n')
            new_lines.insert(-1, '*Figure 6: Comparison of different NLP approaches on sentiment analysis. Fine-tuned BERT achieves the highest accuracy (92%) but requires 2000 labeled examples and 15 minutes of training. Few-shot prompting offers a middle ground (75% accuracy) with only 5 examples and no training time. The trade-off between performance, data requirements, and training time determines the best approach for each use case.*\n')
            new_lines.insert(-1, '\n')

    i += 1

# Write the updated content
with open('../content.md', 'w') as f:
    f.writelines(new_lines)

print("Successfully updated content.md with diagram references!")
print(f"Original lines: {len(lines)}")
print(f"New lines: {len(new_lines)}")

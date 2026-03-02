"""
Code validation script for Chapter 28: Modern NLP
Tests all code blocks from content.md sequentially
"""

import sys
import traceback

# Track results
results = {
    'total_blocks': 0,
    'passed': 0,
    'failed': 0,
    'errors': []
}

def test_block(block_num, description):
    """Decorator to track code block testing"""
    def decorator(func):
        def wrapper():
            results['total_blocks'] += 1
            print(f"\n{'='*70}")
            print(f"Testing Block {block_num}: {description}")
            print(f"{'='*70}")
            try:
                func()
                results['passed'] += 1
                print(f"✓ Block {block_num} PASSED")
                return True
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Block {block_num} ({description}): {str(e)}"
                results['errors'].append({
                    'block': block_num,
                    'description': description,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                print(f"✗ Block {block_num} FAILED: {str(e)}")
                return False
        return wrapper
    return decorator


# =============================================================================
# BLOCK 1: Pre-training Paradigms (BERT, GPT-2, T5)
# =============================================================================

@test_block(1, "Pre-training Paradigms - BERT, GPT-2, T5")
def block_1():
    from transformers import (
        BertTokenizer, BertForMaskedLM,
        GPT2Tokenizer, GPT2LMHeadModel,
        T5Tokenizer, T5ForConditionalGeneration
    )
    import torch

    # Set random seed
    torch.manual_seed(42)

    print("PART 1: Masked Language Modeling (BERT)")

    # Load BERT for masked language modeling
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    bert_model.eval()

    # Example: Predict masked word using bidirectional context
    sentence = "The cat sat on the [MASK]."
    inputs = bert_tokenizer(sentence, return_tensors='pt')

    with torch.no_grad():
        outputs = bert_model(**inputs)
        predictions = outputs.logits

    # Get top 5 predictions for the masked token
    mask_token_index = torch.where(inputs['input_ids'] == bert_tokenizer.mask_token_id)[1]
    mask_token_logits = predictions[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    print(f"\nInput: {sentence}")
    print("Top 5 predictions for [MASK]:")
    for i, token_id in enumerate(top_5_tokens, 1):
        token = bert_tokenizer.decode([token_id])
        print(f"  {i}. {token}")

    print("\nPART 2: Causal Language Modeling (GPT-2)")

    # Load GPT-2 for causal language modeling
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt_model.eval()

    # Example: Predict next word using only left context
    prompt = "The cat sat on the"
    inputs = gpt_tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = gpt_model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    # Get top 5 next-token predictions
    top_5_next = torch.topk(next_token_logits, 5).indices.tolist()

    print(f"\nPrompt: {prompt}")
    print("Top 5 predictions for next word:")
    for i, token_id in enumerate(top_5_next, 1):
        token = gpt_tokenizer.decode([token_id])
        print(f"  {i}. {token}")

    print("\nPART 3: Sequence-to-Sequence (T5)")

    # Load T5 for text-to-text generation
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
    t5_model.eval()

    # T5 frames tasks as text-to-text with task prefixes
    tasks = [
        ("translate English to French: Hello, how are you?", "Translation"),
        ("summarize: The transformer architecture revolutionized NLP by using attention mechanisms.", "Summarization"),
        ("question: What is the capital of France?", "Question Answering")
    ]

    for input_text, task_name in tasks:
        inputs = t5_tokenizer(input_text, return_tensors='pt')

        with torch.no_grad():
            outputs = t5_model.generate(**inputs, max_length=50)

        result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTask: {task_name}")
        print(f"Input: {input_text}")
        print(f"Output: {result}")


# =============================================================================
# BLOCK 2: Fine-tuning BERT for Sentiment Analysis
# =============================================================================

@test_block(2, "Fine-tuning BERT for Sentiment Analysis")
def block_2():
    import numpy as np
    import pandas as pd
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import torch

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("FINE-TUNING BERT FOR SENTIMENT ANALYSIS")

    # Step 1: Load and explore the IMDB dataset
    print("\n[Step 1] Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Use subset for faster training
    train_dataset = dataset['train'].shuffle(seed=42).select(range(5000))
    test_dataset = dataset['test'].shuffle(seed=42).select(range(1000))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"\nExample review:")
    print(f"Text: {train_dataset[0]['text'][:200]}...")
    print(f"Label: {'Positive' if train_dataset[0]['label'] == 1 else 'Negative'}")

    # Step 2: Load pre-trained model and tokenizer
    print("\n[Step 2] Loading DistilBERT model...")
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    print(f"Model: {model_name}")
    print(f"Parameters: {model.num_parameters():,}")

    # Step 3: Tokenize the dataset
    print("\n[Step 3] Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )

    # Apply tokenization to entire dataset
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    print("Tokenization complete.")
    print(f"Sample tokenized input shape: {train_dataset[0]['input_ids'].shape}")

    # Step 4: Define evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Step 5: Set up training arguments
    print("\n[Step 4] Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir='./logs',
        logging_steps=100,
        seed=42
    )

    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Weight decay: {training_args.weight_decay}")

    # Step 6: Initialize Trainer and train
    print("\n[Step 5] Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    train_result = trainer.train()

    print("\nTraining complete!")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")

    # Step 7: Evaluate on test set
    print("\n[Step 6] Evaluating on test set...")
    eval_results = trainer.evaluate()

    print("\nTest Set Results:")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall: {eval_results['eval_recall']:.4f}")

    # Step 8: Test on custom examples
    print("\n[Step 7] Testing on custom examples...")

    custom_reviews = [
        "This movie was absolutely brilliant and captivating!",
        "Terrible waste of time, boring and poorly acted.",
        "It was okay, not great but not bad either.",
        "An extraordinary masterpiece with stunning performances.",
        "Disappointed and underwhelmed by the entire experience."
    ]

    # Prepare inputs
    inputs = tokenizer(custom_reviews, padding=True, truncation=True,
                       max_length=256, return_tensors='pt')

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    print("\nCustom Review Predictions:")
    for review, pred in zip(custom_reviews, predictions):
        sentiment = "Positive" if pred[1] > pred[0] else "Negative"
        confidence = max(pred[0], pred[1]).item()
        print(f"\nReview: {review}")
        print(f"Prediction: {sentiment} (confidence: {confidence:.4f})")

    print("\nFine-tuning complete! Model saved to ./results")


# =============================================================================
# BLOCK 3: Text Generation with Sampling Strategies
# =============================================================================

@test_block(3, "Text Generation with Sampling Strategies")
def block_3():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch

    # Set random seed
    torch.manual_seed(42)

    print("TEXT GENERATION SAMPLING STRATEGIES")

    # Load GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Once upon a time, in a small village"

    print(f"\nPrompt: '{prompt}'\n")

    # Strategy 1: Greedy Decoding
    print("Strategy 1: Greedy Decoding (deterministic)")

    inputs = tokenizer(prompt, return_tensors='pt')
    greedy_output = model.generate(
        **inputs,
        max_length=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    print(f"\n{greedy_text}")

    # Strategy 2: Top-k Sampling
    print("\nStrategy 2: Top-k Sampling (k=50)")

    torch.manual_seed(42)
    topk_output = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_k=50,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

    topk_text = tokenizer.decode(topk_output[0], skip_special_tokens=True)
    print(f"\n{topk_text}")

    # Strategy 3: Nucleus Sampling (Top-p)
    print("\nStrategy 3: Nucleus Sampling (top-p=0.9)")

    torch.manual_seed(42)
    nucleus_output = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )

    nucleus_text = tokenizer.decode(nucleus_output[0], skip_special_tokens=True)
    print(f"\n{nucleus_text}")

    # Strategy 4: Temperature Variations
    print("\nStrategy 4: Temperature Effects")

    temperatures = [0.3, 0.7, 1.2]

    for temp in temperatures:
        torch.manual_seed(42)
        output = model.generate(
            **inputs,
            max_length=50,
            do_sample=True,
            top_p=0.9,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nTemperature {temp}:")
        print(f"{text}")

    # Strategy 5: Controlling Repetition
    print("\nStrategy 5: Preventing Repetition")

    # Without repetition penalty
    torch.manual_seed(42)
    output_repetitive = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    # With repetition penalty
    torch.manual_seed(42)
    output_diverse = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    print("\nWithout repetition control:")
    print(tokenizer.decode(output_repetitive[0], skip_special_tokens=True))

    print("\nWith repetition control:")
    print(tokenizer.decode(output_diverse[0], skip_special_tokens=True))

    # Strategy 6: Multiple Diverse Completions
    print("\nStrategy 6: Multiple Diverse Completions")

    torch.manual_seed(42)
    diverse_outputs = model.generate(
        **inputs,
        max_length=40,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=3,
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"\nGenerating 3 diverse completions for: '{prompt}'\n")
    for i, output in enumerate(diverse_outputs, 1):
        text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Completion {i}:")
        print(f"{text}\n")

    print("Summary: Greedy is deterministic but repetitive, top-k and top-p")
    print("add creativity, temperature controls randomness, and repetition")
    print("penalties improve diversity.")


# =============================================================================
# BLOCK 4: Named Entity Recognition with BERT
# =============================================================================

@test_block(4, "Named Entity Recognition with BERT")
def block_4():
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline
    )
    from datasets import load_dataset
    import numpy as np

    # Set random seed
    np.random.seed(42)

    print("NAMED ENTITY RECOGNITION WITH BERT")

    # Step 1: Load pre-trained NER model
    print("\n[Step 1] Loading pre-trained BERT NER model...")

    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Create NER pipeline
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    print(f"Model: {model_name}")
    print(f"Entity types: PER (person), ORG (organization), LOC (location), MISC (miscellaneous)")

    # Step 2: Test on example sentences
    print("\n[Step 2] Testing on example sentences...")

    examples = [
        "Apple CEO Tim Cook announced new products in San Francisco yesterday.",
        "The United Nations met in Geneva to discuss climate change.",
        "Elon Musk founded Tesla and SpaceX in the United States.",
        "Microsoft acquired LinkedIn for $26 billion in 2016.",
        "Angela Merkel was the Chancellor of Germany from Berlin."
    ]

    for i, text in enumerate(examples, 1):
        print(f"\nExample {i}: {text}")

        # Get predictions
        entities = ner_pipeline(text)

        if entities:
            print("Detected Entities:")
            for entity in entities:
                print(f"  • {entity['word']:<20} → {entity['entity_group']:<6} (confidence: {entity['score']:.3f})")
        else:
            print("No entities detected.")

    # Step 3: Visualize entities in context
    print("\n[Step 3] Visualizing entities with color coding...")

    def visualize_entities(text, entities):
        """Color-code entities in text for visualization."""
        colors = {
            'PER': '\033[92m',
            'ORG': '\033[94m',
            'LOC': '\033[91m',
            'MISC': '\033[93m'
        }
        reset = '\033[0m'

        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)

        colored_text = text
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            entity_type = entity['entity_group']
            color = colors.get(entity_type, '')

            colored_text = (colored_text[:start] +
                           color + colored_text[start:end] + reset +
                           colored_text[end:])

        return colored_text

    test_sentence = "Microsoft CEO Satya Nadella spoke at the conference in Seattle."
    entities = ner_pipeline(test_sentence)

    print(f"\nOriginal: {test_sentence}")
    print(f"Colored:  {visualize_entities(test_sentence, entities)}")
    print("\nLegend: [PER=Green] [ORG=Blue] [LOC=Red] [MISC=Yellow]")

    # Step 4: Demonstrate BIO tagging scheme
    print("\n[Step 4] Understanding BIO tagging...")

    ner_pipeline_tokens = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy=None)

    sentence = "Tim Cook works at Apple in California"
    token_predictions = ner_pipeline_tokens(sentence)

    print(f"\nSentence: {sentence}")
    print("\nToken-level BIO tags:")
    print(f"{'Token':<15} {'Tag':<12} {'Confidence':<10}")
    print("-" * 40)

    for pred in token_predictions:
        if not pred['word'].startswith('##'):
            token = pred['word']
            tag = pred['entity']
            score = pred['score']
            print(f"{token:<15} {tag:<12} {score:.3f}")

    print("\nBIO Scheme Explanation:")
    print("  B-PER: Beginning of a person entity")
    print("  I-PER: Inside (continuation) of a person entity")
    print("  O: Outside any entity (not part of named entity)")

    # Step 5: Error analysis - ambiguous entities
    print("\n[Step 5] Handling ambiguous entities...")

    ambiguous_cases = [
        "Washington signed the declaration in Washington.",
        "I bought an Apple from Apple Store.",
        "Jordan won gold at the Olympics."
    ]

    print("\nAmbiguous cases where context is crucial:")
    for case in ambiguous_cases:
        print(f"\nSentence: {case}")
        entities = ner_pipeline(case)
        if entities:
            for entity in entities:
                print(f"  → {entity['word']}: {entity['entity_group']} ({entity['score']:.3f})")
        else:
            print("  → No entities detected")

    print("\nNER Complete! BERT successfully identifies and classifies entities")
    print("using contextualized representations and BIO tagging.")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CHAPTER 28: MODERN NLP - CODE VALIDATION")
    print("="*70)

    # Run all test blocks
    block_1()
    block_2()
    block_3()
    block_4()

    # Print summary
    print("\n" + "="*70)
    print("CODE VALIDATION SUMMARY")
    print("="*70)
    print(f"Total blocks tested: {results['total_blocks']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")

    if results['failed'] > 0:
        print(f"\n{'='*70}")
        print("FAILURES:")
        print("="*70)
        for error in results['errors']:
            print(f"\nBlock {error['block']}: {error['description']}")
            print(f"Error: {error['error']}")
            print(f"Traceback:\n{error['traceback']}")
    else:
        print("\n✓ All code blocks passed!")

    sys.exit(0 if results['failed'] == 0 else 1)

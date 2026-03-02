"""
Code Review Test for Chapter 28: Modern NLP
Validates all code blocks work correctly
"""

import sys
import traceback

# Track results
results = []

def test_block(block_name, test_func):
    """Test a code block and record results"""
    print(f"\n{'='*70}")
    print(f"Testing: {block_name}")
    print('='*70)
    try:
        test_func()
        results.append({'block': block_name, 'status': 'PASS', 'error': None})
        print(f"✓ {block_name} PASSED")
        return True
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        results.append({'block': block_name, 'status': 'FAIL', 'error': error_msg})
        print(f"✗ {block_name} FAILED")
        print(f"Error: {error_msg}")
        traceback.print_exc()
        return False

# =============================================================================
# BLOCK 1: Part 1 - Understanding Pre-training Paradigms
# =============================================================================

def test_block_1():
    """Test Part 1: Understanding Pre-training Paradigms"""
    from transformers import (
        BertTokenizer, BertForMaskedLM,
        GPT2Tokenizer, GPT2LMHeadModel,
        T5Tokenizer, T5ForConditionalGeneration
    )
    import torch

    # Set random seed
    torch.manual_seed(42)

    print("=" * 70)
    print("PART 1: Masked Language Modeling (BERT)")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("PART 2: Causal Language Modeling (GPT-2)")
    print("=" * 70)

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

    print("\n" + "=" * 70)
    print("PART 3: Sequence-to-Sequence (T5)")
    print("=" * 70)

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
# BLOCK 2: Part 2 - Fine-tuning BERT for Sentiment Analysis
# =============================================================================

def test_block_2():
    """Test Part 2: Fine-tuning BERT for Sentiment Analysis"""
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

    print("=" * 70)
    print("FINE-TUNING BERT FOR SENTIMENT ANALYSIS")
    print("=" * 70)

    # Step 1: Load and explore the IMDB dataset
    print("\n[Step 1] Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # Use subset for faster training
    train_dataset = dataset['train'].shuffle(seed=42).select(range(100))  # Very small for testing
    test_dataset = dataset['test'].shuffle(seed=42).select(range(50))

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

    # Apply tokenization
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

    # Step 5: Set up training arguments (minimal for testing)
    print("\n[Step 4] Setting up training configuration...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # Reduced for testing
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir='./logs',
        logging_steps=10,
        seed=42
    )

    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")

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

    # Step 8: Test on custom examples
    print("\n[Step 7] Testing on custom examples...")

    custom_reviews = [
        "This movie was absolutely brilliant and captivating!",
        "Terrible waste of time, boring and poorly acted.",
        "An extraordinary masterpiece with stunning performances."
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

    print("\n" + "=" * 70)
    print("Fine-tuning complete!")
    print("=" * 70)

# =============================================================================
# BLOCK 3: Part 3 - Text Generation with Sampling Strategies
# =============================================================================

def test_block_3():
    """Test Part 3: Text Generation with Sampling Strategies"""
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch

    # Set random seed
    torch.manual_seed(42)

    print("=" * 70)
    print("TEXT GENERATION SAMPLING STRATEGIES")
    print("=" * 70)

    # Load GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Once upon a time, in a small village"

    print(f"\nPrompt: '{prompt}'\n")

    # Strategy 1: Greedy Decoding
    print("=" * 70)
    print("Strategy 1: Greedy Decoding (deterministic)")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    print("Strategy 2: Top-k Sampling (k=50)")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    print("Strategy 3: Nucleus Sampling (top-p=0.9)")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    print("Strategy 4: Temperature Effects")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    print("Strategy 5: Preventing Repetition")
    print("=" * 70)

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
    print("\n" + "=" * 70)
    print("Strategy 6: Multiple Diverse Completions")
    print("=" * 70)

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

    print("=" * 70)
    print("Summary: Greedy is deterministic but repetitive, top-k and top-p")
    print("add creativity, temperature controls randomness, and repetition")
    print("penalties improve diversity.")
    print("=" * 70)

# =============================================================================
# BLOCK 4: Part 4 - Named Entity Recognition with BERT
# =============================================================================

def test_block_4():
    """Test Part 4: Named Entity Recognition with BERT"""
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        pipeline
    )
    import numpy as np

    # Set random seed
    np.random.seed(42)

    print("=" * 70)
    print("NAMED ENTITY RECOGNITION WITH BERT")
    print("=" * 70)

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
        "Elon Musk founded Tesla and SpaceX in the United States."
    ]

    for i, text in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}: {text}")
        print(f"{'='*70}")

        entities = ner_pipeline(text)

        if entities:
            print("\nDetected Entities:")
            for entity in entities:
                print(f"  • {entity['word']:<20} → {entity['entity_group']:<6} (confidence: {entity['score']:.3f})")
        else:
            print("No entities detected.")

    # Step 3: Visualize entities
    print("\n[Step 3] Visualizing entities with color coding...")

    def visualize_entities(text, entities):
        """Color-code entities in text."""
        colors = {
            'PER': '\033[92m',   # Green
            'ORG': '\033[94m',   # Blue
            'LOC': '\033[91m',   # Red
            'MISC': '\033[93m'   # Yellow
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

    # Step 4: Demonstrate BIO tagging
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
    print("  O: Outside any entity")

    # Step 5: Ambiguous entities
    print("\n[Step 5] Handling ambiguous entities...")

    ambiguous_cases = [
        "Washington signed the declaration in Washington.",
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

    print("\n" + "=" * 70)
    print("NER Complete!")
    print("=" * 70)

# =============================================================================
# SOLUTION 1: AG News Classification
# =============================================================================

def test_solution_1():
    """Test Solution 1: AG News Classification"""
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer
    )
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import numpy as np
    import torch

    np.random.seed(42)
    torch.manual_seed(42)

    # Load AG News dataset (small subset)
    dataset = load_dataset("ag_news")
    train_subset = dataset['train'].shuffle(seed=42).select(range(100))
    test_subset = dataset['test'].shuffle(seed=42).select(range(50))

    print(f"Training samples: {len(train_subset)}")
    print(f"Test samples: {len(test_subset)}")

    label_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    # Load model
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    tokenized_train = train_subset.map(tokenize_function, batched=True)
    tokenized_test = test_subset.map(tokenize_function, batched=True)

    tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Define metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}

    # Training (minimal)
    training_args = TrainingArguments(
        output_dir='./ag_news_results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        evaluation_strategy='epoch',
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    print(f"\nTest Accuracy: {results['eval_accuracy']:.4f}")

    # Test custom headlines
    custom_headlines = [
        "Lakers defeat Celtics in championship game",
        "Stock market reaches new record high"
    ]

    inputs = tokenizer(custom_headlines, padding=True, truncation=True, max_length=128, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    print("\nCustom Headline Predictions:")
    for headline, pred in zip(custom_headlines, predictions):
        predicted_class = label_names[pred.argmax()]
        confidence = pred.max().item()
        print(f"Headline: {headline}")
        print(f"Predicted: {predicted_class} (confidence: {confidence:.4f})")

# =============================================================================
# SOLUTION 2: Question Answering
# =============================================================================

def test_solution_2():
    """Test Solution 2: Question Answering"""
    from transformers import pipeline
    import torch

    torch.manual_seed(42)

    # Load pre-trained QA model
    model_name = 'distilbert-base-uncased-distilled-squad'
    qa_pipeline = pipeline('question-answering', model=model_name)

    custom_qa = [
        {
            'context': "Python is a high-level, interpreted programming language. Created by Guido van Rossum and first released in 1991.",
            'question': "Who created Python?"
        },
        {
            'context': "The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889.",
            'question': "When was the Eiffel Tower built?"
        }
    ]

    print("="*70)
    print("QUESTION ANSWERING")
    print("="*70)

    for i, qa in enumerate(custom_qa, 1):
        result = qa_pipeline(question=qa['question'], context=qa['context'])

        print(f"\n[Example {i}]")
        print(f"Question: {qa['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['score']:.4f}")

# =============================================================================
# SOLUTION 3: Prompting vs Fine-tuning (simplified)
# =============================================================================

def test_solution_3():
    """Test Solution 3: Prompting vs Fine-tuning comparison"""
    import numpy as np
    from datasets import load_dataset
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch

    np.random.seed(42)
    torch.manual_seed(42)

    # Load small IMDB subset
    dataset = load_dataset("imdb")
    test_subset = dataset['test'].shuffle(seed=42).select(range(10))

    print("="*70)
    print("PROMPTING VS FINE-TUNING COMPARISON")
    print("="*70)

    # Few-shot prompting with GPT-2
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt_model.eval()

    few_shot_prompt = """Review: This movie was fantastic!
Sentiment: positive

Review: Boring and poorly made.
Sentiment: negative

Review: {review}
Sentiment:"""

    def predict_sentiment(review):
        prompt = few_shot_prompt.format(review=review[:100])
        inputs = gpt_tokenizer(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = gpt_model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 5,
                do_sample=False,
                pad_token_id=gpt_tokenizer.eos_token_id
            )

        completion = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = completion[len(prompt):].strip().lower()

        if 'positive' in response[:20]:
            return 1
        elif 'negative' in response[:20]:
            return 0
        else:
            return np.random.randint(0, 2)

    # Test few-shot prompting
    print("\nFew-shot Prompting Test:")
    predictions = []
    labels = []

    for ex in test_subset:
        pred = predict_sentiment(ex['text'])
        predictions.append(pred)
        labels.append(ex['label'])

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(labels, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tested on {len(test_subset)} examples")

    print("\nConclusion:")
    print("• Few-shot prompting: Fast, no training, but lower accuracy")
    print("• Fine-tuning: Better performance with sufficient data")

# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CODE REVIEW: Chapter 28 - Modern NLP")
    print("="*70)

    tests = [
        ("Block 1: Pre-training Paradigms", test_block_1),
        ("Block 2: Fine-tuning BERT (Sentiment)", test_block_2),
        ("Block 3: Text Generation Strategies", test_block_3),
        ("Block 4: Named Entity Recognition", test_block_4),
        ("Solution 1: AG News Classification", test_solution_1),
        ("Solution 2: Question Answering", test_solution_2),
        ("Solution 3: Prompting vs Fine-tuning", test_solution_3),
    ]

    for name, test_func in tests:
        test_block(name, test_func)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')

    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed Tests:")
        for r in results:
            if r['status'] == 'FAIL':
                print(f"  - {r['block']}")
                print(f"    Error: {r['error']}")

    # Determine rating
    if failed == 0:
        rating = "ALL_PASS"
    elif failed <= 2:
        rating = "MINOR_FIXES"
    else:
        rating = "BROKEN"

    print(f"\n{'='*70}")
    print(f"FINAL RATING: {rating}")
    print(f"{'='*70}")

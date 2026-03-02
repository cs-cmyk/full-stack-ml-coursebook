import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load 20 Newsgroups dataset - 2 categories for simplicity
categories = ['sci.space', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train',
                                categories=categories,
                                remove=('headers', 'footers', 'quotes'),
                                random_state=42)

# Extract texts and labels
texts = newsgroups.data
y = newsgroups.target
target_names = newsgroups.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.25, random_state=42, stratify=y
)

# Create pipelines with different approaches
pipelines = {
    'CountVectorizer': Pipeline([
        ('vectorizer', CountVectorizer(max_features=1000, stop_words='english')),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'TF-IDF': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'TF-IDF + Bigrams': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=2000, stop_words='english',
                                      ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
}

# Train and evaluate each pipeline
results = {}
for name, pipeline in pipelines.items():
    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Create comparison bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
methods = list(results.keys())
accuracies = list(results.values())
bars = ax1.bar(methods, accuracies, color=['#2196F3', '#4CAF50', '#FF9800'], alpha=0.8)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Comparison: Text Vectorization Methods', fontsize=13, weight='bold')
ax1.set_ylim(0.92, 0.97)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=10, weight='bold')

# Feature importance from best model
best_pipeline = pipelines['TF-IDF + Bigrams']
vectorizer = best_pipeline.named_steps['vectorizer']
classifier = best_pipeline.named_steps['classifier']
feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_[0]

# Get top positive and negative features
top_positive_indices = coefficients.argsort()[-10:][::-1]
top_negative_indices = coefficients.argsort()[:10]

top_features = np.concatenate([top_positive_indices, top_negative_indices])
top_coefs = coefficients[top_features]
top_names = feature_names[top_features]

# Create colors: positive = green, negative = red
colors = ['#4CAF50' if c > 0 else '#F44336' for c in top_coefs]

# Plot feature importance
y_pos = np.arange(len(top_names))
ax2.barh(y_pos, top_coefs, color=colors, alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_names, fontsize=9)
ax2.set_xlabel('Coefficient Value', fontsize=12)
ax2.set_title('Top Features: sci.space (green) vs baseball (red)',
              fontsize=13, weight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('text_classification_results.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Visualization saved to text_classification_results.png")
print(f"\nAccuracy Results:")
for method, acc in results.items():
    print(f"  {method}: {acc:.4f}")

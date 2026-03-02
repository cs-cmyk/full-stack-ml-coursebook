import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load data and prepare train/test splits
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate metrics
TP = cm[1, 1]
TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Confusion matrix with counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            annot_kws={'size': 16, 'weight': 'bold'})
axes[0].set_xlabel('Predicted Class', fontsize=13, weight='bold')
axes[0].set_ylabel('Actual Class', fontsize=13, weight='bold')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, weight='bold')

# Add labels for each quadrant
axes[0].text(0.5, 0.25, 'TN', fontsize=11, ha='center', color='darkblue', weight='bold')
axes[0].text(1.5, 0.25, 'FP', fontsize=11, ha='center', color='darkred', weight='bold')
axes[0].text(0.5, 1.25, 'FN', fontsize=11, ha='center', color='darkred', weight='bold')
axes[0].text(1.5, 1.25, 'TP', fontsize=11, ha='center', color='darkgreen', weight='bold')

# Plot 2: Metrics comparison bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
colors = ['#2196F3', '#9C27B0', '#FF9800', '#4CAF50']

bars = axes[1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_ylim(0, 1.0)
axes[1].set_ylabel('Score', fontsize=13, weight='bold')
axes[1].set_title('Evaluation Metrics Comparison', fontsize=14, weight='bold')
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-04-ml/ch19-classification/diagrams/confusion_matrix_metrics.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: confusion_matrix_metrics.png")

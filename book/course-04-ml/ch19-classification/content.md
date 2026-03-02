> **© 2026 Chirag Shinde. Licensed under CC BY-NC-SA 4.0.**
> See [LICENSE](../../LICENSE) for details.

---

# 19.1: Classification Fundamentals

## Why This Matters

Every day, algorithms decide whether an email is spam, whether a tumor is malignant, or whether a credit card transaction is fraudulent. These aren't predictions of numbers—they're predictions of categories. Classification is the machine learning task that powers these critical decisions, and understanding how it works means understanding the algorithms that shape modern life. Unlike regression, where continuous values are predicted, classification predicts discrete classes, and getting the evaluation wrong can mean the difference between catching cancer early and missing it entirely.

## Intuition

Imagine an assistant hired to sort an inbox. Their job is simple: look at each email and decide whether it's SPAM or NOT SPAM. They can't give a wishy-washy answer like "this email is 47% spam"—they need to make a decision. That's classification.

But here's the interesting part: the assistant isn't flipping a coin. They're making informed probability judgments. When they see an email with words like "FREE!!!" and "CLICK NOW," they might think, "I'm 95% confident this is spam." When they see an email from the boss, they think, "I'm only 5% confident this is spam." Then they apply a simple rule: if the probability is above 50%, label it SPAM; otherwise, label it NOT SPAM.

This is exactly how logistic regression works—the most fundamental classification algorithm. It takes data (features like word frequencies in an email) and calculates the probability that the email belongs to the "spam" class. Then it applies a threshold (usually 0.5) to make the final decision.

Now, here's where it gets real: not all mistakes are equal. If the assistant labels a real email as spam (a false positive), an important message from the boss might be missed. If they let spam through (a false negative), 10 seconds are wasted deleting it. The consequences are asymmetric. In medical diagnosis, the stakes are even higher: missing a cancer case (false negative) could be fatal, while a false alarm (false positive) just means an extra test. This is why sophisticated evaluation metrics beyond simple accuracy are needed.

Finally, imagine the assistant now sorts mail into four categories: SPAM, WORK, PERSONAL, and PROMOTIONS. Same basic idea, but now they're calculating four probabilities (one for each category) and picking the highest one. That's multi-class classification—a natural extension of the binary case.

The power of classification is that once this fundamental pattern is understood, it can be applied to thousands of problems: detecting fraud, diagnosing diseases, recognizing images, filtering spam, predicting customer churn, and so much more.

## Formal Definition

**Classification** is a supervised learning task where a discrete category (class label) is predicted from input features. Given a feature matrix **X** (n × p) and a target vector **y** containing class labels, a function f: **X** → **y** is learned that maps features to class predictions.

For **binary classification**, there are two possible classes, typically encoded as 0 and 1 (or negative/positive). The model learns a decision boundary that separates the two classes in feature space.

**Logistic Regression** is a linear classification algorithm that models the probability of class membership using the logistic (sigmoid) function:

$$
P(y=1|\mathbf{x}) = \sigma(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_p x_p) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \ldots + \theta_p x_p)}}
$$

where:
- **x** is the feature vector for a single sample
- θ = [θ₀, θ₁, ..., θₚ] are the model parameters (intercept and coefficients)
- σ(z) is the sigmoid function that maps any real number to the range [0, 1]

The **sigmoid function** is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The final class prediction is determined by applying a decision threshold (typically 0.5):

$$
\hat{y} = \begin{cases}
1 & \text{if } P(y=1|\mathbf{x}) \geq 0.5 \\
0 & \text{if } P(y=1|\mathbf{x}) < 0.5
\end{cases}
$$

The model parameters θ are learned by maximizing the likelihood of the observed data (Maximum Likelihood Estimation), which is equivalent to minimizing the cross-entropy loss function.

For **multi-class classification** with k classes, logistic regression extends using the softmax function:

$$
P(y=j|\mathbf{x}) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}
$$

where z_j = θ_j^T **x** is the linear combination for class j, and the probabilities sum to 1 across all classes.

> **Key Concept:** Logistic regression predicts probabilities of class membership using the sigmoid function, then applies a threshold to make discrete class predictions. Despite its name, logistic regression is a classification algorithm, not a regression algorithm.

## Visualization

The following code visualizes the sigmoid function to understand how it transforms linear combinations into probabilities:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create z values from -6 to 6
z = np.linspace(-6, 6, 200)

# Apply sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

sigma_z = sigmoid(z)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(z, sigma_z, linewidth=2.5, color='#2E86AB', label='σ(z) = 1/(1+e^(-z))')

# Add decision threshold line
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision threshold (0.5)')
plt.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Shade regions
plt.fill_between(z, 0, sigma_z, where=(sigma_z < 0.5), alpha=0.2, color='blue', label='Predict Class 0')
plt.fill_between(z, sigma_z, 1, where=(sigma_z >= 0.5), alpha=0.2, color='orange', label='Predict Class 1')

# Annotations
plt.annotate('z = θ₀ + θ₁x₁ + θ₂x₂ + ...', xy=(2.5, 0.15), fontsize=11,
             style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
plt.annotate('Output interpreted\nas probability', xy=(-3, 0.85), fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Labels and formatting
plt.xlabel('z (linear combination of features)', fontsize=12)
plt.ylabel('σ(z) - Probability', fontsize=12)
plt.title('The Sigmoid Function: Transforming Any Number Into a Probability', fontsize=14, weight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='center right', fontsize=10)
plt.ylim(-0.05, 1.05)
plt.tight_layout()

# Show the plot
plt.savefig('sigmoid_function.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# [Figure showing S-shaped sigmoid curve from 0 to 1]
# The curve shows how the sigmoid function smoothly transitions from 0 to 1,
# with the steepest change occurring at z=0 (where σ(z)=0.5)
```

**Caption:** The sigmoid function transforms any real-valued input z (the weighted sum of features) into a probability between 0 and 1. Values below 0.5 are classified as class 0; values at or above 0.5 are classified as class 1. The S-shaped curve ensures smooth transitions and never produces outputs outside [0,1].

## Examples

### Example 1: Binary Classification with Logistic Regression

```python
# Binary Classification: Predicting Tumor Malignancy
# Using the Breast Cancer Wisconsin dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data  # 30 features: tumor characteristics
y = data.target  # 0 = malignant, 1 = benign

# Create a DataFrame for exploration
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

print("Dataset shape:", X.shape)
print("\nFeature names (first 5):", data.feature_names[:5])
print("\nClass distribution:")
print(pd.Series(y).value_counts())
print("\nClass balance:")
print(pd.Series(y).value_counts(normalize=True))

# Split into training and test sets (stratify preserves class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)  # Class labels (0 or 1)
y_proba = model.predict_proba(X_test)  # Probabilities for each class

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest set accuracy: {accuracy:.4f}")

# Show predictions for first 5 test samples
print("\nSample predictions (first 5 test samples):")
print("Actual | Predicted | P(malignant) | P(benign)")
print("-" * 50)
for i in range(5):
    print(f"  {y_test[i]}    |     {y_pred[i]}     |    {y_proba[i][0]:.4f}    |  {y_proba[i][1]:.4f}")

# Output:
# Dataset shape: (569, 30)
#
# Feature names (first 5): ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness']
#
# Class distribution:
# 1    357
# 0    212
#
# Class balance:
# 1    0.627417
# 0    0.372583
#
# Training set size: 455
# Test set size: 114
#
# Test set accuracy: 0.9649
#
# Sample predictions (first 5 test samples):
# Actual | Predicted | P(malignant) | P(benign)
# --------------------------------------------------
#   1    |     1     |    0.0333    |  0.9667
#   0    |     0     |    0.9972    |  0.0028
#   1    |     1     |    0.0285    |  0.9715
#   1    |     1     |    0.0101    |  0.9899
#   1    |     1     |    0.1573    |  0.8427
```

The code above demonstrates the following steps:

1. **Data Loading**: The Breast Cancer Wisconsin dataset contains 569 tumor samples with 30 features each (characteristics like radius, texture, smoothness). The target has two classes: 0 (malignant) and 1 (benign). The dataset is moderately imbalanced: about 63% benign and 37% malignant.

2. **Train-Test Split**: `train_test_split()` with `stratify=y` ensures that the class distribution is preserved in both training and test sets. This is critical for classification problems, especially with imbalanced data. Without stratification, random splits might create unrepresentative subsets.

3. **Model Training**: `LogisticRegression()` from scikit-learn fits the model by finding the optimal coefficients θ that maximize the likelihood of the observed data. The `max_iter=10000` parameter ensures the optimization algorithm has enough iterations to converge.

4. **Making Predictions**: Two methods are used:
   - `predict()`: Returns the predicted class labels (0 or 1) by applying a 0.5 threshold to the probabilities
   - `predict_proba()`: Returns the probability for each class. Notice how the first sample has P(malignant)=0.0333 and P(benign)=0.9667, summing to 1.0

5. **Accuracy**: The model achieved 96.49% accuracy on the test set, meaning it correctly classified 110 out of 114 test samples. This is good performance, but the next section will show why accuracy alone can be misleading.

**Key Insight**: The model outputs probabilities first, then converts them to class predictions. This means both the "soft" probability prediction (useful for understanding confidence) and the "hard" class decision (needed for the final prediction) can be accessed.

### Example 2: Decision Boundary Visualization

```python
# Visualize decision boundary using 2 features for clarity
# We'll use 'mean radius' and 'mean texture' (features 0 and 1)
X_2d = X[:, [0, 1]]  # Use only 2 features
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

# Train model on 2D data
model_2d = LogisticRegression(random_state=42, max_iter=10000)
model_2d.fit(X_train_2d, y_train_2d)

# Create mesh for decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict probability across the mesh
Z = model_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
plt.colorbar(label='P(benign)')

# Plot data points
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d,
                     cmap='RdYlBu_r', edgecolors='black', s=50, alpha=0.8)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)

plt.xlabel(data.feature_names[0], fontsize=12)
plt.ylabel(data.feature_names[1], fontsize=12)
plt.title('Logistic Regression Decision Boundary\n(Using 2 Features for Visualization)',
          fontsize=14, weight='bold')
plt.legend(*scatter.legend_elements(), title='Class', labels=['Malignant', 'Benign'])
plt.tight_layout()
plt.savefig('decision_boundary_2d.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# [Figure showing decision boundary with test points]
```

For visualization purposes, a second model is trained using only 2 features (mean radius and mean texture). The contour plot shows how the model divides feature space: the black line is where P(benign) = 0.5, the decision boundary. Points on one side are classified as malignant (blue region), points on the other side as benign (red region). The color intensity shows the model's confidence—darker colors mean higher certainty.

### Example 3: Confusion Matrix and Evaluation Metrics

```python
# Comprehensive Model Evaluation Using Multiple Metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load data and prepare train/test splits (same as before)
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign

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

print("Confusion Matrix:")
print(cm)
print("\nConfusion Matrix Breakdown:")
print(f"True Negatives (TN): {cm[0, 0]} - Correctly predicted malignant")
print(f"False Positives (FP): {cm[0, 1]} - Incorrectly predicted benign (actually malignant)")
print(f"False Negatives (FN): {cm[1, 0]} - Incorrectly predicted malignant (actually benign)")
print(f"True Positives (TP): {cm[1, 1]} - Correctly predicted benign")

# Calculate metrics manually to show the formulas
TP = cm[1, 1]  # True Positives (correctly predicted benign)
TN = cm[0, 0]  # True Negatives (correctly predicted malignant)
FP = cm[0, 1]  # False Positives (predicted benign, actually malignant)
FN = cm[1, 0]  # False Negatives (predicted malignant, actually benign)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}  = (TP + TN) / Total")
print(f"           {TP} + {TN} / {TP + TN + FP + FN} = {accuracy:.4f}")
print(f"\nPrecision: {precision:.4f}  = TP / (TP + FP)")
print(f"           {TP} / ({TP} + {FP}) = {precision:.4f}")
print(f"           'Of all predicted benign, how many were correct?'")
print(f"\nRecall:    {recall:.4f}  = TP / (TP + FN)")
print(f"           {TP} / ({TP} + {FN}) = {recall:.4f}")
print(f"           'Of all actual benign cases, how many did we find?'")
print(f"\nF1 Score:  {f1:.4f}  = 2 * (Precision * Recall) / (Precision + Recall)")
print(f"           Harmonic mean of precision and recall")
print("="*60)

# Verify with sklearn functions
print("\nVerification with sklearn functions:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# Complete classification report
print("\n" + "="*60)
print("COMPLETE CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# Output:
# Confusion Matrix:
# [[42  1]
#  [ 3 68]]
#
# Confusion Matrix Breakdown:
# True Negatives (TN): 42 - Correctly predicted malignant
# False Positives (FP): 1 - Incorrectly predicted benign (actually malignant)
# False Negatives (FN): 3 - Incorrectly predicted malignant (actually benign)
# True Positives (TP): 68 - Correctly predicted benign
#
# ============================================================
# EVALUATION METRICS
# ============================================================
# Accuracy:  0.9649  = (TP + TN) / Total
#            68 + 42 / 114 = 0.9649
#
# Precision: 0.9855  = TP / (TP + FP)
#            68 / (68 + 1) = 0.9855
#            'Of all predicted benign, how many were correct?'
#
# Recall:    0.9577  = TP / (TP + FN)
#            68 / (68 + 3) = 0.9577
#            'Of all actual benign cases, how many did we find?'
#
# F1 Score:  0.9714  = 2 * (Precision * Recall) / (Precision + Recall)
#            Harmonic mean of precision and recall
# ============================================================
```

The confusion matrix is the foundation of all classification metrics. Here's what each number means:

**The Confusion Matrix**:
- **True Negatives (TN = 42)**: The model correctly identified 42 malignant tumors as malignant
- **False Positives (FP = 1)**: The model incorrectly predicted 1 malignant tumor as benign (dangerous!)
- **False Negatives (FN = 3)**: The model incorrectly predicted 3 benign tumors as malignant (unnecessary worry)
- **True Positives (TP = 68)**: The model correctly identified 68 benign tumors as benign

**Why Different Metrics Matter**:

1. **Accuracy (96.49%)**: Overall, 110 out of 114 predictions were correct. This looks great, but accuracy can be misleading. Imagine a dataset where 99% of tumors are benign. A lazy model that always predicts "benign" would get 99% accuracy while being completely useless for detecting malignant tumors!

2. **Precision (98.55%)**: Of all the tumors the model labeled as benign, 98.55% were actually benign. High precision means few false alarms. This matters when the cost of a false positive is high. For example, in a spam filter, high precision means real emails are rarely marked as spam.

3. **Recall (95.77%)**: Of all the actually benign tumors, the model found 95.77% of them. High recall means few positive cases are missed. In cancer diagnosis, high recall is critical—malignant cases cannot be missed. Missing cancer (false negative) is far worse than an unnecessary biopsy (false positive).

4. **F1 Score (97.14%)**: The harmonic mean of precision and recall. It balances both metrics. Use F1 when precision and recall are equally important, or when classes are imbalanced. The harmonic mean (not arithmetic mean) is used because it severely penalizes extreme values—a model with 100% precision but 10% recall would have a low F1 score.

**Medical Context**: In this cancer diagnosis scenario, the model made 4 errors:
- 1 false positive (predicted benign but actually malignant) → Patient might not get treatment
- 3 false negatives (predicted malignant but actually benign) → Patient gets unnecessary follow-up tests

Which error is worse? In medicine, false negatives (missed cancer) are typically more dangerous than false positives (unnecessary tests). This is why recall might be prioritized over precision for cancer screening. Different contexts demand different metric priorities.

### Example 4: Visualization of Metrics

```python
# Visualize confusion matrix with heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Confusion matrix with counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            annot_kws={'size': 16, 'weight': 'bold'})
axes[0].set_xlabel('Predicted Class', fontsize=12, weight='bold')
axes[0].set_ylabel('Actual Class', fontsize=12, weight='bold')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, weight='bold')

# Add labels for each quadrant
axes[0].text(0.5, 0.25, 'TN', fontsize=10, ha='center', color='darkblue', weight='bold')
axes[0].text(1.5, 0.25, 'FP', fontsize=10, ha='center', color='darkred', weight='bold')
axes[0].text(0.5, 1.25, 'FN', fontsize=10, ha='center', color='darkred', weight='bold')
axes[0].text(1.5, 1.25, 'TP', fontsize=10, ha='center', color='darkgreen', weight='bold')

# Plot 2: Metrics comparison bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

bars = axes[1].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_ylim(0, 1.0)
axes[1].set_ylabel('Score', fontsize=12, weight='bold')
axes[1].set_title('Evaluation Metrics Comparison', fontsize=14, weight='bold')
axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# [Figure showing confusion matrix heatmap and metrics bar chart]
```

The visualization provides a clear comparison of all evaluation metrics and the confusion matrix structure, making it easy to identify where the model performs well and where it makes errors.

### Example 5: Multi-Class Classification

```python
# Multi-Class Classification: Iris Species Identification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load the iris dataset
data = load_iris()
X = data.data  # 4 features: sepal length, sepal width, petal length, petal width
y = data.target  # 3 classes: 0 = setosa, 1 = versicolor, 2 = virginica

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("\nClass distribution:")
print(pd.Series(y).value_counts().sort_index())
print("\nClass names:", data.target_names)
print("\nFeature names:", data.feature_names)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression model
# For multi-class, sklearn automatically uses One-vs-Rest or multinomial strategy
model = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Show probability outputs for a single sample
sample_idx = 0
print(f"\nSample prediction (test sample #{sample_idx}):")
print(f"Actual class: {y_test[sample_idx]} ({data.target_names[y_test[sample_idx]]})")
print(f"Predicted class: {y_pred[sample_idx]} ({data.target_names[y_pred[sample_idx]]})")
print(f"\nProbabilities for each class:")
for i, class_name in enumerate(data.target_names):
    print(f"  P({class_name:12s}) = {y_proba[sample_idx, i]:.4f}")
print(f"  Sum of probabilities: {y_proba[sample_idx].sum():.4f}")

# Show predictions for first 8 samples
print("\nPredictions for first 8 test samples:")
print("Actual | Predicted |  P(setosa) | P(versicolor) | P(virginica)")
print("-" * 70)
for i in range(8):
    print(f"  {y_test[i]}    |     {y_pred[i]}     | "
          f"  {y_proba[i, 0]:.4f}   |    {y_proba[i, 1]:.4f}    |   {y_proba[i, 2]:.4f}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n3×3 Confusion Matrix:")
print(cm)
print("\nReading the confusion matrix:")
print("- Row = Actual class, Column = Predicted class")
print(f"- Diagonal elements are correct predictions: {cm.diagonal().sum()} total")
print(f"- Off-diagonal elements are errors: {cm.sum() - cm.diagonal().sum()} total")

# Calculate per-class and overall metrics
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Output:
# Dataset shape: (150, 4)
# Number of classes: 3
#
# Class distribution:
# 0    50
# 1    50
# 2    50
#
# Class names: ['setosa' 'versicolor' 'virginica']
#
# Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#
# Sample prediction (test sample #0):
# Actual class: 1 (versicolor)
# Predicted class: 1 (versicolor)
#
# Probabilities for each class:
#   P(setosa      ) = 0.0000
#   P(versicolor  ) = 0.8847
#   P(virginica   ) = 0.1153
#   Sum of probabilities: 1.0000
#
# Predictions for first 8 test samples:
# Actual | Predicted |  P(setosa) | P(versicolor) | P(virginica)
# ----------------------------------------------------------------------
#   1    |     1     |   0.0000   |    0.8847    |   0.1153
#   0    |     0     |   0.9979   |    0.0021    |   0.0000
#   2    |     2     |   0.0000   |    0.1477    |   0.8523
#   1    |     1     |   0.0000   |    0.7789    |   0.2211
#   1    |     1     |   0.0000   |    0.6328    |   0.3672
#   0    |     0     |   0.9954   |    0.0046    |   0.0000
#   1    |     1     |   0.0000   |    0.8982    |   0.1018
#   2    |     2     |   0.0000   |    0.0344    |   0.9656
#
# 3×3 Confusion Matrix:
# [[15  0  0]
#  [ 0 14  1]
#  [ 0  0 15]]
#
# ======================================================================
# CLASSIFICATION REPORT
# ======================================================================
#               precision    recall  f1-score   support
#
#       setosa       1.00      1.00      1.00        15
#   versicolor       1.00      0.93      0.97        15
#    virginica       0.94      1.00      0.97        15
#
#     accuracy                           0.98        45
#    macro avg       0.98      0.98      0.98        45
# weighted avg       0.98      0.98      0.98        45
```

Multi-class classification extends naturally from binary classification. Here's what changed:

1. **Three Classes**: The Iris dataset has three species to classify instead of two. The classes are perfectly balanced (50 samples each), making this an ideal teaching dataset.

2. **Probability Outputs**: Instead of two probabilities (one for each class in binary), `predict_proba()` now returns three probabilities—one for each species. Notice that they always sum to 1.0. For example, the first test sample has P(versicolor) = 0.8847 and P(virginica) = 0.1153, with setosa essentially at 0. The model is quite confident this is versicolor.

3. **Multinomial Strategy**: `multi_class='multinomial'` is specified, which uses the softmax function to compute probabilities across all three classes simultaneously. The alternative is `multi_class='ovr'` (One-vs-Rest), which trains three separate binary classifiers. For most purposes, multinomial is more accurate and is the default in recent sklearn versions.

4. **3×3 Confusion Matrix**: The confusion matrix is now 3×3 instead of 2×2. Reading it:
   - Diagonal elements (15, 14, 15) are correct predictions
   - The only error is in row 2, column 3: one versicolor was misclassified as virginica
   - Setosa is perfectly separated (this species is very distinct from the others)
   - Versicolor and virginica are sometimes confused (they have overlapping characteristics)

5. **Per-Class Metrics**: The classification report shows precision, recall, and F1 score for each class:
   - Setosa: Perfect 1.00 on all metrics (easy to classify)
   - Versicolor: 1.00 precision (no false positives) but 0.93 recall (missed one)
   - Virginica: 0.94 precision (one false positive) but 1.00 recall (found all of them)

6. **Macro vs Weighted Averages**:
   - **Macro average**: Simple average of per-class metrics (treats each class equally)
   - **Weighted average**: Average weighted by class support (accounts for class imbalance)
   - Since Iris is perfectly balanced, both are identical here

**Key Insight**: The sklearn API is identical for binary and multi-class classification! The code remains nearly unchanged except examining three probabilities instead of two. This consistency makes it easy to extend skills from binary to multi-class problems.

### Example 6: Multi-Class Decision Boundaries

```python
# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=True, ax=axes[0],
            xticklabels=data.target_names,
            yticklabels=data.target_names,
            annot_kws={'size': 14, 'weight': 'bold'})
axes[0].set_xlabel('Predicted Class', fontsize=12, weight='bold')
axes[0].set_ylabel('Actual Class', fontsize=12, weight='bold')
axes[0].set_title('3×3 Confusion Matrix for Iris Classification', fontsize=14, weight='bold')

# Plot 2: Decision boundaries using 2 features (petal length and petal width)
# Use features 2 and 3 (petal length and petal width) for best separation
X_2d = X[:, [2, 3]]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.3, random_state=42, stratify=y
)

model_2d = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial')
model_2d.fit(X_train_2d, y_train_2d)

# Create mesh
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Predict class for each point in mesh
Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
axes[1].contourf(xx, yy, Z, alpha=0.4, cmap='viridis', levels=[0, 1, 2, 3])

# Plot test points
scatter = axes[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_2d,
                         cmap='viridis', edgecolors='black', s=80, alpha=0.9,
                         linewidths=1.5)

axes[1].set_xlabel(data.feature_names[2], fontsize=12, weight='bold')
axes[1].set_ylabel(data.feature_names[3], fontsize=12, weight='bold')
axes[1].set_title('Multi-Class Decision Boundaries\n(Using Petal Features)',
                 fontsize=14, weight='bold')

# Create custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.viridis(i/2),
                              markersize=10, label=data.target_names[i],
                              markeredgecolor='black', markeredgewidth=1.5)
                  for i in range(3)]
axes[1].legend(handles=legend_elements, title='Species', loc='upper left', fontsize=10)
axes[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('multiclass_classification.png', dpi=150, bbox_inches='tight')
plt.show()

# Output:
# [Figure showing 3×3 confusion matrix and decision boundaries]
```

The visualization shows how the model divides 2D feature space into three regions. Each region corresponds to a predicted class. Notice how versicolor and virginica have some overlap in feature space—that's where the model makes its only error.

## Common Pitfalls

**1. Using Accuracy for Imbalanced Data**

Accuracy is the most intuitive metric but also the most dangerous when classes are imbalanced. Consider a fraud detection dataset where 99.5% of transactions are legitimate and 0.5% are fraudulent. A naive model that always predicts "legitimate" achieves 99.5% accuracy while being completely useless—it never catches fraud!

**The Problem**: Accuracy treats all errors equally, but in imbalanced datasets, the minority class (often the important one) contributes little to the overall accuracy.

**What to Do Instead**:
- Always check class distribution with `pd.Series(y).value_counts()`
- For imbalanced data, use precision, recall, and F1 score instead of accuracy
- Create a confusion matrix to see where the errors occur
- Calculate a baseline: what accuracy would result from always predicting the majority class?

Example:
```python
# Check class distribution first!
print(pd.Series(y_train).value_counts(normalize=True))

# If imbalanced, focus on precision/recall instead of accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Or use F1 score as the primary metric
from sklearn.metrics import f1_score
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
```

**2. Forgetting to Stratify When Splitting**

When `train_test_split()` is used without `stratify=y`, the random split might create training and test sets with different class distributions. This is especially problematic with small datasets or rare classes.

**The Problem**: Most minority class samples might end up in the training set with few in the test set (or vice versa), giving unreliable evaluation metrics.

**What to Do Instead**:
```python
# WRONG: Random split might create imbalanced train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RIGHT: Stratified split preserves class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verify the distribution is preserved
print("Training set distribution:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nTest set distribution:")
print(pd.Series(y_test).value_counts(normalize=True))
```

**3. Confusing Precision and Recall**

Students frequently mix up precision and recall because both involve "correct predictions." Here's a foolproof way to remember:

**Precision** asks: "Of all the samples predicted as positive, how many were actually positive?"
- Focus: False Positives (Type I errors)
- Formula: TP / (TP + FP)
- Denominator: What was predicted as positive

**Recall** asks: "Of all the actual positive samples, how many were found?"
- Focus: False Negatives (Type II errors)
- Formula: TP / (TP + FN)
- Denominator: What is actually positive

**Mnemonic**: "Recall = Did you REcall (remember) to find all the positive cases?"

**Context Matters**:
- **High Precision Priority**: Spam filter (don't flag real emails as spam)
- **High Recall Priority**: Cancer screening (can't miss any cases)
- **Balance Both (F1)**: Most balanced datasets

## Practice

**Practice 1**

Using the Wine dataset (`load_wine()`), build a binary classification model to distinguish between class 0 and class 1 wines (filter out class 2 for this exercise).

1. Load the dataset and filter to only include classes 0 and 1
2. Check the class distribution using `value_counts()`
3. Split into train/test sets (80/20, random_state=42, remember to stratify!)
4. Fit a LogisticRegression model
5. Calculate and print: accuracy, precision, recall, and F1 score on the test set
6. Create and display the confusion matrix
7. Answer: Which metric is highest? What does this tell about the model's behavior?

**Hints**:
- Use boolean indexing to filter: `mask = (y == 0) | (y == 1)`
- Import all metrics from `sklearn.metrics`
- Use `confusion_matrix()`, `accuracy_score()`, `precision_score()`, `recall_score()`, `f1_score()`

---

**Practice 2**

Return to the Breast Cancer dataset. A hospital wants to minimize false negatives (missing cancer cases) even if it means more false positives (unnecessary follow-up tests).

1. Build a baseline LogisticRegression model and calculate its confusion matrix and recall score
2. Build a second model with `class_weight='balanced'` to handle any class imbalance
3. Compare the two models by creating a table showing:
   - Accuracy, Precision, Recall, F1 Score
   - Number of false positives (FP) and false negatives (FN)
4. Which model has higher recall? Which has higher precision?
5. For the baseline model, manually adjust the decision threshold to 0.3 (instead of 0.5) by:
   - Getting probabilities with `predict_proba()`
   - Creating predictions where `y_pred_adjusted = (y_proba[:, 1] >= 0.3).astype(int)`
   - Calculating metrics for this adjusted-threshold model
6. Write 3-4 sentences: Which model should the hospital use and why? Consider the medical context where missing cancer is far worse than unnecessary tests.

**Bonus**: Create a function that tests different thresholds (0.1, 0.2, 0.3, ..., 0.9) and plots how precision and recall change.

---

**Practice 3**

Build a fraud detection system for credit card transactions. Historical data shows only 0.5% of transactions are fraudulent (highly imbalanced).

**Part 1: Generate Synthetic Data**

Create a synthetic dataset with realistic patterns (~10,000 transactions, 0.5% fraud rate). Features:
- `amount`: Transaction amount in dollars (fraud tends higher: normal ~$50, fraud ~$300)
- `time_since_last`: Minutes since last transaction (fraud tends shorter: normal ~120, fraud ~30)
- `distance_km`: Distance from previous transaction (fraud tends longer: normal ~5, fraud ~500)
- `is_online`: Boolean (fraud is more often online: normal 30% online, fraud 70% online)

Use `np.random.normal()` for continuous features and `np.random.choice()` for categorical.

**Part 2: Build and Compare Three Models**

1. **Baseline**: Standard logistic regression
2. **Model 2**: Logistic regression with `class_weight='balanced'`
3. **Model 3**: Choose an improvement (different threshold, feature engineering like `amount_per_minute = amount / (time_since_last + 1)`, or both)

**Part 3: Comprehensive Evaluation**

For each model, calculate:
- Confusion matrix (visualized as heatmap)
- Accuracy, Precision, Recall, F1 Score
- Number of missed frauds (false negatives)
- Number of false alarms (false positives)
- **Expected cost**: Each missed fraud costs $200; each false alarm costs $5 in investigation time
  - Total cost = (FN × $200) + (FP × $5)

**Part 4: Write a Report**

Create a markdown report with:
1. **Problem Statement**: Why this is challenging (class imbalance, asymmetric costs)
2. **Data Generation**: Briefly describe the synthetic data methodology
3. **Model Comparison Table**: Side-by-side metrics for all three models
4. **Cost-Benefit Analysis**: Which model minimizes expected cost? Show calculations.
5. **Recommendation**: Which model to deploy in production? Justify with both metrics and business reasoning.
6. **Threshold Exploration**: Plot precision vs recall for different thresholds (0.1 to 0.9) for the best model

## Solutions

**Solution 1**

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
import pandas as pd
import numpy as np

# Load wine dataset
data = load_wine()
X = data.data
y = data.target

# Filter to only classes 0 and 1
mask = (y == 0) | (y == 1)
X = X[mask]
y = y[mask]

# Check class distribution
print("Class distribution:")
print(pd.Series(y).value_counts())
print("\nClass balance:")
print(pd.Series(y).value_counts(normalize=True))

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("EVALUATION METRICS")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("="*50)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Analysis
metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}
highest_metric = max(metrics_dict, key=metrics_dict.get)
print(f"\nHighest metric: {highest_metric} ({metrics_dict[highest_metric]:.4f})")
print(f"\nThis indicates the model has very few errors overall, with particularly")
print(f"strong performance on {highest_metric.lower()}. The high values across all")
print(f"metrics suggest the two wine classes are well-separated in feature space.")
```

**Brief explanation**: The solution filters the dataset to binary classes, uses stratified splitting to preserve class balance, trains a logistic regression model, and calculates all four key metrics. The confusion matrix shows where errors occur. Typically, all metrics will be high (>0.95) for this dataset because wine classes 0 and 1 are quite distinct.

---

**Solution 2**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
import pandas as pd
import numpy as np

# Load data
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Baseline model
model_baseline = LogisticRegression(random_state=42, max_iter=10000)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
cm_baseline = confusion_matrix(y_test, y_pred_baseline)

# Balanced model
model_balanced = LogisticRegression(
    random_state=42, max_iter=10000, class_weight='balanced'
)
model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

# Threshold-adjusted model (baseline with threshold=0.3)
y_proba_baseline = model_baseline.predict_proba(X_test)
y_pred_adjusted = (y_proba_baseline[:, 1] >= 0.3).astype(int)
cm_adjusted = confusion_matrix(y_test, y_pred_adjusted)

# Create comparison table
def calc_metrics(y_true, y_pred, cm):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'False Positives': cm[0, 1],
        'False Negatives': cm[1, 0]
    }

results = pd.DataFrame({
    'Baseline': calc_metrics(y_test, y_pred_baseline, cm_baseline),
    'Balanced': calc_metrics(y_test, y_pred_balanced, cm_balanced),
    'Adjusted (0.3)': calc_metrics(y_test, y_pred_adjusted, cm_adjusted)
})

print("Model Comparison:")
print(results.round(4))

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("The hospital should use the threshold-adjusted model (or balanced model).")
print("Medical diagnosis prioritizes recall over precision because missing a cancer")
print("case (false negative) has severe consequences, while a false positive only")
print("results in additional testing. The adjusted threshold model maximizes recall,")
print("catching more cancer cases at the cost of more false alarms, which is the")
print("appropriate trade-off for this medical context.")
print("="*70)

# Bonus: Threshold exploration
thresholds = np.arange(0.1, 1.0, 0.1)
precisions = []
recalls = []

for thresh in thresholds:
    y_pred_thresh = (y_proba_baseline[:, 1] >= thresh).astype(int)
    precisions.append(precision_score(y_test, y_pred_thresh))
    recalls.append(recall_score(y_test, y_pred_thresh))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, marker='o', label='Precision', linewidth=2)
plt.plot(thresholds, recalls, marker='s', label='Recall', linewidth=2)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision vs Recall at Different Thresholds', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_exploration.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Brief explanation**: The solution compares three approaches: baseline logistic regression, balanced class weights, and manual threshold adjustment. The comparison table shows that lowering the threshold increases recall (fewer missed cancers) at the cost of lower precision (more false alarms). For medical diagnosis, this trade-off is appropriate. The bonus visualization shows how precision and recall change across different thresholds.

---

**Solution 3**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# Part 1: Generate synthetic fraud data
np.random.seed(42)

n_transactions = 10000
fraud_rate = 0.005
n_fraud = int(n_transactions * fraud_rate)
n_normal = n_transactions - n_fraud

# Normal transactions
amount_normal = np.random.normal(50, 20, n_normal)
time_normal = np.random.normal(120, 40, n_normal)
distance_normal = np.random.normal(5, 3, n_normal)
online_normal = np.random.choice([0, 1], n_normal, p=[0.7, 0.3])

# Fraudulent transactions
amount_fraud = np.random.normal(300, 100, n_fraud)
time_fraud = np.random.normal(30, 10, n_fraud)
distance_fraud = np.random.normal(500, 200, n_fraud)
online_fraud = np.random.choice([0, 1], n_fraud, p=[0.3, 0.7])

# Combine into dataset
X = pd.DataFrame({
    'amount': np.concatenate([amount_normal, amount_fraud]),
    'time_since_last': np.concatenate([time_normal, time_fraud]),
    'distance_km': np.concatenate([distance_normal, distance_fraud]),
    'is_online': np.concatenate([online_normal, online_fraud])
})
y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

# Ensure positive values
X['amount'] = np.abs(X['amount'])
X['time_since_last'] = np.abs(X['time_since_last'])
X['distance_km'] = np.abs(X['distance_km'])

print("Dataset created:")
print(f"Total transactions: {len(X)}")
print(f"Fraud rate: {y.mean():.4f}")
print(f"\nClass distribution:")
print(pd.Series(y).value_counts())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Part 2: Build three models

# Model 1: Baseline
model1 = LogisticRegression(random_state=42, max_iter=10000)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

# Model 2: Balanced class weights
model2 = LogisticRegression(
    random_state=42, max_iter=10000, class_weight='balanced'
)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

# Model 3: Feature engineering + adjusted threshold
X_train_eng = X_train.copy()
X_test_eng = X_test.copy()
X_train_eng['amount_per_minute'] = X_train_eng['amount'] / (X_train_eng['time_since_last'] + 1)
X_test_eng['amount_per_minute'] = X_test_eng['amount'] / (X_test_eng['time_since_last'] + 1)

model3 = LogisticRegression(
    random_state=42, max_iter=10000, class_weight='balanced'
)
model3.fit(X_train_eng, y_train)
y_proba3 = model3.predict_proba(X_test_eng)
y_pred3 = (y_proba3[:, 1] >= 0.3).astype(int)  # Lower threshold

# Part 3: Comprehensive evaluation

def evaluate_model(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    cost = (fn * 200) + (fp * 5)

    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Total Cost': cost
    }

results = pd.DataFrame([
    evaluate_model(y_test, y_pred1, 'Baseline'),
    evaluate_model(y_test, y_pred2, 'Balanced'),
    evaluate_model(y_test, y_pred3, 'Engineered + Threshold')
])

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(results.to_string(index=False))
print("="*70)

# Visualize confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
cms = [
    confusion_matrix(y_test, y_pred1),
    confusion_matrix(y_test, y_pred2),
    confusion_matrix(y_test, y_pred3)
]
titles = ['Baseline', 'Balanced', 'Engineered + Threshold']

for ax, cm, title in zip(axes, cms, titles):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('fraud_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# Part 4: Report

print("\n" + "="*70)
print("REPORT")
print("="*70)
print("\n1. PROBLEM STATEMENT")
print("Fraud detection is challenging due to extreme class imbalance (0.5% fraud)")
print("and asymmetric costs. Missing fraud ($200 cost) is 40x more expensive than")
print("false alarms ($5 cost). Standard accuracy is misleading for this problem.")

print("\n2. DATA GENERATION")
print("Synthetic dataset with 10,000 transactions (50 fraudulent). Fraudulent")
print("transactions have higher amounts, shorter times between transactions,")
print("longer distances, and higher online rates.")

print("\n3. MODEL COMPARISON")
print(results[['Model', 'Recall', 'FN', 'FP', 'Total Cost']].to_string(index=False))

print("\n4. COST-BENEFIT ANALYSIS")
best_model_idx = results['Total Cost'].idxmin()
best_model = results.iloc[best_model_idx]
print(f"Best model: {best_model['Model']}")
print(f"Total cost: ${best_model['Total Cost']:.2f}")
print(f"Missed frauds: {int(best_model['FN'])}")
print(f"False alarms: {int(best_model['FP'])}")

print("\n5. RECOMMENDATION")
print(f"Deploy the {best_model['Model']} model. It minimizes total cost by")
print(f"prioritizing recall (catching fraud) over precision (avoiding false alarms).")
print(f"The cost analysis shows this is the optimal business decision.")
print("="*70)

# Threshold exploration for best model
thresholds = np.arange(0.1, 1.0, 0.1)
precisions = []
recalls = []

for thresh in thresholds:
    y_pred_thresh = (y_proba3[:, 1] >= thresh).astype(int)
    precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_thresh))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, marker='o', label='Precision', linewidth=2)
plt.plot(thresholds, recalls, marker='s', label='Recall', linewidth=2)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision-Recall Trade-off (Best Model)', fontsize=14, weight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fraud_threshold_exploration.png', dpi=150, bbox_inches='tight')
plt.show()
```

**Brief explanation**: This comprehensive solution creates synthetic fraud data with realistic patterns, builds and compares three models with different approaches, and performs a cost-benefit analysis. The engineered features (amount per minute) capture fraud patterns better. The lower threshold (0.3) increases recall to catch more fraud. The cost analysis shows which model minimizes business losses, which is the key metric for deployment decisions rather than just accuracy or F1 score.

## Key Takeaways

- **Classification predicts discrete categories (classes), not continuous values**. The fundamental distinction from regression is that the output space is finite and discrete—boundaries are drawn between classes, not fitting a curve through points.

- **Logistic regression uses the sigmoid function to convert linear combinations of features into probabilities between 0 and 1**, then applies a threshold (usually 0.5) to make discrete predictions. Despite its name, logistic regression is a classification algorithm, not a regression algorithm.

- **The confusion matrix is the foundation of all classification metrics**. Every metric—accuracy, precision, recall, F1—is calculated from the four values in the confusion matrix: True Positives, True Negatives, False Positives, and False Negatives. Always create a confusion matrix before trusting summary metrics.

- **Accuracy is misleading for imbalanced datasets**. A model that always predicts the majority class can have high accuracy while being completely useless. For imbalanced data, use precision, recall, and F1 score instead. Always check class distribution first.

- **Precision and recall represent different error trade-offs**: Precision minimizes false positives (Type I errors), while recall minimizes false negatives (Type II errors). The right metric depends on the business context—spam filters prioritize precision; cancer screening prioritizes recall. F1 score balances both when errors have similar costs.

- **Multi-class classification extends naturally from binary classification** with the same sklearn API. The model outputs k probabilities (one per class) that sum to 1.0, and the class with the highest probability is predicted. The confusion matrix becomes k×k, but interpretation remains the same.

- **Class imbalance requires special handling** through stratified splitting, appropriate metric selection, class weighting (`class_weight='balanced'`), or threshold adjustment. Never train a classification model on imbalanced data without considering these strategies.

**Next:** Chapter 19.2 covers advanced classification algorithms including decision trees, random forests, and support vector machines.

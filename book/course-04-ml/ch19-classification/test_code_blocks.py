"""
Code Review Test Runner for Chapter 19: Classification Fundamentals
Tests all code blocks in sequence to verify they execute correctly.
"""

import sys
import traceback

# Track test results
results = {
    'total': 0,
    'passed': 0,
    'failed': 0,
    'failures': []
}

def test_block(block_num, description, code_func):
    """Test a single code block"""
    results['total'] += 1
    print(f"\n{'='*70}")
    print(f"Testing Block {block_num}: {description}")
    print('='*70)
    try:
        code_func()
        results['passed'] += 1
        print(f"✓ Block {block_num} PASSED")
        return True
    except Exception as e:
        results['failed'] += 1
        error_msg = f"Block {block_num}: {str(e)}"
        results['failures'].append({
            'block': block_num,
            'description': description,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        print(f"✗ Block {block_num} FAILED: {str(e)}")
        return False

# =============================================================================
# BLOCK 1: Sigmoid Function Visualization
# =============================================================================
def block_1():
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

    # Save instead of show for testing
    plt.savefig('/tmp/sigmoid_function.png', dpi=150, bbox_inches='tight')
    plt.close()

test_block(1, "Sigmoid Function Visualization", block_1)

# =============================================================================
# BLOCK 2: Binary Classification with Logistic Regression
# =============================================================================
def block_2():
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
    plt.savefig('/tmp/decision_boundary_2d.png', dpi=150, bbox_inches='tight')
    plt.close()

test_block(2, "Binary Classification with Logistic Regression", block_2)

# =============================================================================
# BLOCK 3: Confusion Matrix and Evaluation Metrics
# =============================================================================
def block_3():
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
    plt.savefig('/tmp/confusion_matrix_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

test_block(3, "Confusion Matrix and Evaluation Metrics", block_3)

# =============================================================================
# BLOCK 4: Multi-Class Classification
# =============================================================================
def block_4():
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
    plt.savefig('/tmp/multiclass_classification.png', dpi=150, bbox_inches='tight')
    plt.close()

test_block(4, "Multi-Class Classification", block_4)

# =============================================================================
# BLOCK 5: Common Pitfall Example - Class Distribution Check
# =============================================================================
def block_5():
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score

    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # This is example code from the "Common Pitfalls" section
    # Check class distribution first!
    print(pd.Series(y).value_counts(normalize=True))

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # If imbalanced, focus on precision/recall instead of accuracy
    print(classification_report(y_test, y_pred))

    # Or use F1 score as your primary metric
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

test_block(5, "Common Pitfall Example - Class Distribution Check", block_5)

# =============================================================================
# BLOCK 6: Common Pitfall Example - Stratified Split Verification
# =============================================================================
def block_6():
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    # Load data
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # RIGHT: Stratified split preserves class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Verify the distribution is preserved
    print("Training set distribution:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("\nTest set distribution:")
    print(pd.Series(y_test).value_counts(normalize=True))

test_block(6, "Common Pitfall Example - Stratified Split Verification", block_6)

# =============================================================================
# Print Summary
# =============================================================================
print("\n" + "="*70)
print("CODE REVIEW SUMMARY")
print("="*70)
print(f"Total blocks tested: {results['total']}")
print(f"Passed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Success rate: {results['passed']/results['total']*100:.1f}%")

if results['failures']:
    print("\n" + "="*70)
    print("FAILURES:")
    print("="*70)
    for failure in results['failures']:
        print(f"\nBlock {failure['block']}: {failure['description']}")
        print(f"Error: {failure['error']}")
        print("Traceback:")
        print(failure['traceback'])

sys.exit(0 if results['failed'] == 0 else 1)

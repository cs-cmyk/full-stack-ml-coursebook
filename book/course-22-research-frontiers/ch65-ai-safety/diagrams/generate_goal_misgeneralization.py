# Demonstrating goal misgeneralization with spurious correlations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

class GoalMisgeneralizationDemo:
    """
    Demonstrates how models can learn spurious correlations during training
    that don't generalize to deployment (goal misgeneralization)
    """

    def create_data_with_spurious_correlation(self, n_samples=1000):
        """
        Create dataset where class label correlates with a spurious feature
        during training but not in deployment
        """
        # True features that determine the class
        X_true, y = make_classification(n_samples=n_samples, n_features=4,
                                        n_informative=4, n_redundant=0,
                                        n_clusters_per_class=2, random_state=42)

        # Add spurious feature that correlates with class in training
        # but is actually just random noise
        spurious_feature = np.random.normal(0, 1, n_samples)

        # Make spurious feature correlate with class in training data
        # Class 0: spurious feature tends negative
        # Class 1: spurious feature tends positive
        spurious_feature[y == 0] += -1.5  # Shift negative for class 0
        spurious_feature[y == 1] += 1.5   # Shift positive for class 1

        # Combine true and spurious features
        X_with_spurious = np.column_stack([X_true, spurious_feature])

        return X_true, X_with_spurious, y, spurious_feature

    def create_deployment_data(self, n_samples=200):
        """
        Create deployment data where spurious correlation is broken
        """
        X_true, y = make_classification(n_samples=n_samples, n_features=4,
                                        n_informative=4, n_redundant=0,
                                        n_clusters_per_class=2, random_state=123)

        # In deployment, spurious feature is UNCORRELATED with class
        spurious_feature = np.random.normal(0, 1, n_samples)
        # Intentionally break the correlation
        spurious_feature[y == 0] += 0.2   # Minimal shift
        spurious_feature[y == 1] += -0.2

        X_with_spurious = np.column_stack([X_true, spurious_feature])

        return X_true, X_with_spurious, y, spurious_feature

# Create demonstration
demo = GoalMisgeneralizationDemo()

# Generate training data (with spurious correlation)
X_true_train, X_spurious_train, y_train, spurious_train = \
    demo.create_data_with_spurious_correlation(n_samples=1000)

# Generate deployment data (spurious correlation broken)
X_true_deploy, X_spurious_deploy, y_deploy, spurious_deploy = \
    demo.create_deployment_data(n_samples=200)

# Train two models:
# Model 1: Uses only true features (aligned with intended goal)
model_true = LogisticRegression(random_state=42)
model_true.fit(X_true_train, y_train)

# Model 2: Uses true + spurious features (may learn wrong objective)
model_spurious = LogisticRegression(random_state=42)
model_spurious.fit(X_spurious_train, y_train)

# Evaluate on training distribution
train_acc_true = accuracy_score(y_train, model_true.predict(X_true_train))
train_acc_spurious = accuracy_score(y_train, model_spurious.predict(X_spurious_train))

# Evaluate on deployment distribution (distribution shift)
deploy_acc_true = accuracy_score(y_deploy, model_true.predict(X_true_deploy))
deploy_acc_spurious = accuracy_score(y_deploy, model_spurious.predict(X_spurious_deploy))

# Visualize spurious feature distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top row: Training distribution
ax1, ax2 = axes[0]
ax1.hist(spurious_train[y_train==0], bins=30, alpha=0.7, label='Class 0', color='#1976D2')
ax1.hist(spurious_train[y_train==1], bins=30, alpha=0.7, label='Class 1', color='#D32F2F')
ax1.set_title('Training: Spurious Feature Distribution\n(Strong correlation with class)',
              fontsize=12, fontweight='bold')
ax1.set_xlabel('Spurious Feature Value', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.legend()
ax1.axvline(0, color='black', linestyle='--', alpha=0.5)

# Feature importance for model trained with spurious feature
feature_names = ['True 1', 'True 2', 'True 3', 'True 4', 'Spurious']
importances = np.abs(model_spurious.coef_[0])
colors = ['#2E7D32']*4 + ['#C62828']
ax2.barh(feature_names, importances, color=colors)
ax2.set_title('Training: Learned Feature Importance\n(Model learns spurious correlation)',
              fontsize=12, fontweight='bold')
ax2.set_xlabel('Absolute Coefficient', fontsize=10)

# Bottom row: Deployment distribution
ax3, ax4 = axes[1]
ax3.hist(spurious_deploy[y_deploy==0], bins=30, alpha=0.7, label='Class 0', color='#1976D2')
ax3.hist(spurious_deploy[y_deploy==1], bins=30, alpha=0.7, label='Class 1', color='#D32F2F')
ax3.set_title('Deployment: Spurious Feature Distribution\n(Correlation broken)',
              fontsize=12, fontweight='bold')
ax3.set_xlabel('Spurious Feature Value', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.legend()
ax3.axvline(0, color='black', linestyle='--', alpha=0.5)

# Performance comparison
models = ['True Features\nOnly', 'True + Spurious\nFeatures']
train_accs = [train_acc_true, train_acc_spurious]
deploy_accs = [deploy_acc_true, deploy_acc_spurious]

x = np.arange(len(models))
width = 0.35

bars1 = ax4.bar(x - width/2, train_accs, width, label='Training', color='#388E3C')
bars2 = ax4.bar(x + width/2, deploy_accs, width, label='Deployment', color='#C62828')

ax4.set_ylabel('Accuracy', fontsize=10)
ax4.set_title('Performance: Goal Misgeneralization\n(Spurious correlations hurt deployment)',
              fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=10)
ax4.legend()
ax4.set_ylim([0.5, 1.0])
ax4.axhline(0.8, color='black', linestyle=':', alpha=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('/home/chirag/ds-book/book/course-22/ch65/diagrams/goal_misgeneralization.png', dpi=150, bbox_inches='tight')
print("Generated: goal_misgeneralization.png")
print(f"Model with spurious: Training={train_acc_spurious:.3f}, Deployment={deploy_acc_spurious:.3f}")
print(f"Model without spurious: Training={train_acc_true:.3f}, Deployment={deploy_acc_true:.3f}")

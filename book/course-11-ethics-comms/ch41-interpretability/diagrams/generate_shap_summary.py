import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Use TreeExplainer for tree-based models (fast)
explainer = shap.TreeExplainer(rf_model)

# Compute SHAP values for test set
shap_values = explainer.shap_values(X_test)

# SHAP returns array for each class; we want class 1 (malignant)
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]  # Class 1 (malignant)
else:
    # For newer versions, shap_values might be 3D (samples, features, classes)
    if len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

# Create beeswarm summary plot showing SHAP values for all test instances
shap.summary_plot(
    shap_values_class1,
    X_test,
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: shap_summary.png")

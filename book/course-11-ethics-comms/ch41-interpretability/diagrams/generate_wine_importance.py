import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Load Wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train RandomForest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# Permutation importance
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': X.columns,
    'perm_importance': perm_imp.importances_mean
}).sort_values('perm_importance', ascending=False)

# SHAP importance
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# SHAP returns list of arrays for multi-class; compute mean absolute SHAP across all classes
if isinstance(shap_values, list):
    # For list format: average over classes, then instances
    shap_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    # For 3D array format (samples, features, classes)
    if len(shap_values.shape) == 3:
        shap_abs = np.abs(shap_values).mean(axis=2).mean(axis=0)
    else:
        shap_abs = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    'feature': X.columns,
    'shap_importance': shap_abs
}).sort_values('shap_importance', ascending=False)

# Side-by-side bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Permutation importance
top_perm = perm_df.head(10)
axes[0].barh(range(len(top_perm)), top_perm['perm_importance'], color='#2196F3', alpha=0.8, edgecolor='black')
axes[0].set_yticks(range(len(top_perm)))
axes[0].set_yticklabels(top_perm['feature'], fontsize=11)
axes[0].set_xlabel('Importance', fontweight='bold', fontsize=12)
axes[0].set_title('Permutation Importance (Top 10)', fontweight='bold', fontsize=13)
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# SHAP importance
top_shap = shap_df.head(10)
axes[1].barh(range(len(top_shap)), top_shap['shap_importance'], color='#FF9800', alpha=0.8, edgecolor='black')
axes[1].set_yticks(range(len(top_shap)))
axes[1].set_yticklabels(top_shap['feature'], fontsize=11)
axes[1].set_xlabel('Importance (mean |SHAP|)', fontweight='bold', fontsize=12)
axes[1].set_title('SHAP Importance (Top 10)', fontweight='bold', fontsize=13)
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('wine_importance_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: wine_importance_comparison.png")

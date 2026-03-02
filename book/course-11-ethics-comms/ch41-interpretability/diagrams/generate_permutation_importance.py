import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
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

# Train black-box model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Compute permutation importance on test set
perm_importance = permutation_importance(
    rf_model, X_test, y_test,
    n_repeats=10,  # Repeat shuffling 10 times for stability
    random_state=42,
    scoring='roc_auc'
)

# Create DataFrame of results
perm_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

# Visualize top 10
fig, ax = plt.subplots(figsize=(10, 6))
top_10 = perm_df.head(10)
ax.barh(range(len(top_10)), top_10['importance_mean'],
        xerr=top_10['importance_std'], color='#2196F3', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'], fontsize=12)
ax.invert_yaxis()
ax.set_xlabel('Importance (AUC drop when shuffled)', fontweight='bold', fontsize=12)
ax.set_title('Permutation Feature Importance (Random Forest)', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: permutation_importance.png")

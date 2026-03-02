import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Load Diabetes dataset
diabetes = load_diabetes()
X_diab = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_diab = diabetes.target

# Train GradientBoosting
gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
gb.fit(X_diab, y_diab)

# PDP + ICE for 'bmi'
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(
    gb, X_diab, features=['bmi'],
    kind='both',
    ice_lines_kw={'alpha': 0.05, 'linewidth': 0.5, 'color': '#607D8B'},  # Gray
    pd_line_kw={'color': '#F44336', 'linewidth': 3},  # Red
    ax=ax, subsample=200, random_state=42
)
ax.set_title('PDP and ICE for BMI (Diabetes Progression)', fontweight='bold', fontsize=14)
ax.set_ylabel('Predicted Disease Progression', fontweight='bold', fontsize=12)
ax.set_xlabel('BMI (standardized)', fontweight='bold', fontsize=12)
ax.legend(['ICE (individuals)', 'PDP (average)'], loc='upper left')
plt.tight_layout()
plt.savefig('diabetes_pdp_ice.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: diabetes_pdp_ice.png")

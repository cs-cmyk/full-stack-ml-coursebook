import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Load data
housing = fetch_california_housing()
X_house = pd.DataFrame(housing.data, columns=housing.feature_names)
y_house = housing.target  # Median house value in $100k

# Train/test split
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_house, y_house, test_size=0.3, random_state=42
)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_h_train, y_h_train)

# Create PDP + ICE plot for 'MedInc' (median income)
fig, ax = plt.subplots(figsize=(10, 6))

disp = PartialDependenceDisplay.from_estimator(
    gb_model,
    X_h_test,
    features=['MedInc'],
    kind='both',  # Both PDP and ICE
    ice_lines_kw={'alpha': 0.1, 'linewidth': 0.5, 'color': '#607D8B'},  # Gray ICE lines
    pd_line_kw={'color': '#F44336', 'linewidth': 3, 'label': 'PDP (average)'},  # Red PDP line
    ax=ax,
    random_state=42,
    subsample=200  # Sample 200 ICE lines to avoid clutter
)

ax.set_ylabel('Predicted House Value ($100k)', fontweight='bold', fontsize=12)
ax.set_xlabel('MedInc (Median Income, $10k)', fontweight='bold', fontsize=12)
ax.set_title('Partial Dependence (PDP) and Individual Conditional Expectation (ICE)',
             fontweight='bold', fontsize=14)
ax.legend(['ICE (individual)', 'PDP (average)'], loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('pdp_ice_plot.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Generated: pdp_ice_plot.png")

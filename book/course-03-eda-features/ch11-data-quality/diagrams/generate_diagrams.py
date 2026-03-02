"""
Generate all diagrams for Chapter 11: Data Quality and Cleaning
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style and colors
plt.style.use('default')
COLORS = {
    'blue': '#2196F3',
    'green': '#4CAF50',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'gray': '#607D8B'
}

np.random.seed(42)

print("Generating diagrams for Chapter 11: Data Quality and Cleaning")
print("=" * 70)

# ============================================================================
# Diagram 1: Missing Data Types (MCAR, MAR, MNAR)
# ============================================================================
print("\n1. Generating missing_data_types.png...")

n_samples = 100
age = np.random.randint(20, 70, n_samples)
income = 30000 + age * 1000 + np.random.normal(0, 10000, n_samples)

df_demo = pd.DataFrame({
    'Age': age,
    'Income_MCAR': income.copy(),
    'Income_MAR': income.copy(),
    'Income_MNAR': income.copy()
})

# MCAR: Randomly remove 20% of values
mcar_mask = np.random.random(n_samples) < 0.2
df_demo.loc[mcar_mask, 'Income_MCAR'] = np.nan

# MAR: Remove income for older people
mar_mask = (age > 50) & (np.random.random(n_samples) < 0.4)
df_demo.loc[mar_mask, 'Income_MAR'] = np.nan

# MNAR: Remove high incomes
mnar_mask = (income > 60000) & (np.random.random(n_samples) < 0.5)
df_demo.loc[mnar_mask, 'Income_MNAR'] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# MCAR visualization
axes[0].scatter(df_demo['Age'], df_demo['Income_MCAR'], alpha=0.6,
                label='Observed', color=COLORS['blue'], s=50)
axes[0].scatter(df_demo.loc[df_demo['Income_MCAR'].isna(), 'Age'],
                [20000] * mcar_mask.sum(), color=COLORS['red'],
                marker='x', s=100, label='Missing', linewidths=2)
axes[0].set_title('MCAR: Missing Completely at Random\n(Missing scattered randomly)',
                   fontsize=12, fontweight='bold')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Income', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# MAR visualization
axes[1].scatter(df_demo['Age'], df_demo['Income_MAR'], alpha=0.6,
                label='Observed', color=COLORS['blue'], s=50)
axes[1].scatter(df_demo.loc[df_demo['Income_MAR'].isna(), 'Age'],
                [20000] * mar_mask.sum(), color=COLORS['red'],
                marker='x', s=100, label='Missing', linewidths=2)
axes[1].axvline(x=50, color=COLORS['orange'], linestyle='--',
                alpha=0.7, linewidth=2, label='Age > 50')
axes[1].set_title('MAR: Missing at Random\n(Missing related to Age—observed)',
                   fontsize=12, fontweight='bold')
axes[1].set_xlabel('Age', fontsize=12)
axes[1].set_ylabel('Income', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

# MNAR visualization
axes[2].scatter(df_demo['Age'], df_demo['Income_MNAR'], alpha=0.6,
                label='Observed', color=COLORS['blue'], s=50)
axes[2].scatter(df_demo.loc[df_demo['Income_MNAR'].isna(), 'Age'],
                [20000] * mnar_mask.sum(), color=COLORS['red'],
                marker='x', s=100, label='Missing', linewidths=2)
axes[2].axhline(y=60000, color=COLORS['purple'], linestyle='--',
                alpha=0.7, linewidth=2, label='Income > $60k')
axes[2].set_title('MNAR: Missing Not at Random\n(High earners hide income)',
                   fontsize=12, fontweight='bold')
axes[2].set_xlabel('Age', fontsize=12)
axes[2].set_ylabel('Income', fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('missing_data_types.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✓ Saved missing_data_types.png")

# ============================================================================
# Diagram 2: Quality Issues Visualization
# ============================================================================
print("\n2. Generating quality_issues_visualization.png...")

# Load California Housing dataset
housing = fetch_california_housing()
df_raw = pd.DataFrame(housing.data, columns=housing.feature_names)
df_raw['MedHouseVal'] = housing.target

# Introduce quality issues
df = df_raw.copy()
n = len(df)

# Missing values
mcar_mask_inc = np.random.random(n) < 0.15
mcar_mask_rooms = np.random.random(n) < 0.15
df.loc[mcar_mask_inc, 'MedInc'] = np.nan
df.loc[mcar_mask_rooms, 'AveRooms'] = np.nan

# Duplicates
duplicate_indices = np.random.choice(df.index, 30, replace=False)
df = pd.concat([df, df.loc[duplicate_indices]], ignore_index=True)

# Outliers
outlier_indices = np.random.choice(df.index, 20, replace=False)
df.loc[outlier_indices, 'MedInc'] = df.loc[outlier_indices, 'MedInc'] * 5

# Calculate statistics
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percent': missing_pct
})

Q1 = df['MedInc'].quantile(0.25)
Q3 = df['MedInc'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Missing data bar chart
missing_viz = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
axes[0, 0].bar(range(len(missing_viz)), missing_viz['Missing_Count'],
               color=COLORS['orange'], edgecolor='black', linewidth=1)
axes[0, 0].set_xticks(range(len(missing_viz)))
axes[0, 0].set_xticklabels(missing_viz.index, rotation=45, ha='right', fontsize=11)
axes[0, 0].set_title('Missing Values by Column', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('Count', fontsize=12)
axes[0, 0].grid(axis='y', alpha=0.3)

# Missing data heatmap
sns.heatmap(df.head(100).isnull(), cbar=True, cmap='YlOrRd', ax=axes[0, 1],
            cbar_kws={'label': 'Missing'})
axes[0, 1].set_title('Missing Data Pattern (First 100 Rows)\nYellow/Red = Missing',
                      fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Features', fontsize=12)
axes[0, 1].set_ylabel('Samples', fontsize=12)

# Box plot showing outliers
box_data = df['MedInc'].dropna()
bp = axes[1, 0].boxplot(box_data, vert=True, patch_artist=True,
                         boxprops=dict(facecolor=COLORS['blue'], alpha=0.6),
                         medianprops=dict(color=COLORS['red'], linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
axes[1, 0].axhline(upper_bound, color=COLORS['red'], linestyle='--',
                    linewidth=2, label=f'Upper bound: {upper_bound:.2f}')
axes[1, 0].axhline(lower_bound, color=COLORS['red'], linestyle='--',
                    linewidth=2, label=f'Lower bound: {lower_bound:.2f}')
axes[1, 0].set_ylabel('MedInc', fontsize=12)
axes[1, 0].set_title('Outliers in MedInc (IQR Method)', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xticklabels(['MedInc'], fontsize=12)

# Distribution comparison
axes[1, 1].hist(df_raw['MedInc'], bins=50, alpha=0.6, label='Original',
                color=COLORS['blue'], edgecolor='black', linewidth=0.5)
axes[1, 1].hist(df['MedInc'].dropna(), bins=50, alpha=0.6, label='With Outliers',
                color=COLORS['red'], edgecolor='black', linewidth=0.5)
axes[1, 1].set_xlabel('MedInc', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Distribution: Original vs. Contaminated',
                      fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('quality_issues_visualization.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✓ Saved quality_issues_visualization.png")

# ============================================================================
# Diagram 3: Imputation Comparison
# ============================================================================
print("\n3. Generating imputation_comparison.png...")

# Prepare clean data for imputation comparison
df_clean = df.drop_duplicates()
p01 = df_clean['MedInc'].quantile(0.01)
p99 = df_clean['MedInc'].quantile(0.99)
df_clean['MedInc'] = df_clean['MedInc'].clip(lower=p01, upper=p99)

# Split data
X = df_clean.drop('MedHouseVal', axis=1)
y = df_clean['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compare imputation strategies
strategies = {}

# Mean Imputation
imputer_mean = SimpleImputer(strategy='mean')
X_train_mean = pd.DataFrame(imputer_mean.fit_transform(X_train), columns=X_train.columns)
X_test_mean = pd.DataFrame(imputer_mean.transform(X_test), columns=X_test.columns)
model = LinearRegression()
model.fit(X_train_mean, y_train)
y_pred = model.predict(X_test_mean)
strategies['Mean\nImputation'] = np.sqrt(mean_squared_error(y_test, y_pred))

# Median Imputation
imputer_median = SimpleImputer(strategy='median')
X_train_median = pd.DataFrame(imputer_median.fit_transform(X_train), columns=X_train.columns)
X_test_median = pd.DataFrame(imputer_median.transform(X_test), columns=X_test.columns)
model = LinearRegression()
model.fit(X_train_median, y_train)
y_pred = model.predict(X_test_median)
strategies['Median\nImputation'] = np.sqrt(mean_squared_error(y_test, y_pred))

# KNN Imputation
imputer_knn = KNNImputer(n_neighbors=5)
X_train_knn = pd.DataFrame(imputer_knn.fit_transform(X_train), columns=X_train.columns)
X_test_knn = pd.DataFrame(imputer_knn.transform(X_test), columns=X_test.columns)
model = LinearRegression()
model.fit(X_train_knn, y_train)
y_pred = model.predict(X_test_knn)
strategies['KNN\nImputation'] = np.sqrt(mean_squared_error(y_test, y_pred))

# Visualize results
fig, ax = plt.subplots(figsize=(10, 6))
strategies_list = list(strategies.keys())
rmse_values = list(strategies.values())
best_idx = rmse_values.index(min(rmse_values))
colors = [COLORS['green'] if i == best_idx else COLORS['orange'] for i in range(len(strategies_list))]

bars = ax.bar(strategies_list, rmse_values, color=colors, edgecolor='black',
              linewidth=1.5, alpha=0.8)
ax.set_ylabel('RMSE (lower is better)', fontsize=13, fontweight='bold')
ax.set_title('Comparison of Imputation Strategies', fontsize=14, fontweight='bold')
ax.axhline(min(rmse_values), color=COLORS['green'], linestyle='--',
           alpha=0.4, linewidth=2, label='Best Performance')
ax.set_xlabel('Imputation Strategy', fontsize=13, fontweight='bold')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, rmse_values)):
    height = bar.get_height()
    label_text = f'{val:.4f}'
    if i == best_idx:
        label_text += '\n★ BEST'
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.003,
            label_text,
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(rmse_values) * 1.15)

plt.tight_layout()
plt.savefig('imputation_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("   ✓ Saved imputation_comparison.png")

print("\n" + "=" * 70)
print("All diagrams generated successfully!")
print("=" * 70)

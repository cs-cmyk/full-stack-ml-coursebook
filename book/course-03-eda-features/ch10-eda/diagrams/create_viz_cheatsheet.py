"""
Create Visualization Cheat Sheet for EDA Chapter
Shows 9 common plot types in a 3x3 grid with examples
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Generate sample data
n = 100
data = pd.DataFrame({
    'numeric_1': np.random.normal(50, 10, n),
    'numeric_2': np.random.normal(30, 8, n),
    'category': np.random.choice(['A', 'B', 'C'], n),
    'binary': np.random.choice([0, 1], n)
})

# Create figure with 3x3 grid
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Define color palette
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

# Row 1: UNIVARIATE PLOTS
# 1. Histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(data['numeric_1'], bins=20, color=colors[0], alpha=0.7, edgecolor='black')
ax1.set_title('HISTOGRAM', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Value', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.text(0.5, -0.35, 'Use Case: Distribution shape\nWhen: Single numeric variable',
         transform=ax1.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax1.grid(axis='y', alpha=0.3)

# 2. Box Plot
ax2 = fig.add_subplot(gs[0, 1])
bp = ax2.boxplot(data['numeric_1'], patch_artist=True, widths=0.5)
for patch in bp['boxes']:
    patch.set_facecolor(colors[1])
ax2.set_title('BOX PLOT', fontsize=14, fontweight='bold', pad=10)
ax2.set_ylabel('Value', fontsize=11)
ax2.set_xticklabels(['Variable'])
ax2.text(0.5, -0.35, 'Use Case: Quartiles & outliers\nWhen: Detect extreme values',
         transform=ax2.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax2.grid(axis='y', alpha=0.3)

# 3. Count Plot
ax3 = fig.add_subplot(gs[0, 2])
category_counts = data['category'].value_counts()
bars = ax3.bar(category_counts.index, category_counts.values, color=colors[2:], alpha=0.7)
ax3.set_title('COUNT PLOT', fontsize=14, fontweight='bold', pad=10)
ax3.set_xlabel('Category', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.text(0.5, -0.35, 'Use Case: Categorical frequencies\nWhen: Compare class sizes',
         transform=ax3.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax3.grid(axis='y', alpha=0.3)

# Row 2: BIVARIATE PLOTS
# 4. Scatter Plot
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(data['numeric_1'], data['numeric_2'], alpha=0.6, s=40, c=colors[0])
ax4.set_title('SCATTER PLOT', fontsize=14, fontweight='bold', pad=10)
ax4.set_xlabel('Feature 1', fontsize=11)
ax4.set_ylabel('Feature 2', fontsize=11)
ax4.text(0.5, -0.35, 'Use Case: Numeric relationships\nWhen: Check correlation, linearity',
         transform=ax4.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax4.grid(alpha=0.3)

# 5. Box Plot by Group
ax5 = fig.add_subplot(gs[1, 1])
data_grouped = [data[data['category']==cat]['numeric_1'].values for cat in ['A', 'B', 'C']]
bp2 = ax5.boxplot(data_grouped, labels=['A', 'B', 'C'], patch_artist=True)
for patch, color in zip(bp2['boxes'], colors[2:]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_title('BOX BY GROUP', fontsize=14, fontweight='bold', pad=10)
ax5.set_xlabel('Category', fontsize=11)
ax5.set_ylabel('Value', fontsize=11)
ax5.text(0.5, -0.35, 'Use Case: Compare distributions\nWhen: Numeric split by category',
         transform=ax5.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax5.grid(axis='y', alpha=0.3)

# 6. Heatmap
ax6 = fig.add_subplot(gs[1, 2])
corr_data = data[['numeric_1', 'numeric_2', 'binary']].corr()
im = ax6.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax6.set_xticks(np.arange(len(corr_data.columns)))
ax6.set_yticks(np.arange(len(corr_data.columns)))
ax6.set_xticklabels(['N1', 'N2', 'B'], fontsize=10)
ax6.set_yticklabels(['N1', 'N2', 'B'], fontsize=10)
# Add correlation values
for i in range(len(corr_data.columns)):
    for j in range(len(corr_data.columns)):
        text = ax6.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)
ax6.set_title('HEATMAP', fontsize=14, fontweight='bold', pad=10)
ax6.text(0.5, -0.35, 'Use Case: Correlation matrix\nWhen: Many feature relationships',
         transform=ax6.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Row 3: MULTIVARIATE PLOTS
# 7. Pair Plot (simplified representation)
ax7 = fig.add_subplot(gs[2, 0])
# Create mini scatter plots
np.random.seed(42)
x1 = np.random.normal(0, 1, 30)
y1 = x1 + np.random.normal(0, 0.3, 30)
ax7.scatter(x1, y1, alpha=0.6, s=20, c=colors[0], label='Var1-Var2')
x2 = np.random.normal(2, 1, 30)
y2 = -x2 + np.random.normal(0, 0.3, 30)
ax7.scatter(x2, y2, alpha=0.6, s=20, c=colors[1], label='Var2-Var3')
ax7.set_title('PAIR PLOT', fontsize=14, fontweight='bold', pad=10)
ax7.set_xlabel('Variable X', fontsize=11)
ax7.set_ylabel('Variable Y', fontsize=11)
ax7.legend(fontsize=9)
ax7.text(0.5, -0.35, 'Use Case: All pairwise relationships\nWhen: Explore many features at once',
         transform=ax7.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax7.grid(alpha=0.3)

# 8. Faceted Plot
ax8 = fig.add_subplot(gs[2, 1])
for i, cat in enumerate(['A', 'B', 'C']):
    subset = data[data['category'] == cat]['numeric_1']
    ax8.hist(subset, bins=15, alpha=0.5, label=cat, color=colors[i+2])
ax8.set_title('FACETED PLOT', fontsize=14, fontweight='bold', pad=10)
ax8.set_xlabel('Value', fontsize=11)
ax8.set_ylabel('Frequency', fontsize=11)
ax8.legend(title='Category', fontsize=9)
ax8.text(0.5, -0.35, 'Use Case: Compare by third variable\nWhen: Show distribution split by groups',
         transform=ax8.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax8.grid(axis='y', alpha=0.3)

# 9. Correlation Matrix (full)
ax9 = fig.add_subplot(gs[2, 2])
# Create a larger correlation matrix
np.random.seed(42)
corr_matrix = np.random.rand(5, 5)
corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
np.fill_diagonal(corr_matrix, 1)
im2 = ax9.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')
ax9.set_xticks(np.arange(5))
ax9.set_yticks(np.arange(5))
ax9.set_xticklabels(['F1', 'F2', 'F3', 'F4', 'F5'], fontsize=10)
ax9.set_yticklabels(['F1', 'F2', 'F3', 'F4', 'F5'], fontsize=10)
ax9.set_title('CORRELATION MATRIX', fontsize=14, fontweight='bold', pad=10)
ax9.text(0.5, -0.35, 'Use Case: Feature multicollinearity\nWhen: Identify redundant features',
         transform=ax9.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Add row labels on the left
fig.text(0.02, 0.82, 'UNIVARIATE\nOne variable', fontsize=13, fontweight='bold',
         rotation=90, va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='#4CAF50', alpha=0.3))
fig.text(0.02, 0.50, 'BIVARIATE\nTwo variables', fontsize=13, fontweight='bold',
         rotation=90, va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='#FF9800', alpha=0.3))
fig.text(0.02, 0.18, 'MULTIVARIATE\nMany variables', fontsize=13, fontweight='bold',
         rotation=90, va='center', ha='center',
         bbox=dict(boxstyle='round', facecolor='#9C27B0', alpha=0.3))

# Add main title
fig.suptitle('EDA Visualization Cheat Sheet: Choose the Right Plot for Your Question',
             fontsize=18, fontweight='bold', y=0.98)

# Save figure
plt.savefig('/home/chirag/ds-book/book/course-03-eda-features/ch10-eda/diagrams/03_visualization_cheatsheet.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Visualization cheat sheet created successfully!")
print("Saved to: diagrams/03_visualization_cheatsheet.png")

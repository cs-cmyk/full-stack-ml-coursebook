#!/usr/bin/env python3
"""Generate pairplot diagram"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set random seed and style
np.random.seed(42)
sns.set_style('whitegrid')

# Load Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Create pairplot for comprehensive view
pairplot = sns.pairplot(df, hue='target', diag_kind='hist',
                        palette=['#2196F3', '#4CAF50', '#FF9800'],
                        plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'black'},
                        height=2.5)
pairplot.fig.suptitle('Iris Dataset: Pairwise Feature Relationships',
                       y=1.02, fontsize=14, weight='bold')
plt.savefig('pairplot.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated pairplot.png")

#!/usr/bin/env python3
"""Generate log transformation comparison diagram"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load data
np.random.seed(42)
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Before transformation
axes[0].hist(df['AveRooms'], bins=50, alpha=0.7, color='#FF9800', edgecolor='black')
axes[0].set_xlabel('Average Rooms', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title(f'Before Log Transform\nSkew = {df["AveRooms"].skew():.2f}',
                  fontsize=12, weight='bold')
axes[0].grid(True, alpha=0.3)

# After transformation
log_rooms = np.log1p(df['AveRooms'])  # log1p handles zeros safely
axes[1].hist(log_rooms, bins=50, alpha=0.7, color='#4CAF50', edgecolor='black')
axes[1].set_xlabel('Log(Average Rooms)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title(f'After Log Transform\nSkew = {log_rooms.skew():.2f}',
                  fontsize=12, weight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('log_transformation.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Generated log_transformation.png")

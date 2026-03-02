# Chapter 12 Diagrams - Generation Summary

## Overview
All diagrams for Chapter 12: Correlation and Relationships have been generated successfully.

## Generated Diagrams

### 1. correlation_patterns.png
- **Type**: Matplotlib scatter plots (3×3 grid)
- **Purpose**: Educational visualization showing how different correlation coefficients (r) manifest visually
- **Content**: 9 scatter plots showing r values: +1.0, +0.9, +0.7, +0.5, +0.3, 0.0, -0.3, -0.7, -0.95
- **Dimensions**: 14×12 inches at 150 DPI
- **Color palette**: Blue (#2196F3) for data points, red (#F44336) for regression lines
- **File size**: ~346 KB
- **Referenced in**: content.md lines 58-128 (Visual section)

### 2. income_vs_value.png
- **Type**: Matplotlib scatter + hexbin plots (side-by-side)
- **Purpose**: Demonstrate strong positive correlation using California Housing data
- **Content**:
  - Left panel: Scatter plot with regression line (MedInc vs MedHouseVal)
  - Right panel: Hexbin density plot showing data concentration
- **Correlation**: r = 0.688 (strong positive)
- **Dimensions**: 14×5 inches at 150 DPI
- **Color palette**: Blue (#2196F3) for scatter, Blues colormap for hexbin, red (#F44336) for regression line
- **File size**: ~393 KB
- **Referenced in**: content.md lines 210-244 (Code Example section)

### 3. correlation_heatmap.png
- **Type**: Seaborn heatmap
- **Purpose**: Show full correlation matrix for all features in California Housing dataset
- **Content**: 9×9 heatmap with annotations showing pairwise correlations
- **Features**: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, MedHouseVal
- **Dimensions**: 12×9 inches at 150 DPI
- **Color palette**: Coolwarm diverging colormap (blue-white-red)
- **File size**: ~205 KB
- **Referenced in**: content.md lines 298-323 (Code Example section)

## Color Palette Used
Consistent with textbook standards:
- **Blue**: #2196F3 (primary data color)
- **Dark Blue**: #1976D2 (edges, accents)
- **Red**: #F44336 (regression lines, highlights)
- **Colormap**: coolwarm for heatmap, Blues for hexbin

## Technical Specifications
- **Resolution**: 150 DPI (high quality for print and digital)
- **Max width**: All diagrams ≤ 800px effective width
- **Background**: White
- **Font size**: Minimum 12pt for readability
- **Layout**: All use plt.tight_layout() before saving

## Generation Script
Location: `generate_diagrams.py`
- Standalone Python script that can regenerate all diagrams
- Uses random seed (42) for reproducibility
- Includes progress output and error handling

## Dataset Information
- **California Housing Dataset**: 20,640 samples, 9 features
- **Source**: sklearn.datasets.fetch_california_housing()
- **Purpose**: Real-world data for demonstrating correlation analysis

## Notes
- All diagrams follow the textbook's educational style
- Clear axis labels, titles, and legends on all plots
- Annotations include correlation values for clarity
- Heatmap includes color bar with labeled scale
- All figures saved with bbox_inches='tight' to avoid clipping
